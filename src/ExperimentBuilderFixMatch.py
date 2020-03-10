import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import os
import numpy as np
import time
from torch.optim import SGD

from torch.optim.adam import Adam
from torch.optim.lr_scheduler import MultiStepLR, _LRScheduler

from ERF_Scheduler import ERF

from storage_utils import save_statistics
from sklearn.metrics import f1_score, precision_score, recall_score


class ExperimentBuilderFixMatch(nn.Module):
    def __init__(self, network_model, experiment_name, num_epochs, train_data, val_data, train_data_unlabeled,
                 test_data, use_gpu, continue_from_epoch=-1,
                 scheduler=None, optimiser=None, sched_params=None, optim_params=None,
                 threshold=None, lambda_u=None, pretrained_weights_locations=None):
        """
        Initializes an ExperimentBuilder object. Such an object takes care of running training and evaluation of a deep net
        on a given dataset. It also takes care of saving per epoch models and automatically inferring the best val model
        to be used for evaluating the test set metrics.
        :param network_model: A pytorch nn.Module which implements a network architecture.
        :param experiment_name: The name of the experiment. This is used mainly for keeping track of the experiment and creating and directory structure that will be used to save logs, model parameters and other.
        :param num_epochs: Total number of epochs to run the experiment
        :param train_data: An object of the DataProvider type. Contains the training set.
        :param val_data: An object of the DataProvider type. Contains the val set.
        :param test_data: An object of the DataProvider type. Contains the test set.
        :param use_gpu: A boolean indicating whether to use a GPU or not.
        :param continue_from_epoch: An int indicating whether we'll start from scrach (-1) or whether we'll reload a previously saved model of epoch 'continue_from_epoch' and continue training from there.
        """
        super(ExperimentBuilderFixMatch, self).__init__()

        self.experiment_name = experiment_name
        self.model = network_model
        # self.model.reset_parameters()
        self.device = torch.cuda.current_device()

        if torch.cuda.device_count() > 1 and use_gpu:
            self.device = torch.cuda.current_device()
            self.model.to(self.device)
            self.model = nn.DataParallel(module=self.model)
            print('Use Multi GPU', self.device)
        elif torch.cuda.device_count() == 1 and use_gpu:
            self.device = torch.cuda.current_device()
            self.model.to(self.device)  # sends the model from the cpu to the gpu
            print('Use GPU', self.device)
        else:
            print("use CPU")
            self.device = torch.device('cpu')  # sets the device to be CPU
            print(self.device)

        self.ema = EMA(model=self.model, alpha=0.999)

        # re-initialize network parameters
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.train_data_unlabeled = train_data_unlabeled
        self.temp = train_data_unlabeled

        self.loss_lambda_u = lambda_u
        self.threshold = threshold

        if optimiser is None or optimiser == 'Adam':
            self.optimizer = Adam(self.parameters(), amsgrad=False,
                                  weight_decay=optim_params['weight_decay'],
                                  lr=sched_params['lr_max'])
        elif optimiser == 'SGD':
            self.optimizer = SGD(self.parameters(),
                                 lr=sched_params['lr_max'],
                                 momentum=optim_params['momentum'],
                                 nesterov=optim_params['nesterov'],
                                 weight_decay=optim_params['weight_decay'])

        self.scheduler_type = scheduler
        if scheduler == 'ERF':
            self.scheduler = ERF(self.optimizer,
                                 min_lr=sched_params['lr_min'],
                                 alpha=sched_params['erf_alpha'],
                                 beta=sched_params['erf_beta'],
                                 epochs=num_epochs)
        elif scheduler == 'Step':
            self.scheduler = MultiStepLR(self.optimizer,
                                         milestones=[30, 60],
                                         gamma=0.1)
        elif scheduler == 'Cos':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                  T_max=num_epochs,
                                                                  eta_min=0.00001)
        elif scheduler == 'FixMatchCos':
            self.scheduler = WarmupCosineLrScheduler(self.optimizer, max_iter=num_epochs*len(train_data), warmup_iter=0)
        else:
            self.scheduler = None
            self.scheduler_type = None

        print('System learnable parameters')
        num_conv_layers = 0
        num_linear_layers = 0
        total_num_parameters = 0
        for name, value in self.named_parameters():
            print(name, value.shape)
            if all(item in name for item in ['conv', 'weight']):
                num_conv_layers += 1
            if all(item in name for item in ['linear', 'weight']):
                num_linear_layers += 1
            total_num_parameters += np.prod(value.shape)

        print('Total number of parameters', total_num_parameters)
        print('Total number of conv layers', num_conv_layers)
        print('Total number of linear layers', num_linear_layers)

        # Generate the directory names
        self.experiment_folder = os.path.abspath(experiment_name)
        self.experiment_logs = os.path.abspath(os.path.join(self.experiment_folder, "result_outputs"))
        self.experiment_saved_models = os.path.abspath(os.path.join(self.experiment_folder, "saved_models"))
        print(self.experiment_folder, self.experiment_logs)
        # Set best models to be at 0 since we are just starting
        self.best_val_model_idx = 0
        self.best_val_model_acc = 0.

        if not os.path.exists(self.experiment_folder):  # If experiment directory does not exist
            os.mkdir(self.experiment_folder)  # create the experiment directory

        if not os.path.exists(self.experiment_logs):
            os.mkdir(self.experiment_logs)  # create the experiment log directory

        if not os.path.exists(self.experiment_saved_models):
            os.mkdir(self.experiment_saved_models)  # create the experiment saved models directory

        if pretrained_weights_locations is not None:
            self.load_pre_trained_model(model_save_dir=pretrained_weights_locations,
                                        model_save_name="train_model",
                                        model_idx='best')

        self.num_epochs = num_epochs
        # self.criterion = nn.CrossEntropyLoss().to(self.device)  # send the loss computation to the GPU
        self.labelled_criterion = nn.CrossEntropyLoss().to(self.device)  # send the loss computation to the GPU
        self.unlabelled_criterion = nn.CrossEntropyLoss().to(self.device)  # send the loss computation to the GPU

        if continue_from_epoch == -2:
            try:
                self.best_val_model_idx, self.best_val_model_acc, self.state = self.load_model(
                    model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                    model_idx='latest')  # reload existing model from epoch and return best val model index
                # and the best val acc of that model
                self.starting_epoch = self.state['current_epoch_idx']
            except:
                print("Model objects cannot be found, initializing a new model and starting from scratch")
                self.starting_epoch = 0
                self.state = dict()

        elif continue_from_epoch != -1:  # if continue from epoch is not -1 then
            self.best_val_model_idx, self.best_val_model_acc, self.state = self.load_model(
                model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                model_idx=continue_from_epoch)  # reload existing model from epoch and return best val model index
            # and the best val acc of that model
            self.starting_epoch = self.state['current_epoch_idx']
        else:
            self.starting_epoch = 0
            self.state = dict()

    def get_num_parameters(self):
        total_num_params = 0
        for param in self.parameters():
            total_num_params += np.prod(param.shape)

        return total_num_params

    def guess_labels(self, u, threshold):
        with torch.no_grad():
            probs = torch.softmax(self.model.forward(u), dim=1)
            scores, lbs = torch.max(probs, dim=1)
            idx = scores > threshold
            lbs = lbs[idx]

        return lbs.detach(), idx

    def run_train_iter(self, x, y, u):
        """
        Receives the inputs and targets for the model and runs a training iteration. Returns loss and accuracy metrics.
        :param x: The inputs to the model. A numpy array of shape batch_size, channels, height, width
        :param y: The targets for the model. A numpy array of shape batch_size, num_classes
        :return: the loss and accuracy for this batch
        """
        self.train()  # sets model to training mode (in case batch normalization or other methods have different procedures for training and evaluation)

        if len(y.shape) > 1:
            y = np.argmax(y, axis=1)  # convert one hot encoded labels to single integer labels

        # print(type(x))

        if type(x) is np.ndarray:
            x, y = torch.Tensor(x).float().to(device=self.device), torch.Tensor(y).long().to(
                device=self.device)  # send data to device as torch tensors

        u_weak = u[0]
        u_strong = u[1]
        u_weak = u_weak.to(self.device, non_blocking=True)
        u_strong = u_strong.to(self.device, non_blocking=True)

        u_guessed_labels, valid_idx = self.guess_labels(u_weak, self.threshold)
        u_guessed_labels = u_guessed_labels.to(self.device, non_blocking=True)

        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        if valid_idx.any():
            u_strong = u_strong[valid_idx]
            u_strong_out = self.model.forward(u_strong)
            unlabelled_loss = self.unlabelled_criterion(input=u_strong_out, target=u_guessed_labels)
        else:
            unlabelled_loss = torch.tensor(0)

        out = self.model.forward(x)  # forward the data in the model
        labelled_loss = self.labelled_criterion(input=out, target=y)  # compute loss

        loss = labelled_loss + (self.loss_lambda_u * unlabelled_loss)

        self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
        loss.backward()  # backpropagate to compute gradients for current iter loss

        self.ema.update_params()

        self.optimizer.step()  # update network parameters

        if self.scheduler is not None and self.scheduler_type is 'FixMatchCos':
            self.scheduler.step()

        _, predicted = torch.max(out.data, 1)  # get argmax of predictions
        accuracy = np.mean(list(predicted.eq(y.data).cpu()))  # compute accuracy
        return loss.data.detach().cpu().numpy(), accuracy

    def run_evaluation_iter(self, x, y, model):
        """
        Receives the inputs and targets for the model and runs an evaluation iterations. Returns loss and accuracy metrics.
        :param x: The inputs to the model. A numpy array of shape batch_size, channels, height, width
        :param y: The targets for the model. A numpy array of shape batch_size, num_classes
        :return: the loss and accuracy for this batch
        """
        self.eval()  # sets the system to validation mode
        if len(y.shape) > 1:
            y = np.argmax(y, axis=1)  # convert one hot encoded labels to single integer labels
        if type(x) is np.ndarray:
            x, y = torch.Tensor(x).float().to(device=self.device), torch.Tensor(y).long().to(
                device=self.device)  # convert data to pytorch tensors and send to the computation device

        x = x.to(self.device)
        y = y.to(self.device)
        out = model.forward(x)  # forward the data in the model
        loss = F.cross_entropy(out, y)  # compute loss
        _, predicted = torch.max(out.data, 1)  # get argmax of predictions
        accuracy = np.mean(list(predicted.eq(y.data).cpu()))  # compute accuracy

        y_cpu = y.data.cpu()
        predicted_cpu = predicted.cpu()

        f1 = f1_score(y_cpu, predicted_cpu, average='macro')
        precision = precision_score(y_cpu, predicted_cpu, average='macro')
        recall = recall_score(y_cpu, predicted_cpu, average='macro')

        return loss.data.detach().cpu().numpy(), accuracy, f1, precision, recall

    def save_model(self, model_save_dir, model_save_name, model_idx, state):
        """
        Save the network parameter state and current best val epoch idx and best val accuracy.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :param best_validation_model_idx: The index of the best validation model to be stored for future use.
        :param best_validation_model_acc: The best validation accuracy to be stored for use at test time.
        :param model_save_dir: The directory to store the state at.
        :param state: The dictionary containing the system state.

        """
        state['network'] = self.state_dict()  # save network parameter and other variables.
        torch.save(state, f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(
            model_idx))))  # save state at prespecified filepath

    def run_training_epoch(self, current_epoch_losses):
        with tqdm.tqdm(total=len(self.train_data), file=sys.stdout) as pbar_train:  # create a progress bar for training
            for idx, (x, y) in enumerate(self.train_data):  # get data batches
                try:
                    augmented_all = self.train_data_unlabeled.next()
                except:
                    self.train_data_unlabeled = iter(self.temp)
                    augmented_all = self.train_data_unlabeled.next()

                loss, accuracy = self.run_train_iter(x=x, y=y, u=augmented_all)  # take a training iter step
                current_epoch_losses["train_loss"].append(loss)  # add current iter loss to the train loss list
                current_epoch_losses["train_acc"].append(accuracy)  # add current iter acc to the train acc list
                pbar_train.update(1)
                pbar_train.set_description("loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))

            self.ema.update_buffer()

        return current_epoch_losses

    def run_validation_epoch(self, current_epoch_losses):
        self.ema.apply_shadow()
        self.ema.model.eval()
        self.ema.model.cuda()

        with tqdm.tqdm(total=len(self.val_data), file=sys.stdout) as pbar_val:  # create a progress bar for validation
            for x, y in self.val_data:  # get data batches
                loss, accuracy, f1, precision, recall = self.run_evaluation_iter(x=x, y=y, model=self.ema.model)  # run a validation iter

                current_epoch_losses["val_loss"].append(loss)  # add current iter loss to val loss list.
                current_epoch_losses["val_acc"].append(accuracy)  # add current iter acc to val acc lst.
                current_epoch_losses["val_f1"].append(f1)
                current_epoch_losses["val_precision"].append(precision)
                current_epoch_losses["val_recall"].append(recall)

                pbar_val.update(1)  # add 1 step to the progress bar
                pbar_val.set_description("loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))

        self.ema.restore()
        return current_epoch_losses

    def run_testing_epoch(self, current_epoch_losses):
        self.ema.apply_shadow()
        self.ema.model.eval()
        self.ema.model.cuda()
        with tqdm.tqdm(total=len(self.test_data), file=sys.stdout) as pbar_test:  # ini a progress bar
            for x, y in self.test_data:  # sample batch
                # compute loss and accuracy by running an evaluation step
                loss, accuracy, f1, precision, recall = self.run_evaluation_iter(x=x, y=y, model=self.ema.model)

                current_epoch_losses["test_loss"].append(loss)  # save test loss
                current_epoch_losses["test_acc"].append(accuracy)  # save test accuracy
                current_epoch_losses["test_f1"].append(f1)
                current_epoch_losses["test_precision"].append(precision)
                current_epoch_losses["test_recall"].append(recall)

                pbar_test.update(1)  # update progress bar status
                pbar_test.set_description(
                    "loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))  # update progress bar string output
        return current_epoch_losses

    def load_model(self, model_save_dir, model_save_name, model_idx):
        """
        Load the network parameter state and the best val model idx and best val acc to be compared with the future val accuracies, in order to choose the best val model
        :param model_save_dir: The directory to store the state at.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :return: best val idx and best val model acc, also it loads the network state into the system state without returning it
        """
        state = torch.load(f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx))))
        self.load_state_dict(state_dict=state['network'])
        return state['best_val_model_idx'], state['best_val_model_acc'], state

    def load_pre_trained_model(self, model_save_dir, model_save_name, model_idx):
        state = torch.load(f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx))))
        new_dict = {}

        for key in self.state_dict().keys():
            temp_key = key.replace('model.', '')
            temp_key = f'model.layer_dict.encoder.{temp_key}'

            if temp_key in state['network']:
                new_dict[key] = state['network'][temp_key]
            else:
                new_dict[key] = self.state_dict()[key]

        self.load_state_dict(state_dict=new_dict)

    def run_experiment(self):
        """
        Runs experiment train and evaluation iterations, saving the model and best val model and val model accuracy after each epoch
        :return: The summary current_epoch_losses from starting epoch to total_epochs.
        """
        total_losses = {"train_acc": [], "train_loss": [], "val_acc": [],
                        "val_loss": [], "val_f1": [], "val_precision": [], "val_recall": [],
                        "curr_epoch": []}  # initialize a dict to keep the per-epoch metrics
        for i, epoch_idx in enumerate(range(self.starting_epoch, self.num_epochs)):
            epoch_start_time = time.time()
            current_epoch_losses = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": [],
                                    "val_f1": [], "val_precision": [], "val_recall": []}

            current_epoch_losses = self.run_training_epoch(current_epoch_losses)
            current_epoch_losses = self.run_validation_epoch(current_epoch_losses)

            if self.scheduler is not None and self.scheduler_type is not 'FixMatchCos':
                self.scheduler.step()

            val_mean_accuracy = np.mean(current_epoch_losses['val_acc'])
            if val_mean_accuracy > self.best_val_model_acc:  # if current epoch's mean val acc is greater than the saved best val acc then
                self.best_val_model_acc = val_mean_accuracy  # set the best val model acc to be current epoch's val accuracy
                self.best_val_model_idx = epoch_idx  # set the experiment-wise best val idx to be the current epoch's idx

            for key, value in current_epoch_losses.items():
                total_losses[key].append(np.mean(value))
                # get mean of all metrics of current epoch metrics dict,
                # to get them ready for storage and output on the terminal.

            total_losses['curr_epoch'].append(epoch_idx)
            save_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv',
                            stats_dict=total_losses, current_epoch=i,
                            continue_from_mode=True if (
                                    self.starting_epoch != 0 or i > 0) else False)  # save statistics to stats file.

            # load_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv') # How to load a csv file if you need to

            out_string = "_".join(
                ["{}_{:.4f}".format(key, np.mean(value)) for key, value in current_epoch_losses.items()])
            # create a string to use to report our epoch metrics
            epoch_elapsed_time = time.time() - epoch_start_time  # calculate time taken for epoch
            epoch_elapsed_time = "{:.4f}".format(epoch_elapsed_time)
            print("Epoch {}:".format(epoch_idx), out_string, "epoch time", epoch_elapsed_time, "seconds")
            self.state['current_epoch_idx'] = epoch_idx
            self.state['best_val_model_acc'] = self.best_val_model_acc
            self.state['best_val_model_idx'] = self.best_val_model_idx
            self.save_model(model_save_dir=self.experiment_saved_models,
                            # save model and best val idx and best val acc, using the model dir, model name and model idx
                            model_save_name="train_model", model_idx=epoch_idx, state=self.state)
            self.save_model(model_save_dir=self.experiment_saved_models,
                            # save model and best val idx and best val acc, using the model dir, model name and model idx
                            model_save_name="train_model", model_idx='latest', state=self.state)

        print("Generating test set evaluation metrics")
        self.load_model(model_save_dir=self.experiment_saved_models, model_idx=self.best_val_model_idx,
                        # load best validation model
                        model_save_name="train_model")
        current_epoch_losses = {"test_acc": [], "test_loss": [], "test_f1": [], "test_precision": [],
                                "test_recall": []}  # initialize a statistics dict

        current_epoch_losses = self.run_testing_epoch(current_epoch_losses=current_epoch_losses)

        test_losses = {key: [np.mean(value)] for key, value in
                       current_epoch_losses.items()}  # save test set metrics in dict format

        save_statistics(experiment_log_dir=self.experiment_logs, filename='test_summary.csv',
                        # save test set metrics on disk in .csv format
                        stats_dict=test_losses, current_epoch=0, continue_from_mode=False)

        return total_losses, test_losses


class EMA(object):
    # From https://github.com/CoinCheung/fixmatch/
    def __init__(self, model, alpha):
        self.step = 0
        self.model = model
        self.alpha = alpha
        self.shadow = self.get_model_state()
        self.backup = {}
        self.param_keys = [k for k, _ in self.model.named_parameters()]
        self.buffer_keys = [k for k, _ in self.model.named_buffers()]

    def update_params(self):
        decay = min(self.alpha, (self.step + 1) / (self.step + 10))
        state = self.model.state_dict()
        for name in self.param_keys:
            self.shadow[name].copy_(
                decay * self.shadow[name]
                + (1 - decay) * state[name]
            )
        #  for name in self.buffer_keys:
        #      self.shadow[name].copy_(
        #          decay * self.shadow[name]
        #          + (1 - decay) * state[name]
        #      )
        self.step += 1

    def update_buffer(self):
        state = self.model.state_dict()
        for name in self.buffer_keys:
            self.shadow[name].copy_(state[name])

    def apply_shadow(self):
        self.backup = self.get_model_state()
        self.model.load_state_dict(self.shadow)

    def restore(self):
        self.model.load_state_dict(self.backup)

    def get_model_state(self):
        return {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }


class WarmupCosineLrScheduler(_LRScheduler):
    '''
    This is different from official definition, this is implemented according to
    the paper of fix-match

    # From https://github.com/CoinCheung/fixmatch/
    '''

    def __init__(
            self,
            optimizer,
            max_iter,
            warmup_iter,
            warmup_ratio=5e-4,
            warmup='exp',
            last_epoch=-1,
    ):
        self.max_iter = max_iter
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        super(WarmupCosineLrScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        ratio = self.get_lr_ratio()
        lrs = [ratio * lr for lr in self.base_lrs]
        return lrs

    def get_lr_ratio(self):
        if self.last_epoch < self.warmup_iter:
            ratio = self.get_warmup_ratio()
        else:
            real_iter = self.last_epoch - self.warmup_iter
            real_max_iter = self.max_iter - self.warmup_iter
            ratio = np.cos((7 * np.pi * real_iter) / (16 * real_max_iter))
        return ratio

    def get_warmup_ratio(self):
        assert self.warmup in ('linear', 'exp')
        alpha = self.last_epoch / self.warmup_iter
        if self.warmup == 'linear':
            ratio = self.warmup_ratio + (1 - self.warmup_ratio) * alpha
        elif self.warmup == 'exp':
            ratio = self.warmup_ratio ** (1. - alpha)
        return ratio
