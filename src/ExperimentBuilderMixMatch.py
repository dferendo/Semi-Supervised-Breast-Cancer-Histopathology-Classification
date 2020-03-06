import sys
from copy import deepcopy

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
from torch.optim.lr_scheduler import MultiStepLR

from ERF_Scheduler import ERF

from storage_utils import save_statistics


class WeightEMA(object):
    def __init__(self, model, ema_model, lr, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.wd = 0.02 * lr

    def step(self, model, ema_model):
        one_minus_alpha = 1.0 - self.alpha
        for name, param in ema_model.named_parameters():
            param.data = (self.alpha * param.data) + (model.state_dict()[name].data * one_minus_alpha)
            # self.params[index] = param.mul(1 - self.wd)
        return ema_model


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu


def create_ema_model(model):
    model_out = deepcopy(model)

    for param in model_out.parameters():
        param.detach_()

    return model_out


def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


class ExperimentBuilderMixMatch(nn.Module):
    def __init__(self, network_model, experiment_name, num_epochs, train_data, train_data_unlabeled, val_data,
                 test_data, use_gpu, continue_from_epoch=-1, scheduler=None, optimiser=None, sched_params=None,
                 optim_params=None, lambda_u=100, sharpen_temp=0.5, mixup_alpha=0.75):
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
        super(ExperimentBuilderMixMatch, self).__init__()

        self.experiment_name = experiment_name
        self.model = network_model
        self.model.reset_parameters()
        # self.ema_model = create_ema_model(self.model)
        self.device = torch.cuda.current_device()

        if torch.cuda.device_count() > 1 and use_gpu:
            self.device = torch.cuda.current_device()
            self.model.to(self.device)
            # self.ema_model.to(self.device)  # sends the model from the cpu to the gpu
            self.model = nn.DataParallel(module=self.model)
            # self.ema_model = nn.DataParallel(module=self.ema_model)
            print('Use Multi GPU', self.device)
        elif torch.cuda.device_count() == 1 and use_gpu:
            self.device = torch.cuda.current_device()
            self.model.to(self.device)  # sends the model from the cpu to the gpu
            # self.ema_model.to(self.device)  # sends the model from the cpu to the gpu
            print('Use GPU', self.device)
        else:
            print("use CPU")
            self.device = torch.device('cpu')  # sets the device to be CPU
            print(self.device)

        # re-initialize network parameters
        self.train_data = train_data
        self.train_data_unlabeled = train_data_unlabeled
        self.temp = train_data_unlabeled
        self.val_data = val_data
        self.test_data = test_data
        self.sharpen_temp = sharpen_temp
        self.mixup_alpha = mixup_alpha
        self.loss_lambda_u = lambda_u

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

        # self.ema_optimiser = WeightEMA(ema_model=self.ema_model, model=self.model, lr=sched_params['lr_max'])

        if scheduler == 'ERF':
            self.scheduler = ERF(self.optimizer,
                                 min_lr=sched_params['lr_min'],
                                 alpha=sched_params['erf_alpha'],
                                 beta=sched_params['erf_beta'],
                                 epochs=num_epochs)
        elif scheduler == 'Step':
            self.scheduler = MultiStepLR(self.optimizer,
                                         milestones=[30, 60, 90, 150],
                                         gamma=0.1)
        elif scheduler == 'Cos':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                  T_max=num_epochs,
                                                                  eta_min=0.00001)
        elif scheduler == 'CosWR':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
                                                                            T_0=15,
                                                                            eta_min=0.00001)
        else:
            self.scheduler = None

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

        self.num_epochs = num_epochs

        # self.unlabeled_train_criterion = SemiLoss(self.unlabeled_train_criterion)
        # self.unlabeled_train_criterion = nn.MSELoss().to(self.device, non_blocking=True)
        # self.criterion = nn.CrossEntropyLoss().to(self.device, non_blocking=True)  # send the loss computation to the GPU
        self.criterion = SemiLoss()

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

    def sharpen(self, p, T):
        pt = p ** (1 / T)
        targets_u = pt / pt.sum(dim=1, keepdim=True)
        return targets_u

    def mixup(self, x, y, u_list, q):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        # mixup
        all_inputs = [x]
        all_inputs.extend(u_list)
        all_inputs = torch.cat(all_inputs, dim=0)

        all_targets = [y]
        all_targets.extend([q for i in range(len(u_list))])
        all_targets = torch.cat(all_targets, dim=0)

        l = np.random.beta(self.mixup_alpha, self.mixup_alpha, size=all_inputs.shape[0])
        l = np.maximum(l, 1 - l)
        l = torch.from_numpy(l).float().to(self.device)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        l_t = l.view((l.shape[0], 1))
        mixed_target = l_t * target_a + (1 - l_t) * target_b

        l_i = l.view((l.shape[0], 1, 1, 1))
        mixed_input = l_i * input_a + (1 - l_i) * input_b

        return mixed_input, mixed_target

    def mixmatch(self, x, y, u_list):
        with torch.no_grad():
            # Forward Propagate u_i for all augmentations k
            q_bar = torch.zeros((u_list[0].shape[0], y.shape[1])).to(self.device)
            for augmentation in u_list:
                q_k = torch.softmax(self.model.forward(augmentation), dim=1)
                q_bar += q_k
            q_bar = q_bar / len(u_list)
            q = self.sharpen(q_bar, self.sharpen_temp)
            q = q.detach()

            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            mixed_input, mixed_target = self.mixup(x, y, u_list, q)

            return mixed_input, mixed_target

    def run_train_iter(self, x, u, y, batch_num, batch_total, epoch_num):
        """
        Receives the inputs and targets for the model and runs a training iteration. Returns loss and accuracy metrics.
        :param x: The inputs to the model. A numpy array of shape batch_size, channels, height, width
        :param y: The targets for the model. A numpy array of shape batch_size, num_classes
        :return: the loss and accuracy for this batch
        """
        self.train()  # sets model to training mode (in case batch normalization or other methods have different procedures for training and evaluation)

        # if len(y.shape) > 1:
        #     y = np.argmax(y, axis=1)  # convert one hot encoded labels to single integer labels

        for i in range(len(u)):
            u[i] = u[i].to(self.device, non_blocking=True)

        if type(x) is np.ndarray:
            x, y = torch.Tensor(x).float().to(device=self.device), torch.Tensor(y).long().to(
                device=self.device)  # send data to device as torch tensors

        batch_size = x.shape[0]

        # Apply Mixmatch
        # x and y have same shapes as before
        # u and q have shape batch*k x dims
        mixed_input, mixed_target = self.mixmatch(x, y, u)
        y = mixed_target[:batch_size]
        q = mixed_target[batch_size:]

        # Interleave is simply forming batches of items that come from both labeled and unlabeled batches.
        # Since we only update batch norm for the first batch,
        # it's important that this batch is representative of the whole data.
        # From https://github.com/google-research/mixmatch/issues/5#issuecomment-506432086 and
        # https://github.com/YU1ut/MixMatch-pytorch/blob/a738cc95aae88f76761aeeb405201bc7ae200e7d/train.py#L186
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logits = [self.model.forward(mixed_input[0])]
        for input in mixed_input[1:]:
            logits.append(self.model.forward(input))

        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        rampup = linear_rampup(current=epoch_num + (batch_num / batch_total), rampup_length=self.num_epochs)
        Lx, Lu = self.criterion(logits_x, y, logits_u, q)
        loss = Lx + (Lu * (self.loss_lambda_u * rampup))

        self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
        loss.backward()  # backpropagate to compute gradients for current iter loss

        self.optimizer.step()  # update network parameters
        # self.ema_model = self.ema_optimiser.step(model=self.model, ema_model=self.ema_model)

        if len(y.shape) > 1:
            y = torch.argmax(y, axis=1)  # convert one hot encoded labels to single integer labels

        _, predicted = torch.max(logits_x.data, 1)  # get argmax of predictions
        accuracy = np.mean(list(predicted.eq(y.data).cpu()))  # compute accuracy
        return loss.data.detach().cpu().numpy(), accuracy

    def run_evaluation_iter(self, x, y):
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
        # out = self.ema_model.forward(x)  # forward the data in the model
        out = self.model.forward(x)  # forward the data in the model
        loss = F.cross_entropy(out, y)  # compute loss
        _, predicted = torch.max(out.data, 1)  # get argmax of predictions
        accuracy = np.mean(list(predicted.eq(y.data).cpu()))  # compute accuracy
        return loss.data.detach().cpu().numpy(), accuracy

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

    def run_training_epoch(self, current_epoch_losses, current_epoch):
        with tqdm.tqdm(total=len(self.train_data), file=sys.stdout) as pbar_train:  # create a progress bar for training
            for idx, (x, y) in enumerate(self.train_data):  # get data batches
                try:
                    augmented_all = self.train_data_unlabeled.next()
                except:
                    self.train_data_unlabeled = iter(self.temp)
                    augmented_all = self.train_data_unlabeled.next()

                # take a training iter step
                loss, accuracy = self.run_train_iter(x=x, u=augmented_all, y=y,
                                                     batch_num=idx,
                                                     batch_total=len(self.train_data),
                                                     epoch_num=current_epoch)
                current_epoch_losses["train_loss"].append(loss)  # add current iter loss to the train loss list
                current_epoch_losses["train_acc"].append(accuracy)  # add current iter acc to the train acc list
                pbar_train.update(1)
                pbar_train.set_description("loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))

                # g=0
                # for param in self.ema_model.parameters():
                #     if g==2: break
                #     print(param.data)
                #     g+=1

        return current_epoch_losses

    def run_validation_epoch(self, current_epoch_losses):

        with tqdm.tqdm(total=len(self.val_data), file=sys.stdout) as pbar_val:  # create a progress bar for validation
            for x, y in self.val_data:  # get data batches
                loss, accuracy = self.run_evaluation_iter(x=x, y=y)  # run a validation iter
                current_epoch_losses["val_loss"].append(loss)  # add current iter loss to val loss list.
                current_epoch_losses["val_acc"].append(accuracy)  # add current iter acc to val acc lst.
                pbar_val.update(1)  # add 1 step to the progress bar
                pbar_val.set_description("loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))

        return current_epoch_losses

    def run_testing_epoch(self, current_epoch_losses):

        with tqdm.tqdm(total=len(self.test_data), file=sys.stdout) as pbar_test:  # ini a progress bar
            for x, y in self.test_data:  # sample batch
                loss, accuracy = self.run_evaluation_iter(x=x,
                                                          y=y)  # compute loss and accuracy by running an evaluation step
                current_epoch_losses["test_loss"].append(loss)  # save test loss
                current_epoch_losses["test_acc"].append(accuracy)  # save test accuracy
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

    def run_experiment(self):
        """
        Runs experiment train and evaluation iterations, saving the model and best val model and val model accuracy after each epoch
        :return: The summary current_epoch_losses from starting epoch to total_epochs.
        """
        total_losses = {"train_acc": [], "train_loss": [], "val_acc": [],
                        "val_loss": [], "curr_epoch": []}  # initialize a dict to keep the per-epoch metrics
        for i, epoch_idx in enumerate(range(self.starting_epoch, self.num_epochs)):
            epoch_start_time = time.time()
            current_epoch_losses = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": []}

            current_epoch_losses = self.run_training_epoch(current_epoch_losses, current_epoch=epoch_idx)
            current_epoch_losses = self.run_validation_epoch(current_epoch_losses)

            if self.scheduler is not None:
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
        current_epoch_losses = {"test_acc": [], "test_loss": []}  # initialize a statistics dict

        current_epoch_losses = self.run_testing_epoch(current_epoch_losses=current_epoch_losses)

        test_losses = {key: [np.mean(value)] for key, value in
                       current_epoch_losses.items()}  # save test set metrics in dict format

        save_statistics(experiment_log_dir=self.experiment_logs, filename='test_summary.csv',
                        # save test set metrics on disk in .csv format
                        stats_dict=test_losses, current_epoch=0, continue_from_mode=False)

        return total_losses, test_losses
