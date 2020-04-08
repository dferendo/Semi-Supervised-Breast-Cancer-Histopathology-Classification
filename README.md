# Semi-Supervised Breast Cancer Histopathology Classification

**Abstract:** Semi-supervised learning allows one to leverage unlabeled data to improve classifier performance. Due to the expense and expertise required to label medical data, semi-supervised learning could improve the diagnostic capabilities of deep learning tools in a medical context. We experiment with semi-supervised learning algorithms called MixMatch and FixMatch in order to minimize the number of labeled training examples required to perform binary classification on BreaKHis, a breast cancer imaging benchmark data set. We show that by using only 5 labeled data points for every sub-class, we can achieve a test F<sub>1</sub> score of 0.822 ± 0.001. We additionally demonstrate that by pre-training our network weights using an autoencoder we can greatly reduce training stochasticity.

Please read report.pdf for the full report including model descriptions and more information on the dataset used. 
