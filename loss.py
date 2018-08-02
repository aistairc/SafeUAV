import torch as tr
import torch.nn.functional as F
import numpy as np

def l2_loss(networkOutput, label, Lambda=0):
	res1 = tr.sum( (networkOutput - label)**2, dim=1).sum(dim=1)
	return tr.mean(res1)

def classification_loss(y_pred, y_true):
	# (240, 320, 3) => (153600, 3) and (240, 320, 1) => (153600, )
	y_pred = y_pred.contiguous().view(-1, y_pred.shape[-1])
	y_true = y_true.view(-1)
	y_pred = F.log_softmax(y_pred, dim=-1)
	loss = F.nll_loss(y_pred, y_true)
	return loss

def mIoUMetric(y_pred_batch, y_true_batch, **kwargs):
	def IoU(y_pred_batch, y_true_batch):
		predTrue = y_pred_batch == True
		predFalse = y_pred_batch == False
		labelTrue = y_true_batch == True
		labelFalse = y_true_batch == False

		TP = np.logical_and(predTrue, labelTrue)
		FP = np.logical_and(predTrue, labelFalse)
		FN = np.logical_and(predFalse, labelTrue)
		return TP / (TP + FP + FN + 1e-8)

	y_pred_batch = np.argmax(y_pred_batch, axis=-1)
	res = [IoU((y_pred_batch == i), (y_true_batch == i)[..., 0]) for i in range(3)]
	return np.mean(np.array(res))

def metterMetric(y_pred_batch, y_true_batch, **kwargs):
	F = kwargs["reader"].maximums["depth"]
	y_pred_new = y_pred_batch * F
	y_true_new = y_true_batch * F
	return np.mean(np.abs(y_pred_new - y_true_new))