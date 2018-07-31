import torch as tr
import torch.nn.functional as F

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
