import sys
import os
import numpy as np
import torch.optim as optim
from argparse import ArgumentParser
from torch.optim.lr_scheduler import ReduceLROnPlateau
from neural_wrappers.readers import CitySimReader
from neural_wrappers.models import ModelUNetDilatedConv
from neural_wrappers.pytorch import maybeCuda
from neural_wrappers.callbacks import SaveHistory, SaveModels, Callback, PlotMetricsCallback

from unet_tiny_sum import ModelUNetTinySum
from test_dataset import testDataset
from loss import l2_loss, classification_loss

def getArgs():
	parser = ArgumentParser()
	parser.add_argument("type")
	parser.add_argument("task", help="regression for Depth / classification for HVN")
	parser.add_argument("dataset_path", help="Path to dataset")

	# Dataset and Data stuff
	parser.add_argument("--dir")
	parser.add_argument("--data_dims", default="rgb")
	parser.add_argument("--label_dims", default="depth")
	parser.add_argument("--data_group", default="all")

	# Training stuff
	parser.add_argument("--batch_size", default=10, type=int)
	parser.add_argument("--num_epochs", default=100, type=int)
	parser.add_argument("--optimizer", type=str)
	parser.add_argument("--learning_rate", type=float)
	parser.add_argument("--momentum", default=0.9, type=float)
	parser.add_argument("--patience", default=4, type=int)

	# Model stuff
	parser.add_argument("--model", type=str)
	parser.add_argument("--weights_file")

	args = parser.parse_args()
	assert args.type in ("test_dataset", "train", "retrain")
	assert args.task in ("classification", "regression")
	if not args.type in ("test_dataset", ):
		assert args.model in ("unet_big_concatenate", "unet_tiny_sum")
	if args.type == "retrain":
		args.weights_file = os.path.abspath(args.weights_file)
	args.data_dims = args.data_dims.split(",")
	args.label_dims = args.label_dims.split(",")

	return args

class SchedulerCallback(Callback):
	def __init__(self, optimizer, patience):
		self.scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=patience, eps=1e-8)

	def onEpochEnd(self, **kwargs):
		if not kwargs["validationMetrics"]:
			loss = kwargs["trainMetrics"]["Loss"]
		else:
			loss = kwargs["validationMetrics"]["Loss"]
		self.scheduler.step(loss)

def getModel(args, reader):
	dIn, dOut = reader.getNumDimensions(args.data_dims), reader.getNumDimensions(args.label_dims)
	if args.model == "unet_big_concatenate":
		model = ModelUNetDilatedConv(dIn=dIn, dOut=dOut, numFilters=64, bottleneckMode="dilate2_serial_concatenate")
	elif args.model == "unet_tiny_sum":
		model = ModelUNetTinySum(dIn=dIn, dOut=dOut, numFilters=16)
	model = maybeCuda(model)
	return model

def setOptimizer(args, model):
	if args.optimizer == "SGD":
		model.setOptimizer(optim.SGD, lr=args.learning_rate, momentum=args.momentum, nesterov=False)
	elif args.optimizer == "Nesterov":
		model.setOptimizer(optim.SGD, lr=args.learning_rate, momentum=args.momentum, nesterov=True)
	elif args.optimizer == "RMSProp":
		model.setOptimizer(optim.RMSProp, lr=args.learning_rate)
	elif args.optimizer == "Adam":
		model.setOptimizer(optim.Adam, lr=args.learning_rate)
	else:
		assert False, "%s" % args.optimizer

def changeDirectory(Dir, expectExist):
	assert os.path.exists(Dir) == expectExist
	print("Changing to working directory:", Dir)
	if expectExist == False:
		os.makedirs(Dir)
	os.chdir(Dir)

def main():
	args = getArgs()

	hvnTransform = "hvn_two_dims" if args.task == "regression" else "identity_long"
	reader = CitySimReader(args.dataset_path, dataDims=args.data_dims, labelDims=args.label_dims, \
		resizer=(240, 320), hvnTransform=hvnTransform, dataGroup=args.data_group)
	print(reader.summary())

	if args.type == "test_dataset":
		testDataset(reader, args)
		sys.exit(0)

	model = getModel(args, reader)
	setOptimizer(args, model)
	criterion = l2_loss if args.task == "regression" else classification_loss
	model.setCriterion(criterion)
	print(model.summary())

	generator = reader.iterate("train", miniBatchSize=args.batch_size)
	steps = reader.getNumIterations("train", miniBatchSize=args.batch_size)
	valGenerator = reader.iterate("validation", miniBatchSize=args.batch_size)
	valSteps = reader.getNumIterations("validation", miniBatchSize=args.batch_size)

	if args.type == "train":
		changeDirectory(args.dir, expectExist=False)

		callbacks = [SaveModels(type="all"), SaveHistory("history.txt", mode="write"), \
			PlotMetricsCallback(["Loss"], ["min"]), SchedulerCallback(model.optimizer, args.patience)]
		callbacks[1].file.write(reader.summary())
		callbacks[1].file.write(model.summary())

		model.train()
		model.train_generator(generator, stepsPerEpoch=steps, numEpochs=args.num_epochs, callbacks=callbacks, \
			validationGenerator=valGenerator, validationSteps=valSteps)
	elif args.type == "retrain":
		changeDirectory(args.dir, expectExist=True)
		model.loadModel(args.weights_file)

		model.train()
		model.train_generator(generator, stepsPerEpoch=steps, numEpochs=args.num_epochs, \
			callbacks=None, validationGenerator=valGenerator, validationSteps=valSteps)
	else:
		pass

if __name__ == "__main__":
	main()