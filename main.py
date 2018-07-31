import sys
import numpy as np
import torch.optim as optim
from argparse import ArgumentParser
from neural_wrappers.readers import CitySimReader
from neural_wrappers.models import ModelUNetDilatedConv
from neural_wrappers.pytorch import maybeCuda

from unet_tiny_sum import ModelUNetTinySum
from test_dataset import testDataset
from loss import l2_loss, classification_loss


def getArgs():
	parser = ArgumentParser()
	parser.add_argument("type")
	parser.add_argument("task", help="regression for Depth / classification for HVN")
	parser.add_argument("dataset_path", help="Path to dataset")

	# Dataset and Data stuff
	parser.add_argument("--data_dims", default="rgb")
	parser.add_argument("--label_dims", default="depth")

	# Training stuff
	parser.add_argument("--batch_size", default=10, type=int)
	parser.add_argument("--num_epochs", default=100, type=int)
	parser.add_argument("--optimizer", type=str)
	parser.add_argument("--learning_rate", type=float)
	parser.add_argument("--momentum", default=0.9, type=float)

	# Model stuff
	parser.add_argument("--model", type=str)

	args = parser.parse_args()
	assert args.type in ("test_dataset", "train")
	assert args.task in ("classification", "regression")
	if not args.type in ("test_dataset", ):
		assert args.model in ("unet_big_concatenate", "unet_tiny_sum")
	# assert args.type in ("train", "train_pretrained", "test", "retrain", "export_new_labels", "test_videos")
	args.data_dims = args.data_dims.split(",")
	args.label_dims = args.label_dims.split(",")

	return args

def getModel(args, reader):
	def getNumDims(allDims, Dims, hvnNumDims):
		numDims = 0
		for dim in Dims:
			assert dim in allDims
			if dim == "rgb":
				numDims += 3
			elif dim == "depth":
				numDims += 1
			elif dim == "hvn_gt_p1":
				numDims += hvnNumDims
			else:
				assert False, "%s" % dim
		return numDims

	hvnNumDims = 2 if args.task == "regression" else 1
	allDims = reader.allDims
	dIn, dOut = getNumDims(allDims, args.data_dims, hvnNumDims), getNumDims(allDims, args.label_dims, hvnNumDims)
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

def main():
	args = getArgs()

	hvnTransform = "hvn_two_dims" if args.task == "regression" else "identity_long"
	reader = CitySimReader(args.dataset_path, dataDims=args.data_dims, labelDims=args.label_dims, \
		hvnTransform=hvnTransform)
	print(reader.summary())

	if args.type == "test_dataset":
		testDataset(reader, args)
		sys.exit(0)

	model = getModel(args, reader)
	setOptimizer(args, model)
	criterion = l2_loss if args.task == "regression" else classification_loss
	model.setCriterion(criterion)
	print(model.summary())

	if args.type == "train":
		generator = reader.iterate("train", miniBatchSize=args.batch_size)
		steps = reader.getNumIterations("train", miniBatchSize=args.batch_size)
		valGenerator = reader.iterate("train", miniBatchSize=args.batch_size)
		valSteps = reader.getNumIterations("train", miniBatchSize=args.batch_size)

		model.train_generator(generator, steps, numEpochs=args.num_epochs, callbacks=[], \
			validationGenerator=valGenerator, validationSteps=valSteps)
	else:
		pass

if __name__ == "__main__":
	main()