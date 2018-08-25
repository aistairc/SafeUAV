import sys
import os
import h5py
import numpy as np
from main import getModel
from neural_wrappers.readers import CitySimReader
from argparse import ArgumentParser
from neural_wrappers.callbacks import Callback
from loss import l2_loss, classification_loss

def getArgs():
	parser = ArgumentParser()
	parser.add_argument("task", help="regression for Depth / classification for HVN")
	parser.add_argument("dataset_path", help="Path to dataset")

	parser.add_argument("--batch_size", default=10, type=int)

	# Model stuff
	parser.add_argument("--model", type=str)
	parser.add_argument("--weights_file")
	parser.add_argument("--label_dims")

	args = parser.parse_args()
	assert not args.weights_file is None
	args.weights_file = os.path.abspath(args.weights_file)
	assert args.task in ("classification", "regression")
	assert args.model in ("unet_big_concatenate", "unet_tiny_sum")

	args.data_dims = ["rgb"]
	if args.task == "classification":
		args.label_dims = ["hvn_gt_p1"]
	else:
		args.label_dims = ["depth"]
	return args

class ExportCallback(Callback):
	def __init__(self, reader, originalLabelsName, newLabelsName):
		self.file = h5py.File("%s.h5" % (newLabelsName), "w")
		dType = reader.dataset["train"][originalLabelsName].dtype
		trainShape = list(reader.dataset["train"][originalLabelsName].shape)
		valShape = list(reader.dataset["validation"][originalLabelsName].shape)
		trainShape[1 : 3] = 240, 320
		valShape[1 : 3] = 240, 320

		self.newLabelsName = newLabelsName

		self.file.create_group("train")
		self.file.create_group("validation")
		self.file["train"].create_dataset(newLabelsName, dtype=dType, shape=trainShape)
		self.file["validation"].create_dataset(newLabelsName, dtype=dType, shape=valShape)

		self.group = None
		self.index = None

	def setGroup(self, group):
		assert group in ("train", "validation")
		self.group = group
		self.index = 0

	def onIterationEnd(self, **kwargs):
		results = kwargs["results"]
		dataset = self.file[self.group][self.newLabelsName]
		for i in range(len(results)):
			result = results[i]
			if "hvn" in self.newLabelsName:
				result = np.argmax(result, axis=-1)
			else:
				result = result[..., 0]
			dataset[self.index] = result
			self.index += 1
			if self.index % 10 == 0:
				print("Done %d" % (self.index))

def main():
	args = getArgs()

	hvnTransform = "hvn_two_dims" if args.task == "regression" else "identity_long"
	reader = CitySimReader(args.dataset_path, dataDims=args.data_dims, labelDims=args.label_dims, \
		resizer=(240, 320), hvnTransform=hvnTransform, dataGroup="all")
	trainGenerator = reader.iterate_once("train", args.batch_size)
	trainSteps = reader.getNumIterations("train", args.batch_size)
	valGenerator = reader.iterate_once("validation", args.batch_size)
	valSteps = reader.getNumIterations("validation", args.batch_size)
	print(reader.summary())

	model = getModel(args, reader)
	criterion = l2_loss if args.task == "regression" else classification_loss
	model.setCriterion(criterion)
	print(model.summary())

	modelName = "tiny" if "tiny" in args.model else "big"
	newLabelDim = "depth_%s_it1" % (modelName) if args.label_dims[0] == "depth" else "hvn_%s_it1_p1" % (modelName)
	callback = ExportCallback(reader, args.label_dims[0], newLabelDim)

	callback.setGroup("train")
	model.test_generator(trainGenerator, trainSteps, callbacks=[callback])
	callback.setGroup("validation")
	model.test_generator(valGenerator, valSteps, callbacks=[callback])

if __name__ == "__main__":
	main()