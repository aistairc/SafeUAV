import sys
import numpy as np
from neural_wrappers.readers import CitySimReader
from argparse import ArgumentParser

from test_dataset import testDataset

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

	args = parser.parse_args()
	assert args.type in ("test_dataset", )
	assert args.task in ("classification", "regression")
	# assert args.type in ("train", "train_pretrained", "test", "retrain", "export_new_labels", "test_videos")
	args.data_dims = args.data_dims.split(",")
	args.label_dims = args.label_dims.split(",")
	return args

def main():
	args = getArgs()

	hvnTransform = "hvn_two_dims" if args.task == "regression" else "identity_long"
	datasetReader = CitySimReader(args.dataset_path, dataDims=args.data_dims, labelDims=args.label_dims, \
		hvnTransform=hvnTransform)

	if args.type == "test_dataset":
		testDataset(datasetReader, args)
		sys.exit(0)

if __name__ == "__main__":
	main()