import sys
import os
import moviepy.editor as mpy
import pims
import numpy as np
import torch as tr
from neural_wrappers.utilities import minMaxNormalizeData
from neural_wrappers.readers import CitySimReader
from neural_wrappers.pytorch import maybeCuda
from Mihlib import show_plots, plot_image
from functools import partial
from lycon import resize, Interpolation
from matplotlib.cm import hot

from main import getModel, SchedulerCallback
from argparse import ArgumentParser

def getArgs():
	parser = ArgumentParser()
	parser.add_argument("task", help="regression for Depth / classification for HVN")
	parser.add_argument("in_video", help="Path to input vieo")
	parser.add_argument("out_video", help="Path to output video")

	# Model stuff
	parser.add_argument("--model", type=str)
	parser.add_argument("--weights_file")
	parser.add_argument("--data_dims", default="rgb")
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

def minMaxNormalizeFrame(frame):
	Min, Max = np.min(frame), np.max(frame)
	frame -= Min
	frame /= (Max - Min)
	frame *= 255
	return np.uint8(frame)

def make_frame(t, model, video, fps, inputShape, args):
	t = min(int(t * fps), len(video) - 1)
	currentFrame = np.array(video[t])
	outH, outW = currentFrame.shape[0 : 2]
	inH, inW = inputShape

	image = resize(currentFrame, height=inH, width=inW, interpolation=Interpolation.CUBIC)
	image = minMaxNormalizeData(np.float32(image), np.min(image), np.max(image))
	image = np.expand_dims(image, axis=0)

	trImage = maybeCuda(tr.from_numpy(image))
	trResult = model.forward(trImage)
	npResult = trResult.detach().cpu().numpy()[0]
	npResult = minMaxNormalizeFrame(npResult)
	# npResult = resize(npResult, height=outH, width=outW, interpolation=Interpolation.CUBIC)

	if args.label_dims == ["depth"]:
		frame = hot(npResult)[..., 0 : 3]
	# HVN
	else:
		hvnFrame = np.argmax(npResult, axis=-1)
		frame = np.zeros((*hvnFrame.shape, 3), dtype=np.float32)
		frame[np.where(hvnFrame == 0)] = (255, 0, 0)
		frame[np.where(hvnFrame == 1)] = (255, 255, 0)
		frame[np.where(hvnFrame == 2)] = (0, 0, 255)

	# print(image.shape)
	# plot_image(image[0])
	# plot_image(frame)
	# show_plots()
	finalFrame = minMaxNormalizeFrame(frame)
	return finalFrame

def main():
	args = getArgs()
	video = pims.Video(args.in_video)
	fps = video.frame_rate
	duration = max(0, int(len(video) / fps) - 2)
	print("In video: %s. Out video: %s FPS: %s. Duration: %s." % (args.in_video, args.out_video, fps, duration))

	hvnTransform = "hvn_two_dims" if args.task == "regression" else "identity_long"
	dIn = CitySimReader.getNumDimensions(args.data_dims, hvnTransform)
	dOut = 3 if args.task == "classification" else CitySimReader.getNumDimensions(args.label_dims, hvnTransform)
	model = getModel(args, dIn, dOut)
	model.loadWeights(args.weights_file)

	inH, inW = 240, 320
	frameCallback = partial(make_frame, model=model, video=video, fps=fps, inputShape=(inH, inW), args=args)
	clip = mpy.VideoClip(frameCallback, duration=duration)
	clip.write_videofile(args.out_video, fps=fps, verbose=False, progress_bar=True)

if __name__ == "__main__":
	main()