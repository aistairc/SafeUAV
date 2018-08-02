import numpy as np
from Mihlib import npGetInfo

def testDataset(reader, args):
	generator = reader.iterate("train", miniBatchSize=args.batch_size)
	steps = reader.getNumIterations("train", miniBatchSize=args.batch_size)
	from Mihlib import plot_image, show_plots
	from functools import partial

	for items in generator:
		datas, labels = items
		MB = datas.shape[0]

		def hvnPlotter(x, hvnTransform):
			print(npGetInfo(x))
			hvn = np.zeros((*x.shape[0 : 2], 3), dtype=np.uint8)
			if hvnTransform in ("identity_long", "identity"):
				hvn[np.where(x == 0)] = (255, 0, 0)
				hvn[np.where(x == 1)] = (255, 255, 0)
				hvn[np.where(x == 2)] = (0, 0, 255)
			elif hvnTransform == "hvn_two_dims":
				hvn[..., 0 : 3] = (0, 0, 255)
				hvn[np.where(x[..., 0] == 1)] = (255, 0, 0)
				hvn[np.where(x[..., 1] == 1)] = (255, 255, 0)
			return hvn

		def plotter(data, dims, hvnTransform, startingIndex, totalItems):
			j = 0
			for i, dim in enumerate(dims):
				if dim == "rgb":
					rgb = data[..., j : j + 3]
					print(npGetInfo(rgb))
					plot_image(rgb, new_figure=(startingIndex==1 and i==0), show_axis=False, \
						title="RGB", axis=(1, totalItems, startingIndex + i))
					j += 3
				elif dim == "hvn_gt_p1":
					if hvnTransform in ("identity", "identity_long"):
						hvn = hvnPlotter(data[..., j], hvnTransform)
					elif hvnTransform == "hvn_two_dims":
						hvn = hvnPlotter(data[..., j : j + 2], hvnTransform)
					plot_image(hvn, new_figure=(startingIndex==1 and i==0), show_axis=False, title="HVN", \
						axis=(1, totalItems, startingIndex + i))
					j += 1
				elif dim == "depth":
					depth = data[..., j]
					print(npGetInfo(depth))
					plot_image(depth, cmap="hot", new_figure=(startingIndex==1 and i==0), show_axis=False, \
						title="Depth", axis=(1, totalItems, startingIndex + i))
					j += 1
			return len(dims)

		for i in range(MB):
			data = datas[i]
			label = labels[i]
			totalItems = len(args.data_dims) + len(args.label_dims)
			ix = plotter(data, args.data_dims, reader.hvnTransform, startingIndex=1, totalItems=totalItems)
			plotter(label, args.label_dims, reader.hvnTransform, startingIndex=ix + 1, totalItems=totalItems)
			show_plots()