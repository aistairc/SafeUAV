# SafeUAV: Learning to estimate depth and safe landing areas for UAVs from synthetic data

This repository holds the implementation of the paper, presented at the UAVision2018 workshop (ECCV).

https://sites.google.com/site/aerialimageunderstanding/safeuav-learning-to-estimate-depth-and-safe-landing-areas-for-uavs (for checkpoints, paper and other information).

## Standard variables
```sh
model=unet_tiny_sun/unet_big_concatenate/deeplabv3plus/unet_classic (pick one)
dir=test_dir
lr=0.001
patience=4
factor=0.1
num_epochs=100
```

# Training a model

## HVO

```sh
python main.py train classification /path/to/dataset.h5 --model=$model --dir=$dir --label_dims=hvn_gt_p1 --batch_size=N --optimizer=Adam --learning_rate=$lr --patience=$patience --factor=$factor --num_epochs=$num_epochs
```

## Depth

```sh
python main.py train regression /path/to/dataset.h5 --model=$model --dir=$dir --label_dims=depth --batch_size=N --optimizer=Adam --learning_rate=$lr --patience=$patience --factor=$factor --num_epochs=$num_epochs
```


# Running a pre-trained model

## HVO

```sh
python main.py test classification /path/to/dataset.h5 --model=$model --weights_file=/path/to/checkpoint.pkl --test_plot_results=1 --label_dims=hvn_gt_p1 --batch_size=N
```

## Depth

```sh
python main.py test regression /path/to/dataset.h5 --model=$model --weights_file=/path/to/checkpoint.pkl --test_plot_results=1 --label_dims=depth --batch_size=N
```

# Running on an existing video (outputs another video)

## HVO
```sh
python main_inference_video.py classification in_video.mp4 out_video.mp4 --model=$model --weights_file=/path/to/checkpoint.pkl
```

## Depth
```sh
python main_inference_video.py regression in_video.mp4 out_video.mp4 --model=$model --weights_file=/path/to/checkpoint.pkl
```
