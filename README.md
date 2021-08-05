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

# Installation for Jetson

    $ sudo apt update && sudo apt -y upgrade
    $ sudo apt -y install < packages.txt

    $ pip3 install --upgrade pip
    $ pip3 install -r requirements.txt

    $ cd ..
    $ git clone https://gitlab.com/mihaicristianpirvu/neural-wrappers.git
    $ cd neural-wrappers
    $ git checkout 3dcc404b08f0e356904d1a1dd16382c3ae4aa752


## Use Docker Image
Docker version >= 20.10.

You can download pre-train model, datasets and Docker image [here][safeuav/data].


    $ docker build -t <repository> docker/

Run

    $ docker run --privileged --gpus all --it -v <data>:/SafeUAV/SafeUAV/data <repository>


[safeuav/data]: <https://aistmail-my.sharepoint.com/:f:/r/personal/ishitsuka_hikaru_aist_go_jp/Documents/aipj/data/SafeUAV?csf=1&web=1&e=AdgeZI>