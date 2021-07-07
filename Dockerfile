#FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
#FROM nvidia/cuda-arm64:11.0-devel-ubuntu18.04
FROM nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.7-py3

WORKDIR /SafeUAV

ENV TZ=Asia/Tokyo
ENV DEBIAN_FRONTEND=noninteractive

COPY requirements.txt $WORKDIR
COPY datasets/ $WORKDIR
COPY snapshots/ $WORKDIR
COPY videos/ $WORKDIR

RUN apt-get -y update
RUN apt-get -y upgrade

#RUN apt-get -y install python3-numpy python3-opencv python3-pip emacs git cmake build-essential libjpeg-dev libpng-dev
RUN apt-get -y install python3-opencv python3-h5py emacs cmake build-essential libjpeg-dev libpng-dev ffmpeg

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

RUN git clone https://gitlab.com/mihaicristianpirvu/SafeUAV.git
RUN git clone https://gitlab.com/mihaicristianpirvu/Mihlib.git
RUN cd Mihlib && \
    python3 build-cpp.py && \
    cd $WORKDIR

RUN git clone https://gitlab.com/mihaicristianpirvu/neural-wrappers.git
RUN cd neural-wrappers && \
    git checkout 3dcc404b08f0e356904d1a1dd16382c3ae4aa752 && \
    cd $WORKDIR

ENV PYTHONPATH=$PYTHONPATH:$WORKDIR/neural-wrappers:$WORKDIR/Mihlib/src
