FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

WORKDIR /SafeUAV

ENV TZ=Asia/Tokyo

COPY requirements.txt $WORKDIR
COPY datasets/ $WORKDIR
COPY snapshots/ $WORKDIR
COPY videos/ $WORKDIR

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y install tzdata

RUN apt-get -y install python3-numpy python3-opencv python3-pip emacs git cmake build-essential libjpeg-dev libpng-dev
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

RUN git clone https://gitlab.com/mihaicristianpirvu/SafeUAV.git
RUN git clone https://gitlab.com/mihaicristianpirvu/Mihlib.git
RUN cd Mihlib && \
    python3 build-cpp.py && \
    $WORKDIR
RUN git clone https://gitlab.com/mihaicristianpirvu/neural-wrappers.git
RUN cd neural-wrappers && \
    git checkout 3dcc404b08f0e356904d1a1dd16382c3ae4aa752 && \
    cd $WORKDIR

ENV PYTHONPATH=$PYTHONPATH:$WORKDIR/neural-wrappers:$WORKDIR/Mihlib/src
