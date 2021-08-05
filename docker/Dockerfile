FROM nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.7-py3

WORKDIR /SafeUAV

ENV TZ=Asia/Tokyo
ENV DEBIAN_FRONTEND=noninteractive

COPY requirements.txt $WORKDIR
# COPY datasets/ $WORKDIR
# COPY snapshots/ $WORKDIR
# COPY videos/ $WORKDIR

RUN apt-get update && apt-get -y upgrade && apt-get install -y \
    python3-opencv \
    python3-h5py \
    emacs \
    cmake \
    build-essential \
    libjpeg-dev \
    libpng-dev \
    ffmpeg

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# RUN git clone https://gitlab.com/mihaicristianpirvu/SafeUAV.git
RUN git clone https://github.com/ishitsuka-hikaru/SafeUAV.git
# RUN git clone https://gitlab.com/mihaicristianpirvu/neural-wrappers.git
# RUN cd neural-wrappers && \
#     git checkout 3dcc404b08f0e356904d1a1dd16382c3ae4aa752 && \
#     cd $WORKDIR
RUN git clone https://github.com/ishitsuka-hikaru/neural-wrappers.git
