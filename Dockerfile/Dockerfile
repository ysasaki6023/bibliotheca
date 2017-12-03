FROM nvidia/cuda:7.5-cudnn5-devel

RUN apt-get update -y \
&& DEBIAN_FRONTEND=noninteractive apt-get install -y -q \
    git \
    python3-dev \
    python3-pip \
    libjpeg-dev \
    libpng-dev \
    libtiff5-dev \
    libjasper-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    pkg-config \
    cmake \
    libgtk2.0-dev \
    libeigen3-dev \
    libtheora-dev \
    libvorbis-dev \
    libxvidcore-dev \
    libx264-dev \
    sphinx-common \
    libtbb-dev \
    yasm \
    libfaac-dev \
    libopencore-amrnb-dev \
    libopencore-amrwb-dev \
    libopenexr-dev \
    libgstreamer-plugins-base1.0-dev \
    libavcodec-dev \
    libavutil-dev \
    libavfilter-dev \
    libavformat-dev \
    libavresample-dev \
    qt5-default \
    ffmpeg \
    vim \
    wget && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN git clone --depth 100 https://github.com/opencv/opencv.git \
    && cd opencv \
    && git checkout 3.3.1 \
    && mkdir build \
    && cd build \
    && cmake \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_FFMPEG=ON \
        -DWITH_TBB=ON \
        -DWITH_CUDA=ON \
        -DWITH_CUBLAS=ON \
        -DENABLE_FAST_MATH=1 \
        -DCUDA_FAST_MATH=1 \
        -DWITH_GTK=OFF \
        -DBUILD_DOCS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DINSTALL_C_EXAMPLES=OFF \
        -DINSTALL_PYTHON_EXAMPLES=OFF \
        -DCUDA_NVCC_FLAGS="-D_FORCE_INLINES" .. \
    && make -j${NB_PROCS} \
    && make install \
    && /bin/bash -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf' \
    && ldconfig \
    && cd /root \
    && rm -fr opencv

RUN git clone https://github.com/ysasaki6023/settings.git \
    && cd settings \
    && cp .vimrc ~/

# NOTE: Tensorflow 1.4.0 was compiled using cuDNN version 6, so the base image must use it.
RUN pip3 install --upgrade pip3 && \
    cupy==2.1.0 \
    chainer==3.1.0 \
    tensorflow-gpu==1.4.0 \
    ipython==5.5.0 \
    pandas==0.21.0 \
    jupyter==1.0.0 \
    keras==2.0.6 \
    imgaug==0.2.5 \
    h5py


WORKDIR /dev
ENV HOME /dev
