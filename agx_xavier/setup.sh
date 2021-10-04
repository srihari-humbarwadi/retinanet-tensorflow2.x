sudo apt install -y \
    python3-scipy \
    python3-h5py \
    python3-pil \
    python3-matplotlib \
    libhdf5-serial-dev \
    hdf5-tools \
    libhdf5-dev \
    zlib1g-dev \
    zip \
    libjpeg8-dev \
    liblapack-dev \
    libblas-dev \
    gfortran

sudo pip3 install -U pip
sudo pip3 install cython tqdm easydict testresources setuptools==49.6.0
sudo pip3 install -U --no-deps \
    numpy==1.19.4 \
    future==0.18.2 \
    mock==3.0.5 \
    keras==2.6.0 \
    keras_preprocessing==1.1.2 \
    keras_applications==1.0.8 \
    gast==0.4.0 \
    onnx==1.10.1 \
    futures \
    protobuf \
    pybind11 \
    pkgconfig \
    pycocotools

sudo ln -s /usr/include/locale.h /usr/include/xlocale.h
sudo env H5PY_SETUP_REQUIRES=0 PATH=/usr/local/cuda/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH pip3 install pycuda h5py==3.1.0

wget https://nvidia.box.com/shared/static/bfs688apyvor4eo8sf3y1oqtnarwafww.whl -O /tmp/onnxruntime_gpu-1.9.0-cp36-cp36m-linux_aarch64.whl
sudo pip3 install /tmp/onnxruntime_gpu-1.9.0-cp36-cp36m-linux_aarch64.whl

sudo pip3 install git+https://github.com/onnx/tensorflow-onnx@446494eea2fb80b4beca13914811678399c91e11
sudo pip3 install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com
sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow
