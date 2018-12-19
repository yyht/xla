#!/bin/bash

# Fail on any error.
set -e
# Display commands being run.
set -x

# Initialize all third_party submodules
git submodule update --init --recursive

<<<<<<< HEAD
# Build torch_xla wheel in conda environment
=======
# Install required packages for build
sudo apt-get -y install python-pip git
sudo pip install --upgrade google-api-python-client
sudo pip install --upgrade oauth2client

# Install conda environment
curl -O https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
sh Anaconda3-5.2.0-Linux-x86_64.sh -b
export PATH="$HOME/anaconda3/bin:$PATH"

# Setup conda env
conda create --name pytorch python=3.5 anaconda
source activate pytorch
export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
conda install -y numpy pyyaml mkl mkl-include setuptools cmake cffi typing bazel

# Install torch within conda env
# TODO(jysohn): once pytorch/pytorch JIT bug is fixed install nightly wheel instead
sudo /sbin/ldconfig "${HOME}/anaconda3/lib/" "${HOME}/anaconda3/envs/pytorch/lib"
pip install ../../../gfile/torch-1.0.0a0+$(head -c 7 .torch_commit_id)-cp35-cp35m-linux_x86_64.whl

# Build pytorch-wheel in conda environment
>>>>>>> ca244f8a0fdb7cad690f2975ea08d6968859781e
export NO_CUDA=1
python setup.py bdist_wheel

# Artifacts for pytorch-tpu wheel build collected (as nightly and with date)
ls -lah dist
mkdir build_artifacts
cp dist/* build_artifacts
cd dist && rename "s/\+\w{7}/\+nightly/" *.whl && cd ..
cd build_artifacts && rename "s/^torch_xla/torch_xla-$(date -d "yesterday" +%Y%m%d)/" *.whl && cd ..
mv dist/* build_artifacts
mv build_artifacts/* ../../../

