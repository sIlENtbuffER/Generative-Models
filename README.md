# Generative Models, From Scratch

A learning project that works through generative models from first principles, no ML frameworks involved. Each model on the roadmap gets implemented up to three times, in increasing order of how much a framework normally hides from you:

1. **NumPy** (`numpy_impl/`) — get the algorithm itself right
2. **C** (`c_impl/`) — hand-written tensors and memory management
3. **CUDA** (`cuda_impl/`) — hand-written GPU kernels

See [AGENTS.md](AGENTS.md) for the full methodology and project rules.

## Roadmap

![Generative models learning roadmap](docs/roadmap.png)

## Project layout

```
numpy_impl/     NumPy reference implementations
c_impl/         C implementations
cuda_impl/      CUDA implementations
configs/        JSON training configs
data/           datasets
```

## Setup

### Environment

Needs conda (or miniconda) and a C compiler.

```bash
git clone https://github.com/sIlENtbuffER/Generative-Models.git
cd Generative-Models
conda env create -f environment.yml
conda activate genmodels_NV_CU130_py312
```

`environment.yml` pulls in `gcc_linux-64`, which targets Linux x86-64. On macOS or Linux ARM, swap that line before creating the env:

```bash
# macOS (Intel)
sed 's/gcc_linux-64/clang_osx-64/' environment.yml > /tmp/environment.yml && conda env create -f /tmp/environment.yml

# macOS (Apple Silicon)
sed 's/gcc_linux-64/clang_osx-arm64/' environment.yml > /tmp/environment.yml && conda env create -f /tmp/environment.yml

# Linux ARM (aarch64)
sed 's/gcc_linux-64/gcc_linux-aarch64/' environment.yml > /tmp/environment.yml && conda env create -f /tmp/environment.yml
```

### Datasets

MNIST (grayscale, 28×28):

```bash
mkdir -p data/MNIST/raw
cd data/MNIST/raw
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
    curl -LO "https://ossci-datasets.s3.amazonaws.com/mnist/${f}.gz"
    gunzip "${f}.gz"
done
cd -
```

CelebA (color, 64×64):

```bash
mkdir -p data/CelebA
curl -L -r 0-250000000 -o data/CelebA/img_align_celeba.zip "https://huggingface.co/datasets/Yuehao/celeba/resolve/main/img_align_celeba.zip"
python scripts/prepare_celeba.py
```

JPEG decoding isn't worth hand-writing, so [`scripts/prepare_celeba.py`](scripts/prepare_celeba.py) does the preprocessing once with Pillow. It center-crops each image and resizes to 64×64, and writes `data/CelebA/celeba.bin` as a small header plus raw CHW `uint8` pixels. `COUNT` at the top of the script sets how many images to use; raise the byte range above it if you want more.

## Running

Both implementations take a JSON config path as their first argument — hyperparameters, dataset, and where to write samples/checkpoints — and fall back to [`configs/default.json`](configs/default.json) when none is given.

### NumPy

```bash
python numpy_impl/train.py configs/default.json
```

### C / CUDA

```bash
cmake -G Ninja -B build
cmake --build build
```

This always builds `./build/genmodels_train_c`. If a CUDA toolkit is available, it's picked up automatically and `./build/genmodels_train_cuda` is built alongside it.

```bash
./build/genmodels_train_c configs/default.json       # CPU
./build/genmodels_train_cuda configs/default.json    # GPU, if CUDA is available
```
