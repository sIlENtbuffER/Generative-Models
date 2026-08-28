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

### Dataset

```bash
mkdir -p data/MNIST/raw
cd data/MNIST/raw
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
    curl -LO "https://ossci-datasets.s3.amazonaws.com/mnist/${f}.gz"
    gunzip "${f}.gz"
done
cd -
```

## Running

Both implementations read [`configs/default.json`](configs/default.json) for hyperparameters, dataset, and where to write samples/checkpoints.

### NumPy

```bash
python numpy_impl/train.py
```

### C

```bash
cmake -G Ninja -B build
cmake --build build
./build/vae_train_c
```
