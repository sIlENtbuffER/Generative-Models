#ifndef GAN_TRAINER_CUH
#define GAN_TRAINER_CUH

#include "core/rng.cuh"
#include "models/gan.cuh"
#include "optimizers/optimizer.cuh"
#include "data/data.h"

int gan_train_batch
(
    GAN *gan,
    Optimizer *g_optimizer,
    Optimizer *d_optimizer,
    uint64_t *seed,
    const DeviceTensor *x_real,
    GANWorkspace *ganws,
    GANLoss *loss
);

int gan_train_epoch
(
    GAN *gan,
    Optimizer *g_optimizer,
    Optimizer *d_optimizer,
    uint64_t *seed,
    size_t batch_size,
    size_t epoch,
    Data *data,
    Data *samples,
    const char *sample_dir
);

int gan_sample
(
    const Generator *generator,
    uint64_t *seed,
    Data *output
);

#endif
