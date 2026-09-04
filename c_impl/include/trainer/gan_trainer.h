#ifndef GAN_TRAINER_H
#define GAN_TRAINER_H

#include "core/rng.h"
#include "models/gan.h"
#include "optimizers/optimizer.h"
#include "data/data.h"

int gan_train_batch
(
    GAN *gan,
    Optimizer *g_optimizer,
    Optimizer *d_optimizer,
    RNG *rng,
    const Tensor *x_real,
    GANWorkspace *ganws,
    GANLoss *loss
);

int gan_train_epoch
(
    GAN *gan,
    Optimizer *g_optimizer,
    Optimizer *d_optimizer,
    RNG *rng,
    size_t batch_size,
    size_t epoch,
    Data *data,
    Data *samples,
    const char *sample_dir
);

int gan_sample
(
    const Generator *generator,
    RNG *rng,
    Data *output
);

#endif
