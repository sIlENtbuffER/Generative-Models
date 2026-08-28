#ifndef VAE_TRAINER_H
#define VAE_TRAINER_H

#include "core/rng.h"
#include "models/vae.h"
#include "optimizers/optimizer.h"
#include "data/data.h"

int vae_train_batch
(
    VAE *vae,
    Optimizer *optimizer,
    RNG *rng,
    VAEWorkspace *vae_for_ws,
    VAEWorkspace *vae_bac_ws,
    VAELoss *loss
);

int vae_train_epoch
(
    VAE *vae,
    Optimizer *optimizer,
    RNG *rng,
    size_t batch_size,
    size_t epoch,
    Data *data,
    Data *samples,
    const char *sample_dir
);

int vae_sample
(
    const VAE *vae,
    RNG *rng,
    Data *output
);

#endif
