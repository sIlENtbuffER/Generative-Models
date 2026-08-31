#ifndef VAE_TRAINER_CUH
#define VAE_TRAINER_CUH

#include "core/rng.cuh"
#include "models/vae.cuh"
#include "optimizers/optimizer.cuh"
#include "data/data.h"

int vae_train_batch
(
    VAE *vae,
    Optimizer *optimizer,
    uint64_t *seed,
    VAEWorkspace *vae_for_ws,
    VAEWorkspace *vae_bac_ws,
    VAELoss *loss
);

int vae_train_epoch
(
    VAE *vae,
    Optimizer *optimizer,
    uint64_t *seed,
    size_t batch_size,
    size_t epoch,
    Data *data,
    Data *samples,
    const char *sample_dir
);

int vae_sample
(
    const VAE *vae,
    uint64_t *seed,
    Data *output
);

#endif
