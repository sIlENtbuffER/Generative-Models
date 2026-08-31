#ifndef TRAINER_CUH
#define TRAINER_CUH

#include "core/rng.cuh"
#include "data/data.h"
#include "models/model.cuh"
#include "optimizers/optimizer.cuh"

#include <stddef.h>

int trainer_train
(
    Model *model,
    Optimizer *optimizer,
    uint64_t *seed,
    size_t batch_size,
    size_t start_epoch,
    size_t epochs,
    Data *data,
    size_t num_samples,
    const char *sample_dir,
    const char *checkpoint_dir
);

#endif
