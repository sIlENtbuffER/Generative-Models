#ifndef TRAINER_CUH
#define TRAINER_CUH

#include "core/rng.cuh"
#include "data/data.h"
#include "models/model.cuh"
#include "optimizers/optimizer.cuh"
#include "tools/config.h"

#include <stddef.h>

int trainer_train
(
    Model *model,
    const Config *config,
    uint64_t *seed,
    Data *data
);

#endif
