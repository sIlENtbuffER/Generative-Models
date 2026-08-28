#ifndef TRAINER_H
#define TRAINER_H

#include "core/rng.h"
#include "data/data.h"
#include "models/model.h"
#include "optimizers/optimizer.h"

#include <stddef.h>

int trainer_train
(
    Model *model,
    Optimizer *optimizer,
    RNG *rng,
    size_t batch_size,
    size_t start_epoch,
    size_t epochs,
    Data *data,
    size_t num_samples,
    const char *sample_dir,
    const char *checkpoint_dir
);

#endif
