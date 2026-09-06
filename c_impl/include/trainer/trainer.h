#ifndef TRAINER_H
#define TRAINER_H

#include "core/rng.h"
#include "data/data.h"
#include "models/model.h"
#include "optimizers/optimizer.h"
#include "tools/config.h"

#include <stddef.h>

int trainer_train
(
    Model *model,
    const Config *config,
    RNG *rng,
    Data *data
);

#endif
