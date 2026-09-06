#ifndef MODEL_H
#define MODEL_H

#include "core/parameter.h"
#include "core/rng.h"
#include "tools/checkpoint.h"
#include "tools/config.h"
#include "optimizers/optimizer.h"
#include "data/data.h"

#include <stddef.h>

typedef enum
{
    MODEL_NONE,
    MODEL_VAE,
    MODEL_GAN
} ModelType;

typedef struct Model
{
    ModelType type;
    void *implementation;

    Parameter *parameters;
    size_t num_parameters;
} Model;

int model_build
(
    Model *model,
    const char *name,
    size_t input_dim,
    const cJSON *model_cfg,
    RNG *rng
);

void model_free(Model *model);

int model_save_checkpoint
(
    const Model *model,
    Checkpoint *checkpoint
);

int model_load_checkpoint
(
    Model *model,
    const Checkpoint *checkpoint
);

#endif
