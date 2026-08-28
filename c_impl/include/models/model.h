#ifndef MODEL_H
#define MODEL_H

#include "core/parameter.h"
#include "core/rng.h"
#include "tools/checkpoint.h"
#include "optimizers/optimizer.h"
#include "data/data.h"

#include <stddef.h>

typedef enum
{
    MODEL_NONE,
    MODEL_VAE
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
    size_t hidden_dim,
    size_t latent_dim,
    RNG *rng
);

void model_free(Model *model);

int model_save_checkpoint
(
    const Model *model,
    Checkpoint *checkpoint
);

#endif
