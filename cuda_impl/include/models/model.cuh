#ifndef MODEL_CUH
#define MODEL_CUH

#include "core/parameter.cuh"
#include "core/rng.cuh"
#include "tools/checkpoint.h"
#include "tools/config.h"
#include "optimizers/optimizer.cuh"
#include "data/data.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

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
    uint64_t *seed
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

#ifdef __cplusplus
}
#endif

#endif
