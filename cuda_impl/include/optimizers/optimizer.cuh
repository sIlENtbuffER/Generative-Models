#ifndef OPTIMIZER_CUH
#define OPTIMIZER_CUH

#include "core/parameter.cuh"
#include "tools/checkpoint.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum
{
    OPTIMIZER_NONE,
    OPTIMIZER_ADAM
} OptimizerType;

typedef struct Optimizer
{
    OptimizerType type;
    void *implementation;
} Optimizer;

int optimizer_build
(
    Optimizer *optimizer,
    const char *name,
    const Parameter *parameters,
    size_t num_parameters,
    float learning_rate,
    float beta1,
    float beta2,
    float eps
);

void optimizer_free(Optimizer *optimizer);

int optimizer_step(Optimizer *optimizer);

int optimizer_save_checkpoint
(
    Optimizer *optimizer,
    Checkpoint *checkpoint
);

int optimizer_load_checkpoint
(
    Optimizer *optimizer,
    const Checkpoint *checkpoint
);

#ifdef __cplusplus
}
#endif

#endif
