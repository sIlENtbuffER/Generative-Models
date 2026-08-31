#ifndef ADAM_CUH
#define ADAM_CUH

#include "core/parameter.cuh"
#include "tools/checkpoint.h"

typedef struct
{
    Parameter parameter;
    DeviceTensor m;
    DeviceTensor v;
} AdamElement;

typedef struct
{
    AdamElement *adam_element;
    size_t num_parameters;
    float lr;
    float beta1;
    float beta2;
    float beta1_power;
    float beta2_power;
    float eps;
    size_t step;
} Adam;

int adam_alloc
(
    Adam *adam,
    const Parameter *parameters,
    size_t num_parameters,
    float lr,
    float beta1,
    float beta2,
    float eps
);

void adam_free(Adam *adam);

int adam_step(Adam *adam);

int adam_save_checkpoint
(
    Adam *adam,
    Checkpoint *checkpoint
);

int adam_load_checkpoint
(
    Adam *adam,
    const Checkpoint *checkpoint
);

#endif
