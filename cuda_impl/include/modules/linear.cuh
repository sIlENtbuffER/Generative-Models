#ifndef LINEAR_CUH
#define LINEAR_CUH

#include "core/tensor.cuh"
#include "core/rng.cuh"

#include <stddef.h>
#include <math.h>

typedef struct
{
    size_t in_dim;
    size_t out_dim;

    DeviceTensor W;
    DeviceTensor b;

    DeviceTensor dW;
    DeviceTensor db;

    DeviceTensor x_T;
    DeviceTensor W_T;
} Linear;

int linear_alloc
(
    Linear *linear,
    size_t in_dim,
    size_t out_dim
);

void linear_free(Linear *linear);

int linear_he_init(
    Linear *linear,
    uint64_t *seed
);

int linear_forward
(
    const Linear *linear,
    const DeviceTensor *x,
    DeviceTensor *output
);

int linear_backward
(
    Linear *linear,
    const DeviceTensor *x,
    const DeviceTensor *grad,
    DeviceTensor *output
);

#endif
