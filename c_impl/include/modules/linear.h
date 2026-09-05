#ifndef LINEAR_H
#define LINEAR_H

#include "core/tensor.h"
#include "core/rng.h"

#include <stddef.h>
#include <math.h>

typedef struct
{
    size_t in_dim;
    size_t out_dim;

    Tensor W;
    Tensor b;

    Tensor dW;
    Tensor db;
} Linear;

int linear_alloc
(
    Linear *linear,
    size_t in_dim,
    size_t out_dim
);

void linear_free(Linear *linear);

int linear_he_init
(
    Linear *linear,
    RNG *rng
);

int linear_normal_init
(
    Linear *linear,
    RNG *rng,
    float std
);

int linear_forward
(
    const Linear *linear,
    const Tensor *x,
    Tensor *output
);

int linear_backward
(
    Linear *linear,
    const Tensor *x,
    const Tensor *grad,
    Tensor *output
);

#endif
