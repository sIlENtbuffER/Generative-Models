#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

#define TENSOR_MAX_DIMS 8 // For cuda use

typedef struct
{
    size_t ndim;
    size_t shape[TENSOR_MAX_DIMS];
    size_t strides[TENSOR_MAX_DIMS];
    size_t numel;
    float *data;
} Tensor;

int tensor_alloc
(
    Tensor *tensor,
    size_t ndim,
    const size_t *shape
);

int tensor_alloc_1d
(
    Tensor *tensor,
    size_t dim0
);

int tensor_alloc_2d
(
    Tensor *tensor,
    size_t dim0,
    size_t dim1
);

void tensor_free(Tensor *tensor);

void tensor_fill
(
    Tensor *tensor,
    float value
);

int tensor_is_same_shape
(
    const Tensor *a,
    const Tensor *b
);

int tensor_get_addr
(
    const Tensor *tensor,
    const size_t *indices,
    size_t *address
);

int tensor_get
(
    const Tensor *tensor,
    const size_t *indices,
    float *value
);

int tensor_set
(
    Tensor *tensor,
    const size_t *indices,
    float value
);

int tensor_reshape
(
    Tensor *tensor,
    size_t ndim,
    const size_t *shape
);

int tensor_matvec
(
    const Tensor *matrix,
    const Tensor *vector,
    Tensor *output
);

int tensor_matmul_output_shape
(
    const Tensor *a,
    const Tensor *b,
    size_t *output_ndim,
    size_t output_shape[TENSOR_MAX_DIMS]
);

int tensor_matmul
(
    const Tensor *a,
    const Tensor *b,
    Tensor *output
);

int tensor_transpose
(
    const Tensor *input,
    const size_t *axes,
    Tensor *output
);

#endif
