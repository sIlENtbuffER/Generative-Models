#ifndef TENSOR_CUH
#define TENSOR_CUH

#include <stddef.h>

#define TENSOR_MAX_DIMS 8

typedef struct
{
    size_t ndim;
    size_t shape[TENSOR_MAX_DIMS];
    size_t strides[TENSOR_MAX_DIMS];
    size_t numel;
    float *data;
} DeviceTensor;

int tensor_alloc
(
    DeviceTensor *tensor,
    size_t ndim,
    const size_t *shape
);

int tensor_alloc_1d
(
    DeviceTensor *tensor,
    size_t dim0
);

int tensor_alloc_2d
(
    DeviceTensor *tensor,
    size_t dim0,
    size_t dim1
);

void tensor_free(DeviceTensor *tensor);

int tensor_fill
(
    DeviceTensor *tensor,
    float value
);

int tensor_is_same_shape
(
    const DeviceTensor *a,
    const DeviceTensor *b
);

int tensor_matmul
(
    const DeviceTensor *a,
    const DeviceTensor *b,
    DeviceTensor *output
);

int tensor_transpose
(
    const DeviceTensor *input,
    const size_t *axes,
    DeviceTensor *output
);

int tensor_device_to_host
(
    const DeviceTensor *src,
    float *dst
);

int tensor_host_to_device
(
    const float *src,
    DeviceTensor *dst
);

#endif
