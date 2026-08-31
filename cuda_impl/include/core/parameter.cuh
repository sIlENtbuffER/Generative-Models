#ifndef PARAMETER_CUH
#define PARAMETER_CUH

#include "core/tensor.cuh"

typedef struct
{
    const char *name;
    DeviceTensor *value;
    const DeviceTensor *grad;
} Parameter;

#endif
