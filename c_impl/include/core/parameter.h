#ifndef PARAMETER_H
#define PARAMETER_H

#include "core/tensor.h"

typedef struct
{
    const char *name;
    Tensor *value;
    const Tensor *grad;
} Parameter;

#endif
