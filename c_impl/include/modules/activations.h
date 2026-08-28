#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "core/tensor.h"

int relu_forward
(
    const Tensor *x,
    Tensor *output
);

int relu_backward
(
    const Tensor *fw_input,
    const Tensor *grad,
    Tensor *output
);

int sigmoid_forward
(
    const Tensor *x,
    Tensor *output
);


int sigmoid_backward
(
    const Tensor *fw_output,
    const Tensor *grad,
    Tensor *output
);

#endif
