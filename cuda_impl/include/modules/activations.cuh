#ifndef ACTIVATIONS_CUH
#define ACTIVATIONS_CUH

#include "core/tensor.cuh"

int relu_forward
(
    const DeviceTensor *x,
    DeviceTensor *output
);

int relu_backward
(
    const DeviceTensor *fw_input,
    const DeviceTensor *grad,
    DeviceTensor *output
);

int sigmoid_forward
(
    const DeviceTensor *x,
    DeviceTensor *output
);


int sigmoid_backward
(
    const DeviceTensor *fw_output,
    const DeviceTensor *grad,
    DeviceTensor *output
);

int leaky_relu_forward
(
    const DeviceTensor *x,
    float slope,
    DeviceTensor *output
);

int leaky_relu_backward
(
    const DeviceTensor *fw_input,
    const DeviceTensor *grad,
    float slope,
    DeviceTensor *output
);

int tanh_forward
(
    const DeviceTensor *x,
    DeviceTensor *output
);

int tanh_backward
(
    const DeviceTensor *fw_output,
    const DeviceTensor *grad,
    DeviceTensor *output
);

#endif
