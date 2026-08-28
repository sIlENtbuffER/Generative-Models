#include "modules/activations.h"

#include <math.h>

int relu_forward(const Tensor *x, Tensor *output) {
    if (!tensor_is_same_shape(x, output)) return -1;
    
    for (size_t i=0; i<x->numel; i++) {
        output->data[i] = x->data[i] > 0.0f? x->data[i] : 0.0f;
    }

    return 0;
}

int relu_backward(const Tensor *fw_input, const Tensor *grad, Tensor *output) {
    if (!tensor_is_same_shape(grad, fw_input) || !tensor_is_same_shape(grad, output)) return -1;

    for (size_t i=0; i<grad->numel; i++) {
        output->data[i] = fw_input->data[i] > 0.0f? grad->data[i] : 0.0f;
    }

    return 0;
}


int sigmoid_forward(const Tensor *x, Tensor *output) {
    if (!tensor_is_same_shape(x, output)) return -1;

    for (size_t i=0; i<x->numel; i++) {
        // Avoid overflow
        float value = x->data[i];
        if (value >= 0.0f) {
            output->data[i] = 1.0f / (1.0f + expf(-value));
        } else {
            output->data[i] = expf(value) / (1.0f + expf(value));
        }
    }

    return 0;
}

int sigmoid_backward(const Tensor *fw_output, const Tensor *grad, Tensor *output) {
    if (!tensor_is_same_shape(grad, fw_output) || !tensor_is_same_shape(grad, output)) return -1;

    for (size_t i=0; i<grad->numel; i++) {
        output->data[i] = grad->data[i] * fw_output->data[i] * (1.0f - fw_output->data[i]);
    }

    return 0;
}
