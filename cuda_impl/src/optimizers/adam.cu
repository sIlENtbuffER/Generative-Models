#include "optimizers/adam.cuh"
#include "core/cuda.cuh"
#include "core/tensor.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

__global__ static void adam_step_kernel(float *m, float *v, float *value, const float *grad, size_t numel, float beta1, float beta2, float beta1_power, float beta2_power, float lr, float eps) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;

    m[i] = beta1 * m[i] + (1.0f - beta1) * grad[i];
    v[i] = beta2 * v[i] + (1.0f - beta2) * grad[i] * grad[i];

    float m_hat = m[i] / (1.0f - beta1_power);
    float v_hat = v[i] / (1.0f - beta2_power);

    value[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

static int adam_element_alloc(AdamElement *adam_element, const Parameter *parameter) {
    *adam_element = (AdamElement){0};

    if (tensor_alloc(&adam_element->m, parameter->value->ndim, parameter->value->shape) != 0 || tensor_alloc(&adam_element->v, parameter->value->ndim, parameter->value->shape) != 0) {
        tensor_free(&adam_element->m);
        tensor_free(&adam_element->v);
        return -1;
    }
    adam_element->parameter = *parameter;
    return 0;
}

int adam_alloc(Adam *adam, const Parameter *parameters, size_t num_parameters, float lr, float beta1, float beta2, float eps) {
    *adam = (Adam){0};
    if (lr <= 0.0f || beta1 < 0.0f || beta1 >= 1.0f || beta2 < 0.0f || beta2 >= 1.0f || eps <= 0.0f) return -1;
    adam->adam_element = (AdamElement*)calloc(num_parameters, sizeof *adam->adam_element);
    if (adam->adam_element == NULL) return -1;
    
    adam->num_parameters = num_parameters;
    adam->lr = lr;
    adam->beta1 = beta1;
    adam->beta2 = beta2;
    adam->beta1_power = 1.0f;
    adam->beta2_power = 1.0f;
    adam->eps = eps;
    adam->step = 0;

    for (size_t i=0; i<num_parameters; i++) {
        if (adam_element_alloc(&adam->adam_element[i], &parameters[i]) != 0) {
            adam_free(adam);
            return -1;
        }
    }

    return 0;
}

void adam_free(Adam *adam) {
    for (size_t i = 0; i < adam->num_parameters; i++) {
        tensor_free(&adam->adam_element[i].m);
        tensor_free(&adam->adam_element[i].v);
    }

    free(adam->adam_element);
    *adam = (Adam){0};
}

int adam_step(Adam *adam) {
    adam->step++;
    adam->beta1_power *= adam->beta1;
    adam->beta2_power *= adam->beta2;

    for (size_t p=0; p<adam->num_parameters; p++) {
        AdamElement *element = &adam->adam_element[p];
        DeviceTensor *value = element->parameter.value;
        const DeviceTensor *grad = element->parameter.grad;
        adam_step_kernel<<<cuda_blocks(value->numel), THREADS_PER_BLOCK>>>(
        element->m.data, element->v.data, element->parameter.value->data, element->parameter.grad->data,
        value->numel, adam->beta1, adam->beta2, adam->beta1_power, adam->beta2_power, adam->lr, adam->eps);
    }

    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int adam_save_checkpoint(Adam *adam, Checkpoint *checkpoint) {
    char name[CHECKPOINT_TENSOR_NAME_SIZE];
    Tensor host = {0};

    for (size_t i=0; i<adam->num_parameters; i++) {
        AdamElement *element = &adam->adam_element[i];

        if (tensor_alloc(&host, element->m.ndim, element->m.shape) != 0) goto fail;
        if (tensor_device_to_host(&element->m, host.data) != 0) goto fail;
        snprintf(name, sizeof name, "optimizer.m.%s", element->parameter.name);
        if (checkpoint_take_tensor(checkpoint, name, &host) != 0) goto fail;

        if (tensor_alloc(&host, element->v.ndim, element->v.shape) != 0) goto fail;
        if (tensor_device_to_host(&element->v, host.data) != 0) goto fail;
        snprintf(name, sizeof name, "optimizer.v.%s", element->parameter.name);
        if (checkpoint_take_tensor(checkpoint, name, &host) != 0) goto fail;
    }

    char step[32];
    snprintf(step, sizeof step, "%zu", adam->step);

    tensor_free(&host);

    return checkpoint_set_metadata(checkpoint, "optimizer_step", step);

fail:
    tensor_free(&host);
    return -1;
}

int adam_load_checkpoint(Adam *adam, const Checkpoint *checkpoint) {
    char name[CHECKPOINT_TENSOR_NAME_SIZE];
    const Tensor *cpt;

    const char *step_text = checkpoint_get_metadata(checkpoint, "optimizer_step");
    if (step_text == NULL) return -1;
    adam->step = (size_t)strtoull(step_text, NULL, 10);

    adam->beta1_power = powf(adam->beta1, (float)adam->step);
    adam->beta2_power = powf(adam->beta2, (float)adam->step);

    for (size_t i=0; i<adam->num_parameters; i++) {
        AdamElement *element = &adam->adam_element[i];
        snprintf(name, sizeof name, "optimizer.m.%s", element->parameter.name);
        cpt = checkpoint_get_tensor(checkpoint, name);
        tensor_host_to_device(cpt->data, &element->m);

        snprintf(name, sizeof name, "optimizer.v.%s", element->parameter.name);
        cpt = checkpoint_get_tensor(checkpoint, name);
        tensor_host_to_device(cpt->data, &element->v);
    }
    
    return 0;
}
