#include "optimizers/adam.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

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
    adam->adam_element = calloc(num_parameters, sizeof *adam->adam_element);
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
        Tensor *value = element->parameter.value;
        const Tensor *grad = element->parameter.grad;
        for (size_t i=0; i<value->numel; i++) {
            float g = grad->data[i];

            element->m.data[i] = adam->beta1 * element->m.data[i] + (1.0f - adam->beta1) * g;
            element->v.data[i] = adam->beta2 * element->v.data[i] + (1.0f - adam->beta2) * g * g;

            float m_hat = element->m.data[i] / (1.0f - adam->beta1_power);
            float v_hat = element->v.data[i] / (1.0f - adam->beta2_power);

            value->data[i] -= adam->lr * m_hat / (sqrtf(v_hat) + adam->eps);
        }
    }
    return 0;
}

int adam_save_checkpoint(Adam *adam, Checkpoint *checkpoint) {
    char name[CHECKPOINT_TENSOR_NAME_SIZE];

    for (size_t i=0; i<adam->num_parameters; i++) {
        AdamElement *element = &adam->adam_element[i];
        snprintf(name, sizeof name, "optimizer.m.%s", element->parameter.name);
        if (checkpoint_add_tensor(checkpoint, name, &element->m) != 0) return -1;

        snprintf(name, sizeof name, "optimizer.v.%s", element->parameter.name);
        if (checkpoint_add_tensor(checkpoint, name, &element->v) != 0) return -1;
    }

    char step[32];
    snprintf(step, sizeof step, "%zu", adam->step);

    return checkpoint_set_metadata(checkpoint, "optimizer_step", step);
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
        if (cpt == NULL) return -1;
        memcpy(element->m.data, cpt->data, cpt->numel * sizeof *cpt->data);

        snprintf(name, sizeof name, "optimizer.v.%s", element->parameter.name);
        cpt = checkpoint_get_tensor(checkpoint, name);
        if (cpt == NULL) return -1;
        memcpy(element->v.data, cpt->data, cpt->numel * sizeof *cpt->data);
    }

    return 0;
}
