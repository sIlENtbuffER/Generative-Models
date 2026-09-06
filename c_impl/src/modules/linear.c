#include "modules/linear.h"

int linear_alloc(Linear *linear, size_t in_dim, size_t out_dim) {
    *linear = (Linear){0};
    linear->in_dim = in_dim;
    linear->out_dim = out_dim;

    if (tensor_alloc_2d(&linear->W, in_dim, out_dim) != 0 || tensor_alloc_1d(&linear->b, out_dim) != 0 || tensor_alloc_2d(&linear->dW, in_dim, out_dim) != 0 || tensor_alloc_1d(&linear->db, out_dim) != 0) {
        linear_free(linear);
        return -1;
    }
    return 0;
}

void linear_free(Linear *linear) {
    tensor_free(&linear->W); 
    tensor_free(&linear->b); 
    tensor_free(&linear->dW); 
    tensor_free(&linear->db);

    linear->in_dim = 0;
    linear->out_dim = 0;
}

int linear_he_init(Linear *linear, RNG *rng) {
    if (linear->W.data == NULL || linear->b.data == NULL || linear->in_dim == 0) return -1;
    
    float scale = sqrtf(2.0f / (float)linear->in_dim);
    for (size_t i=0; i<linear->W.numel; i++) {
        linear->W.data[i] = rng_normal(rng) * scale;
    }
    tensor_fill(&linear->b, 0.0f);
    
    return 0;
}

int linear_normal_init(Linear *linear, RNG *rng, float std) {
    if (linear->W.data == NULL || linear->b.data == NULL || linear->in_dim == 0) return -1;
    
    for (size_t i=0; i<linear->W.numel; i++) {
        linear->W.data[i] = rng_normal(rng) * std;
    }
    tensor_fill(&linear->b, 0.0f);
    
    return 0;
}

int linear_forward(const Linear *linear, const Tensor *x, Tensor *output) {
    if (tensor_matmul(x, &linear->W, output) != 0) return -1;

    for (size_t i=0; i<output->shape[0]; i++) {
        for (size_t j=0; j<output->shape[1]; j++) {
            output->data[i * output->shape[1] + j] += linear->b.data[j];
        }
    }

    return 0;
}

int linear_backward(Linear *linear, const Tensor *x, const Tensor *grad, Tensor *output) {
    Tensor x_T = {0};
    Tensor W_T = {0};
    const size_t axes[] = {1, 0};
    int status = -1;

    if (tensor_alloc_2d(&x_T, x->shape[1], x->shape[0]) != 0 || tensor_alloc_2d(&W_T, linear->W.shape[1], linear->W.shape[0]) != 0) goto cleanup;
    if (tensor_transpose(x, axes, &x_T) != 0 || tensor_transpose(&linear->W, axes, &W_T) != 0) goto cleanup;

    if (tensor_matmul(&x_T, grad, &linear->dW) != 0 || tensor_matmul(grad, &W_T, output) != 0) goto cleanup;

    tensor_fill(&linear->db, 0.0f);
    for (size_t i=0; i<grad->shape[0]; i++) {
        for (size_t j=0; j<grad->shape[1]; j++) {
            linear->db.data[j] += grad->data[i * grad->shape[1] + j];
        }
    }
    status = 0;

    cleanup:
        tensor_free(&x_T);
        tensor_free(&W_T);
        return status;
}
