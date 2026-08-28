#include "core/tensor.h"

#include <stdint.h>
#include <stdlib.h>

int tensor_alloc(Tensor *tensor, size_t ndim, const size_t *shape) {
    if (ndim > TENSOR_MAX_DIMS) return -1;

    *tensor = (Tensor){0};
    size_t stride = 1;

    for (size_t i=ndim; i-->0;) {
        tensor->shape[i] = shape[i];
        tensor->strides[i] = stride;

        if (shape[i] != 0 && stride > SIZE_MAX / shape[i]) {
            *tensor = (Tensor){0};
            return -1;
        }

        stride *= shape[i];
    }
    tensor->ndim = ndim;
    tensor->numel = stride;
    tensor->data = calloc(tensor->numel, sizeof *tensor->data);
    if (tensor->data == NULL) return -1;
    return 0;
}

int tensor_alloc_1d(Tensor *tensor, size_t dim0) {
    const size_t shape[] = {dim0};
    return tensor_alloc(tensor, 1, shape);
}

int tensor_alloc_2d(Tensor *tensor, size_t dim0, size_t dim1) {
    const size_t shape[] = {dim0, dim1};
    return tensor_alloc(tensor, 2, shape);
}

void tensor_free(Tensor *tensor) {
    free(tensor->data);
    *tensor = (Tensor){0};
}

void tensor_fill(Tensor *tensor, float value) {
    for (size_t i=0; i<tensor->numel; i++) {
        tensor->data[i] = value;
    }
}

int tensor_is_same_shape(const Tensor *a, const Tensor *b) {
    if (a->ndim != b->ndim) return 0;
    for (size_t i=0; i<a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) return 0;
    }
    return 1;
}

int tensor_get_addr(const Tensor *tensor, const size_t *indices, size_t *address) {
    size_t res = 0;
    
    for (size_t i=0; i<tensor->ndim; i++) {
        if (indices[i] >= tensor->shape[i]) return -1;
        res += indices[i] * tensor->strides[i];
    }
    *address = res;
    return 0;
}

int tensor_get(const Tensor *tensor, const size_t *indices, float *value) {
    size_t addr;
    if (tensor_get_addr(tensor, indices, &addr) != 0) return -1;

    *value = tensor->data[addr];
    return 0;
}

int tensor_set(Tensor *tensor, const size_t *indices, float value) {
    size_t addr;
    if (tensor_get_addr(tensor, indices, &addr) != 0) return -1;

    tensor->data[addr] = value;
    return 0;
}

int tensor_reshape(Tensor *tensor, size_t ndim, const size_t *shape) {
    if (ndim > TENSOR_MAX_DIMS) return -1;
    
    size_t new_strides[TENSOR_MAX_DIMS] = {0};
    size_t new_numel = 1;

    for (size_t i = ndim; i-- > 0;) {
        new_strides[i] = new_numel;
        if (shape[i] != 0 && new_numel > SIZE_MAX / shape[i]) return -1;
        new_numel *= shape[i];
    }
    if (new_numel != tensor->numel) return -1;

    tensor->ndim = ndim;

    for (size_t i=0; i<ndim; i++) {
        tensor->shape[i] = shape[i];
        tensor->strides[i] = new_strides[i];
    }

    for (size_t i=ndim; i<TENSOR_MAX_DIMS; i++) {
        tensor->shape[i] = 0;
        tensor->strides[i] = 0;
    }

    return 0;
}

int tensor_matvec(const Tensor *matrix, const Tensor *vector, Tensor *output) {
    if (matrix->ndim != 2 || vector->ndim != 1 || output->ndim != 1 || vector->shape[0] != matrix->shape[1] || output->shape[0] != matrix->shape[0]) return -1;

    size_t rows = matrix->shape[0];
    size_t cols = matrix->shape[1];

    for (size_t i=0; i<rows; i++) {
        float sum = 0.0f;

        for (size_t j=0; j<cols; j++) {
            sum += matrix->data[i * cols + j] * vector->data[j];
        }

        output->data[i] = sum;
    }

    return 0;
}

int tensor_matmul_output_shape(const Tensor *a, const Tensor *b, size_t *output_ndim, size_t output_shape[TENSOR_MAX_DIMS]) {
    if (a->ndim < 2 || b->ndim < 2 || a->shape[a->ndim - 1] != b->shape[b->ndim - 2]) return -1;

    size_t a_m = a->shape[a->ndim - 2];
    size_t b_n = b->shape[b->ndim - 1];

    size_t a_batch_ndim = a->ndim - 2;
    size_t b_batch_ndim = b->ndim - 2;
    size_t batch_ndim = a_batch_ndim > b_batch_ndim ? a_batch_ndim : b_batch_ndim;
    size_t ndim = batch_ndim + 2;

    size_t a_padding = batch_ndim - a_batch_ndim;
    size_t b_padding = batch_ndim - b_batch_ndim;

    for (size_t i=0; i<batch_ndim; i++) {
        size_t a_dim = i < a_padding ? 1 : a->shape[i - a_padding];
        size_t b_dim = i < b_padding ? 1 : b->shape[i - b_padding];

        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) return -1;

        output_shape[i] = a_dim > b_dim ? a_dim : b_dim;
    }

    output_shape[batch_ndim] = a_m;
    output_shape[batch_ndim + 1] = b_n;
    *output_ndim = ndim;

    return 0;
}

int tensor_matmul(const Tensor *a, const Tensor *b, Tensor *output) {
    size_t batch_ndim = output->ndim - 2;
    size_t a_batch_ndim = a->ndim - 2;
    size_t b_batch_ndim = b->ndim - 2;
    size_t a_padding = batch_ndim - a_batch_ndim;
    size_t b_padding = batch_ndim - b_batch_ndim;
    size_t num_batch = 1;

    for (size_t i=0; i<batch_ndim; i++) {
        if (output->shape[i] != 0 && num_batch > SIZE_MAX / output->shape[i]) return -1;

        num_batch *= output->shape[i];
    }

    tensor_fill(output, 0.0f);

    size_t m = a->shape[a->ndim - 2];
    size_t k_size = a->shape[a->ndim - 1];
    size_t n = b->shape[b->ndim - 1];

    for (size_t batch=0; batch<num_batch; batch++) {
        size_t remaining_idx = batch;
        size_t a_batch_addr = 0;
        size_t b_batch_addr = 0;
        size_t output_batch_addr = 0;

        for (size_t axis=batch_ndim; axis-->0;) {
            size_t coordinate = remaining_idx % output->shape[axis];
            remaining_idx /= output->shape[axis];
            output_batch_addr += coordinate * output->strides[axis];

            // Broadcasting
            if (axis >= a_padding) {
                size_t a_axis = axis - a_padding;
                size_t a_coordinate = a->shape[a_axis] == 1 ? 0 : coordinate;
                a_batch_addr += a_coordinate * a->strides[a_axis];
            }
            if (axis >= b_padding) {
                size_t b_axis = axis - b_padding;
                size_t b_coordinate = b->shape[b_axis] == 1 ? 0 : coordinate;
                b_batch_addr += b_coordinate * b->strides[b_axis];
            }
        }

        size_t a_row_stride = a->strides[a->ndim - 2];
        size_t a_col_stride = a->strides[a->ndim - 1];
        size_t b_row_stride = b->strides[b->ndim - 2];
        size_t b_col_stride = b->strides[b->ndim - 1];
        size_t output_row_stride = output->strides[output->ndim - 2];
        size_t output_col_stride = output->strides[output->ndim - 1];
        
        for (size_t i=0; i<m; i++) {
            for (size_t k=0; k<k_size; k++) {
                float a_value = a->data[a_batch_addr + i * a_row_stride + k * a_col_stride];

                for (size_t j=0; j<n; j++) {
                    size_t output_addr = output_batch_addr + i * output_row_stride + j * output_col_stride;
                    output->data[output_addr] += a_value * b->data[b_batch_addr + k * b_row_stride + j * b_col_stride];
                }
            }
        }
    }

    return 0;
}

int tensor_transpose(const Tensor *input, const size_t *axes, Tensor *output) {
    if (input->numel != output->numel || input->ndim != output->ndim) return -1;
    int seen[TENSOR_MAX_DIMS] = {0};
    for (size_t i=0; i<input->ndim; i++) {
        if (axes[i] >= input->ndim) return -1;
        if (seen[axes[i]]) return -1;
        seen[axes[i]] = 1;
        if (output->shape[i] != input->shape[axes[i]]) return -1;
    }

    for (size_t output_addr=0; output_addr<output->numel; output_addr++) {
        size_t remaining_idx = output_addr;
        size_t input_addr = 0;

        for (size_t output_axis=0; output_axis<output->ndim; output_axis++) {
            size_t coordinate = remaining_idx / output->strides[output_axis];
            remaining_idx %= output->strides[output_axis];

            size_t input_axis = axes[output_axis];
            input_addr += coordinate * input->strides[input_axis];
        }

        output->data[output_addr] = input->data[input_addr];
    }

    return 0;
}
