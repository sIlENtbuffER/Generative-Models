#include "modules/activations.cuh"
#include "core/cuda.cuh"

#include <math.h>

__global__ static void relu_cal_kernel(const float *x, float *output, float *value, size_t numel) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;

    output[i] = x[i] > 0.0f? value[i] : 0.0f;
}

__global__ static void sigmoid_forward_kernel(const float *x, float *output, size_t numel) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;

    output[i] = x[i] >= 0.0f? (1.0f / (1.0f + expf(-x[i]))) : (expf(x[i]) / (1.0f + expf(x[i])));
}

__global__ static void sigmoid_backward_kernel(const float *fw_output, float *grad, float *output, size_t numel) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;

    output[i] = grad[i] * fw_output[i] * (1.0f - fw_output[i]);
}

__global__ static void leaky_relu_forward_kernel(const float *x, float slope, float *output, size_t numel) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;

    output[i] = x[i] > 0.0f? x[i] : x[i] * slope;
}

__global__ static void leaky_relu_backward_kernel(const float *fw_input, const float *grad, float slope, float *output, size_t numel) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;

    output[i] = fw_input[i] > 0.0f? grad[i] : grad[i] * slope;
}

__global__ static void tanh_forward_kernel(const float *x, float *output, size_t numel) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;

    output[i] = tanhf(x[i]);
}

__global__ static void tanh_backward_kernel(const float *fw_output, const float *grad, float *output, size_t numel) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;

    output[i] = (1 - fw_output[i] * fw_output[i]) * grad[i];
}

int relu_forward(const DeviceTensor *x, DeviceTensor *output) {
    if (!tensor_is_same_shape(x, output)) return -1;
    
    relu_cal_kernel<<<cuda_blocks(output->numel), THREADS_PER_BLOCK>>>(x->data, output->data, x->data, output->numel);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int relu_backward(const DeviceTensor *fw_input, const DeviceTensor *grad, DeviceTensor *output) {
    if (!tensor_is_same_shape(grad, fw_input) || !tensor_is_same_shape(grad, output)) return -1;

    relu_cal_kernel<<<cuda_blocks(output->numel), THREADS_PER_BLOCK>>>(fw_input->data, output->data, grad->data, output->numel);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int sigmoid_forward(const DeviceTensor *x, DeviceTensor *output) {
    if (!tensor_is_same_shape(x, output)) return -1;

    sigmoid_forward_kernel<<<cuda_blocks(output->numel), THREADS_PER_BLOCK>>>(x->data, output->data, output->numel);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int sigmoid_backward(const DeviceTensor *fw_output, const DeviceTensor *grad, DeviceTensor *output) {
    if (!tensor_is_same_shape(grad, fw_output) || !tensor_is_same_shape(grad, output)) return -1;

    sigmoid_backward_kernel<<<cuda_blocks(output->numel), THREADS_PER_BLOCK>>>(fw_output->data, grad->data, output->data, output->numel);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int leaky_relu_forward(const DeviceTensor *x, float slope, DeviceTensor *output) {
    if (!tensor_is_same_shape(x, output)) return -1;

    leaky_relu_forward_kernel<<<cuda_blocks(output->numel), THREADS_PER_BLOCK>>>(x->data, slope, output->data, output->numel);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int leaky_relu_backward(const DeviceTensor *fw_input, const DeviceTensor *grad, float slope, DeviceTensor *output) {
    if (!tensor_is_same_shape(grad, fw_input) || !tensor_is_same_shape(grad, output)) return -1;

    leaky_relu_backward_kernel<<<cuda_blocks(output->numel), THREADS_PER_BLOCK>>>(fw_input->data, grad->data, slope, output->data, output->numel);
    
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int tanh_forward(const DeviceTensor *x, DeviceTensor *output) {
    if (!tensor_is_same_shape(x, output)) return -1;

    tanh_forward_kernel<<<cuda_blocks(output->numel), THREADS_PER_BLOCK>>>(x->data, output->data, output->numel);
    
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int tanh_backward(const DeviceTensor *fw_output, const DeviceTensor *grad, DeviceTensor *output) {
    if (!tensor_is_same_shape(grad, fw_output) || !tensor_is_same_shape(grad, output)) return -1;

    tanh_backward_kernel<<<cuda_blocks(output->numel), THREADS_PER_BLOCK>>>(fw_output->data, grad->data, output->data, output->numel);
    
    CUDA_CHECK(cudaGetLastError());
    return 0;
}
