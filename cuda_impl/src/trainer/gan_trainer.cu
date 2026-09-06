#include "trainer/gan_trainer.cuh"
#include "core/cuda.cuh"
#include "core/rng.cuh"
#include "core/tensor.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

__device__ static float sigmoid(float z) {
    return z >= 0.0f ? 1.0f / (1.0f + expf(-z)) : expf(z) / (1.0f + expf(z));
}

__global__ void gan_z_init_kernel(float *output, size_t numel, uint64_t seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    RNG rng;
    rng_seed(&rng, seed, idx);
    output[idx] = rng_normal(&rng);
    
}

__global__ void gan_y_init_kernel(float *output, size_t batch_size,size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    output[idx] = idx < batch_size ? 0.0f : 1.0f;
}

__global__ void gan_d_loss_kernel(const float *z, const float *y, float *loss, float *dx, size_t batch_size, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    float value = 0.0f;
    if (idx < numel) {
        value = fmaxf(z[idx], 0.0f) - y[idx] * z[idx] + log1pf(expf(-fabsf(z[idx])));
        dx[idx] = (sigmoid(z[idx]) - y[idx]) / (float)(batch_size);
    }

    block_reduce_add(loss, value);
}

__global__ void gan_g_loss_kernel(const float *z, float *loss, float *dx, size_t batch_size, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    float value = 0.0f;
    if (idx < numel) {
        value = fmaxf(-z[idx], 0.0f) + log1pf(expf(-fabsf(z[idx])));
        dx[idx] = (sigmoid(z[idx]) - 1.0f) / (float)batch_size;
    }

    block_reduce_add(loss, value);
}

__global__ void gan_normalize_kernel(float *data, size_t numel, float mean, float stddev) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    data[idx] = (data[idx] - mean) / stddev;
}

__global__ static void gan_sample_pixel_kernel(uint8_t *out, const float *data, size_t numel) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;

    out[i] = (uint8_t)((data[i] + 1.0f) * 127.5f);
}

int gan_train_batch(GAN *gan, Optimizer *g_optimizer, Optimizer *d_optimizer, uint64_t *seed, const DeviceTensor *x_real, GANWorkspace *ganws, GANLoss *loss) {
    const size_t batch_size = x_real->shape[0];
    const size_t g_last = gan->generator.num_layers - 1;
    const size_t d_last = gan->discriminator.num_layers - 1;

    if (gan->d_loss_scratch.data == NULL && tensor_alloc_1d(&gan->d_loss_scratch, 1) != 0) return -1;
    if (gan->g_loss_scratch.data == NULL && tensor_alloc_1d(&gan->g_loss_scratch, 1) != 0) return -1;
    if (tensor_fill(&gan->d_loss_scratch, 0.0f) != 0 || tensor_fill(&gan->g_loss_scratch, 0.0f) != 0) return -1;

    loss->d_loss = 0.0f;
    loss->g_loss = 0.0f;

    gan_z_init_kernel<<<cuda_blocks(ganws->z.numel), THREADS_PER_BLOCK>>>(ganws->z.data, ganws->z.numel, *seed);
    CUDA_CHECK(cudaGetLastError());
    (*seed)++;

    if (generator_forward(&gan->generator, &ganws->z, &ganws->g_for) != 0) return -1;

    const DeviceTensor *x_fake = &ganws->g_for.hidden[g_last];

    // D step
    cudaMemcpy(ganws->x_all.data, x_fake->data, x_fake->numel * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(&ganws->x_all.data[x_fake->numel], x_real->data, x_real->numel * sizeof(float), cudaMemcpyDeviceToDevice);

    gan_y_init_kernel<<<cuda_blocks(ganws->y.numel), THREADS_PER_BLOCK>>>(ganws->y.data, batch_size, ganws->y.numel);
    CUDA_CHECK(cudaGetLastError());

    if (discriminator_forward(&gan->discriminator, &ganws->x_all, &ganws->d_all_for) != 0) return -1;

    const DeviceTensor *d_logits = &ganws->d_all_for.pre[d_last];
    gan_d_loss_kernel<<<cuda_blocks(d_logits->numel), THREADS_PER_BLOCK>>>(d_logits->data, ganws->y.data, gan->d_loss_scratch.data, ganws->d_all_bac.pre[d_last].data, 2 * batch_size, d_logits->numel);
    CUDA_CHECK(cudaGetLastError());
    cudaMemcpy(&loss->d_loss, gan->d_loss_scratch.data, sizeof(float), cudaMemcpyDeviceToHost);

    if (discriminator_backward(&gan->discriminator, &ganws->d_all_for, &ganws->d_all_bac) != 0) return -1;
    if (optimizer_step(d_optimizer) != 0) return -1;

    // G step
    if (discriminator_forward(&gan->discriminator, x_fake, &ganws->d_fake_for) != 0) return -1;

    const DeviceTensor *g_logits = &ganws->d_fake_for.pre[d_last];
    gan_g_loss_kernel<<<cuda_blocks(g_logits->numel), THREADS_PER_BLOCK>>>(g_logits->data, gan->g_loss_scratch.data, ganws->d_fake_bac.pre[d_last].data, batch_size, g_logits->numel);
    CUDA_CHECK(cudaGetLastError());
    cudaMemcpy(&loss->g_loss, gan->g_loss_scratch.data, sizeof(float), cudaMemcpyDeviceToHost);

    if (discriminator_backward(&gan->discriminator, &ganws->d_fake_for, &ganws->d_fake_bac) != 0) return -1;
    cudaMemcpy(ganws->g_bac.hidden[g_last].data, ganws->d_fake_bac.input.data, ganws->d_fake_bac.input.numel * sizeof(float), cudaMemcpyDeviceToDevice);
    if (generator_backward(&gan->generator, &ganws->g_for, &ganws->g_bac) != 0) return -1;
    if (optimizer_step(g_optimizer) != 0) return -1;

    return 0;
}

int gan_train_epoch(GAN *gan, Optimizer *g_optimizer, Optimizer *d_optimizer, uint64_t *seed, size_t batch_size, size_t epoch, Data *data, Data *samples, const char *sample_dir) {
    int status = -1;
    char sample_path[1024];
    GANLoss loss = {};
    size_t seen = 0;
    GANWorkspace ganws = {};
    Tensor x_real_host = {};
    DeviceTensor x_real = {};

    if (gan_workspace_alloc(&ganws, gan, batch_size) != 0) goto cleanup;
    if (tensor_alloc_2d(&x_real, batch_size, gan->discriminator.input_dim) != 0) goto cleanup;
    if (tensor_alloc_2d(&x_real_host, batch_size, gan->discriminator.input_dim) != 0) goto cleanup;

    for (size_t start=0; start+batch_size<=data->count; start+=batch_size) {
        GANLoss batch_loss;

        if (data_batch(data, start, &x_real_host) != 0) goto cleanup;
        cudaMemcpy(x_real.data, x_real_host.data, x_real.numel * sizeof(float), cudaMemcpyHostToDevice);
        gan_normalize_kernel<<<cuda_blocks(x_real.numel), THREADS_PER_BLOCK>>>(x_real.data, x_real.numel, 0.5f, 0.5f);
        CUDA_CHECK(cudaGetLastError());

        if (gan_train_batch(gan, g_optimizer, d_optimizer, seed, &x_real, &ganws, &batch_loss) != 0) goto cleanup;
        (*seed)++;

        loss.d_loss += batch_loss.d_loss * batch_size;
        loss.g_loss += batch_loss.g_loss * batch_size;
        seen += batch_size;
    }

    loss.d_loss /= seen;
    loss.g_loss /= seen;

    printf("Epoch %zu | d_loss: %.4f | g_loss: %.4f\n", epoch, loss.d_loss, loss.g_loss);

    if (gan_sample(&gan->generator, seed, samples) != 0) goto cleanup;
    snprintf(sample_path, sizeof sample_path, "%s/epoch_%zu.png", sample_dir, epoch);

    if (data_write_png_grid(samples, sample_path) != 0) goto cleanup;

    status = 0;

cleanup:
    gan_workspace_free(&ganws);
    tensor_free(&x_real);
    return status;
}

int gan_sample(const Generator *generator, uint64_t *seed, Data *output) {
    int status = -1;
    StackWorkspace ws = {};
    DeviceTensor z = {};
    uint8_t *device_pixels = NULL;

    if (stack_workspace_alloc(&ws, generator->layers, generator->num_layers, output->count) != 0) goto cleanup;
    if (tensor_alloc_2d(&z, output->count, generator->latent_dim) != 0) goto cleanup;

    gan_z_init_kernel<<<cuda_blocks(z.numel), THREADS_PER_BLOCK>>>(z.data, z.numel, *seed);
    CUDA_CHECK(cudaGetLastError());
    (*seed)++;

    if (generator_forward(generator, &z, &ws) != 0) goto cleanup;

    if (cudaMalloc(&device_pixels, ws.hidden[generator->num_layers - 1].numel * sizeof *device_pixels) != cudaSuccess) goto cleanup;
    gan_sample_pixel_kernel<<<cuda_blocks(ws.hidden[generator->num_layers - 1].numel), THREADS_PER_BLOCK>>>(device_pixels, ws.hidden[generator->num_layers - 1].data, ws.hidden[generator->num_layers - 1].numel);
    CUDA_CHECK(cudaGetLastError());
    if (cudaMemcpy(output->pixels, device_pixels, ws.hidden[generator->num_layers - 1].numel * sizeof(uint8_t), cudaMemcpyDeviceToHost) != cudaSuccess) goto cleanup;

    status = 0;

cleanup:
    stack_workspace_free(&ws);
    tensor_free(&z);
    cudaFree(device_pixels);
    return status;
}
