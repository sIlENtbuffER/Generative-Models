#include "trainer/vae_trainer.cuh"
#include "core/cuda.cuh"

#include <stdio.h>

__global__ static void vae_sample_kernel(float *z, uint64_t seed, size_t numel) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;

    RNG rng;
    rng_seed(&rng, seed, i);
    z[i] = rng_normal(&rng);
}

__global__ static void vae_sample_pixel_kernel(uint8_t *out, const float *data, size_t numel) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;

    out[i] = (uint8_t)(data[i] * 255.0f);
}

int vae_train_batch(VAE *vae, Optimizer *optimizer, uint64_t *seed, VAEWorkspace *vae_for_ws, VAEWorkspace *vae_bac_ws, VAELoss *loss) {
    if (vae_forward(vae, seed, vae_for_ws) != 0) return -1;
    if (vae_backward(vae, vae_for_ws, vae_bac_ws, loss) != 0) return -1;
    if (optimizer_step(optimizer) != 0) return -1;
    return 0;
}

int vae_train_epoch(VAE *vae, Optimizer *optimizer, uint64_t *seed, size_t batch_size, size_t epoch, Data *data, Data *samples, const char *sample_dir) {
    int status = -1;
    char sample_path[1024];
    VAELoss loss = {};
    size_t seen = 0;
    VAEWorkspace vae_for_ws = {};
    VAEWorkspace vae_bac_ws = {};
    Tensor data_host = {};
    if (vae_workspace_alloc(vae, batch_size, &vae_for_ws) != 0 || vae_workspace_alloc(vae, batch_size, &vae_bac_ws) != 0) goto cleanup;
    if (tensor_alloc(&data_host, vae_for_ws.input.ndim, vae_for_ws.input.shape) != 0) goto cleanup;

    for (size_t start=0; start+batch_size<=data->count; start+=batch_size) {
        VAELoss batch_loss;

        if (data_batch(data, start, &data_host) != 0) goto cleanup;
        if (tensor_host_to_device(data_host.data, &vae_for_ws.input) != 0) goto cleanup;
        if (vae_train_batch(vae, optimizer, seed, &vae_for_ws, &vae_bac_ws, &batch_loss) != 0) goto cleanup;

        loss.loss += batch_loss.loss * batch_size;
        loss.recon_loss += batch_loss.recon_loss * batch_size;
        loss.kl_loss += batch_loss.kl_loss * batch_size;
        seen += batch_size;
    }

    loss.loss /= seen;
    loss.recon_loss /= seen;
    loss.kl_loss /= seen;
    
    printf("Epoch %zu | loss: %.4f | recon: %.4f | kl: %.4f\n", epoch, loss.loss, loss.recon_loss, loss.kl_loss);
    
    if (vae_sample(vae, seed, samples) != 0) goto cleanup;
    snprintf(sample_path, sizeof sample_path, "%s/epoch_%zu.png", sample_dir, epoch);
    
    if (data_write_png_grid(samples, sample_path) != 0) goto cleanup;

    status = 0;
    
cleanup:
    vae_workspace_free(&vae_for_ws);
    vae_workspace_free(&vae_bac_ws);
    tensor_free(&data_host);
    return status;
}

int vae_sample(const VAE *vae, uint64_t *seed, Data *output){
    int status = -1;
    uint8_t *device_pixels = NULL;
    VAEWorkspace vaews = {};
    if (vae_workspace_alloc(vae, output->count, &vaews) != 0) goto cleanup;

    vae_sample_kernel<<<cuda_blocks(vaews.z.numel), THREADS_PER_BLOCK>>>(vaews.z.data, *seed, vaews.z.numel);
    (*seed)++;

    if (linear_forward(&vae->fc2, &vaews.z, &vaews.dec_pre) != 0 || relu_forward(&vaews.dec_pre, &vaews.dec_hidden) != 0) goto cleanup;
    if (linear_forward(&vae->fc3, &vaews.dec_hidden, &vaews.logits) != 0 || sigmoid_forward(&vaews.logits, &vaews.output) != 0) goto cleanup;

    if (cudaMalloc(&device_pixels, vaews.output.numel * sizeof *device_pixels) != cudaSuccess) goto cleanup;
    vae_sample_pixel_kernel<<<cuda_blocks(vaews.output.numel), THREADS_PER_BLOCK>>>(device_pixels, vaews.output.data, vaews.output.numel);
    if (cudaGetLastError() != cudaSuccess) goto cleanup;
    if (cudaMemcpy(output->pixels, device_pixels, vaews.output.numel * sizeof *device_pixels, cudaMemcpyDeviceToHost) != cudaSuccess) goto cleanup;

    status = 0;

cleanup:
    vae_workspace_free(&vaews);
    cudaFree(device_pixels);
    return status;
}
