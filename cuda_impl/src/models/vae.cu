#include "models/vae.cuh"
#include "core/cuda.cuh"

#include <math.h>

static const char *names[VAE_PARAMETERS] = 
{
    "fc1_W", "fc1_b",
    "fc_mu_W", "fc_mu_b",
    "fc_logvar_W", "fc_logvar_b",
    "fc2_W", "fc2_b",
    "fc3_W", "fc3_b"
};

__global__ static void vae_forward_kernel(float *z, float *eps, const float *mu, const float *logvar, const uint64_t seed, size_t numel) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;

    RNG rng;
    rng_seed(&rng, seed, i);

    eps[i] = rng_normal(&rng);
    z[i] = mu[i] + expf(0.5f * logvar[i]) * eps[i];
}

__global__ static void vae_backward_recon_loss_kernel(float *recon_loss, float *output, const float *x, const float *z, const float *p, size_t batch_size, size_t numel) {
    __shared__ float partial[THREADS_PER_BLOCK];
    
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    partial[threadIdx.x] = 0.0f;
    if (i < numel) {
        partial[threadIdx.x] = fmaxf(z[i], 0.0f) - x[i] * z[i] + log1pf(expf(-fabsf(z[i])));
        output[i] = (p[i] - x[i]) / batch_size;
    }
    __syncthreads();

    // Tree reduction
    for (size_t stride=blockDim.x/2; stride>0; stride/=2) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) atomicAdd(recon_loss, partial[0]);
}

__global__ static void vae_backward_kl_loss_kernel(float *kl_loss, const float *mu, const float *logvar, size_t numel) {
    __shared__ float partial[THREADS_PER_BLOCK];
    
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    partial[threadIdx.x] = 0.0f;
    if (i < numel) {
        partial[threadIdx.x] = mu[i] * mu[i] + expf(logvar[i]) - 1.0f - logvar[i];
    }
    __syncthreads();

    for (size_t stride=blockDim.x/2; stride>0; stride/=2) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) atomicAdd(kl_loss, partial[0]);
}

__global__ static void vae_backward_reparameter_kernel(float *dmu, float *dlogvar, const float *dz, const float *mu, const float *logvar, const float *eps, size_t batch_size, size_t numel) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;

    dmu[i] = dz[i] + mu[i] / batch_size;
    dlogvar[i] = dz[i] * eps[i] * 0.5f * expf(0.5f * logvar[i]) + 0.5f * (expf(logvar[i]) - 1.0f) / batch_size;
}

__global__ static void vae_backward_enc_hidden_kernel(float *denc_hidden, const float *dlogvar, size_t numel) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;

    atomicAdd(&denc_hidden[i], dlogvar[i]);
}

int vae_alloc(VAE *vae, size_t input_dim, size_t hidden_dim, size_t latent_dim) {
    *vae = (VAE){0};
    vae->input_dim = input_dim;
    vae->hidden_dim = hidden_dim;
    vae->latent_dim = latent_dim;
    if (linear_alloc(&vae->fc1, input_dim, hidden_dim) != 0 || linear_alloc(&vae->fc_mu, hidden_dim, latent_dim) != 0 || linear_alloc(&vae->fc_logvar, hidden_dim, latent_dim) != 0 || linear_alloc(&vae->fc2, latent_dim, hidden_dim) != 0 || linear_alloc(&vae->fc3, hidden_dim, input_dim) != 0) {
        vae_free(vae);
        return -1;
    }
    return 0;
}

int vae_free(VAE *vae) {
    linear_free(&vae->fc1);
    linear_free(&vae->fc_mu);
    linear_free(&vae->fc_logvar);
    linear_free(&vae->fc2);
    linear_free(&vae->fc3);
    
    vae->input_dim = 0;
    vae->hidden_dim = 0;
    vae->latent_dim = 0;
    
    return 0;
}

int vae_workspace_alloc(const VAE *vae, size_t batch_size, VAEWorkspace *vaews) {
    *vaews = (VAEWorkspace){0};
    if (tensor_alloc_2d(&vaews->input, batch_size, vae->input_dim) != 0) goto cleanup;
    if (tensor_alloc_2d(&vaews->enc_pre, batch_size, vae->hidden_dim) != 0 || tensor_alloc_2d(&vaews->enc_hidden, batch_size, vae->hidden_dim) != 0) goto cleanup;
    if (tensor_alloc_2d(&vaews->mu, batch_size, vae->latent_dim) != 0 || tensor_alloc_2d(&vaews->logvar, batch_size, vae->latent_dim) != 0 || tensor_alloc_2d(&vaews->eps, batch_size, vae->latent_dim) != 0 || tensor_alloc_2d(&vaews->z, batch_size, vae->latent_dim) != 0) goto cleanup;
    if (tensor_alloc_2d(&vaews->dec_pre, batch_size, vae->hidden_dim) != 0 || tensor_alloc_2d(&vaews->dec_hidden, batch_size, vae->hidden_dim) != 0) goto cleanup;
    if (tensor_alloc_2d(&vaews->logits, batch_size, vae->input_dim) != 0 || tensor_alloc_2d(&vaews->output, batch_size, vae->input_dim) != 0) goto cleanup;
    return 0;

cleanup:
    vae_workspace_free(vaews);
    return -1;
}

int vae_workspace_free(VAEWorkspace *vaews) {
    tensor_free(&vaews->input);
    tensor_free(&vaews->enc_pre);
    tensor_free(&vaews->enc_hidden);
    tensor_free(&vaews->mu);
    tensor_free(&vaews->logvar);
    tensor_free(&vaews->eps);
    tensor_free(&vaews->z);
    tensor_free(&vaews->dec_pre);
    tensor_free(&vaews->dec_hidden);
    tensor_free(&vaews->logits);
    tensor_free(&vaews->output);
    return 0;
}

int vae_init(VAE *vae, uint64_t *seed) {
    if (linear_he_init(&vae->fc1, seed) != 0 || linear_he_init(&vae->fc_mu, seed) != 0 || linear_he_init(&vae->fc_logvar, seed) != 0 || linear_he_init(&vae->fc2, seed) != 0 || linear_he_init(&vae->fc3, seed) != 0) return -1;
    return 0;
}

int vae_forward(const VAE *vae, uint64_t *seed, VAEWorkspace *vaews) {
    if (linear_forward(&vae->fc1, &vaews->input, &vaews->enc_pre) != 0 || relu_forward(&vaews->enc_pre, &vaews->enc_hidden) != 0) return -1;
    if (linear_forward(&vae->fc_mu, &vaews->enc_hidden, &vaews->mu) != 0 || linear_forward(&vae->fc_logvar, &vaews->enc_hidden, &vaews->logvar) != 0) return -1;
    vae_forward_kernel<<<cuda_blocks(vaews->z.numel), THREADS_PER_BLOCK>>>(vaews->z.data, vaews->eps.data, vaews->mu.data, vaews->logvar.data, *seed, vaews->z.numel);
    (*seed)++;
    CUDA_CHECK(cudaGetLastError());

    if (linear_forward(&vae->fc2, &vaews->z, &vaews->dec_pre) != 0 || relu_forward(&vaews->dec_pre, &vaews->dec_hidden) != 0) return -1;
    if (linear_forward(&vae->fc3, &vaews->dec_hidden, &vaews->logits) != 0 || sigmoid_forward(&vaews->logits, &vaews->output) != 0) return -1;

    return 0;
}

int vae_backward(VAE *vae, const VAEWorkspace *vae_for_ws, VAEWorkspace *vae_bac_ws, VAELoss *loss) {
    int status = -1;
    loss->recon_loss = 0.0f;
    loss->kl_loss = 0.0f;
    DeviceTensor logvar_bw = {0};
    DeviceTensor d_recon_loss = {0};
    DeviceTensor d_kl_loss = {0};

    if (tensor_alloc_2d(&logvar_bw, vae_for_ws->enc_hidden.shape[0], vae_for_ws->enc_hidden.shape[1]) != 0) return -1;
    if (tensor_alloc_1d(&d_recon_loss, 1) != 0 || tensor_alloc_1d(&d_kl_loss, 1) != 0) goto cleanup;
    
    vae_backward_recon_loss_kernel<<<cuda_blocks(vae_for_ws->output.numel), THREADS_PER_BLOCK>>>(d_recon_loss.data, vae_bac_ws->logits.data, vae_for_ws->input.data, vae_for_ws->logits.data, vae_for_ws->output.data, vae_for_ws->output.shape[0], vae_for_ws->output.numel);
    vae_backward_kl_loss_kernel<<<cuda_blocks(vae_for_ws->mu.numel), THREADS_PER_BLOCK>>>(d_kl_loss.data, vae_for_ws->mu.data, vae_for_ws->logvar.data, vae_for_ws->mu.numel);
    if (cudaGetLastError() != cudaSuccess) goto cleanup;

    if (tensor_device_to_host(&d_recon_loss, &loss->recon_loss) != 0) goto cleanup;
    if (tensor_device_to_host(&d_kl_loss, &loss->kl_loss) != 0) goto cleanup;

    loss->recon_loss /= vae_for_ws->output.shape[0];
    loss->kl_loss = 0.5 * loss->kl_loss / vae_for_ws->output.shape[0];
    loss->loss = loss->recon_loss + loss->kl_loss;

    // Decoder
    if (linear_backward(&vae->fc3, &vae_for_ws->dec_hidden, &vae_bac_ws->logits, &vae_bac_ws->dec_hidden) != 0) goto cleanup;
    if (relu_backward(&vae_for_ws->dec_pre, &vae_bac_ws->dec_hidden, &vae_bac_ws->dec_pre) != 0) goto cleanup;
    if (linear_backward(&vae->fc2, &vae_for_ws->z, &vae_bac_ws->dec_pre, &vae_bac_ws->z) != 0) goto cleanup;

    // Reparameterization
    vae_backward_reparameter_kernel<<<cuda_blocks(vae_for_ws->z.numel), THREADS_PER_BLOCK>>>(vae_bac_ws->mu.data, vae_bac_ws->logvar.data, vae_bac_ws->z.data, vae_for_ws->mu.data, vae_for_ws->logvar.data, vae_for_ws->eps.data, vae_for_ws->z.shape[0], vae_for_ws->z.numel);
    if (cudaGetLastError() != cudaSuccess) goto cleanup;

    // Encoder
    if (linear_backward(&vae->fc_mu, &vae_for_ws->enc_hidden, &vae_bac_ws->mu, &vae_bac_ws->enc_hidden) != 0 || linear_backward(&vae->fc_logvar, &vae_for_ws->enc_hidden, &vae_bac_ws->logvar, &logvar_bw) != 0) goto cleanup;
    vae_backward_enc_hidden_kernel<<<cuda_blocks(vae_bac_ws->enc_hidden.numel), THREADS_PER_BLOCK>>>(vae_bac_ws->enc_hidden.data, logvar_bw.data, vae_bac_ws->enc_hidden.numel);
    if (cudaGetLastError() != cudaSuccess) goto cleanup;
    if (relu_backward(&vae_for_ws->enc_pre, &vae_bac_ws->enc_hidden, &vae_bac_ws->enc_pre) != 0) goto cleanup;
    if (linear_backward(&vae->fc1, &vae_for_ws->input, &vae_bac_ws->enc_pre, &vae_bac_ws->input) != 0) goto cleanup;

    status = 0;

cleanup:
    tensor_free(&logvar_bw);
    tensor_free(&d_recon_loss);
    tensor_free(&d_kl_loss);
    return status;
}

void vae_parameters(VAE *vae, Parameter parameters[VAE_PARAMETERS]) {
    Linear *layers[] = {&vae->fc1, &vae->fc_mu, &vae->fc_logvar, &vae->fc2, &vae->fc3};

    for (size_t i=0; i<5; i++) {
        parameters[2 * i] = (Parameter) {.name = names[2 * i], .value = &layers[i]->W, .grad = &layers[i]->dW};
        parameters[2 * i + 1] = (Parameter) {.name = names[2 * i + 1], .value = &layers[i]->b, .grad = &layers[i]->db};
    }
}
