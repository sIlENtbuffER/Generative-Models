#include "models/vae.h"

static const char *names[VAE_PARAMETERS] = 
{
    "fc1_W", "fc1_b",
    "fc_mu_W", "fc_mu_b",
    "fc_logvar_W", "fc_logvar_b",
    "fc2_W", "fc2_b",
    "fc3_W", "fc3_b"
};

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

int vae_init(VAE *vae, RNG *rng) {
    if (linear_he_init(&vae->fc1, rng) != 0 || linear_he_init(&vae->fc_mu, rng) != 0 || linear_he_init(&vae->fc_logvar, rng) != 0 || linear_he_init(&vae->fc2, rng) != 0 || linear_he_init(&vae->fc3, rng) != 0) return -1;
    return 0;
}

int vae_forward(const VAE *vae, RNG *rng, VAEWorkspace *vaews) {
    if (linear_forward(&vae->fc1, &vaews->input, &vaews->enc_pre) != 0 || relu_forward(&vaews->enc_pre, &vaews->enc_hidden) != 0) return -1;
    if (linear_forward(&vae->fc_mu, &vaews->enc_hidden, &vaews->mu) != 0 || linear_forward(&vae->fc_logvar, &vaews->enc_hidden, &vaews->logvar) != 0) return -1;
    for (size_t i=0; i<vaews->z.numel; i++) {
        vaews->eps.data[i] = rng_normal(rng);
        vaews->z.data[i] = vaews->mu.data[i] + expf(0.5f * vaews->logvar.data[i]) * vaews->eps.data[i];
    }

    if (linear_forward(&vae->fc2, &vaews->z, &vaews->dec_pre) != 0 || relu_forward(&vaews->dec_pre, &vaews->dec_hidden) != 0) return -1;
    if (linear_forward(&vae->fc3, &vaews->dec_hidden, &vaews->logits) != 0 || sigmoid_forward(&vaews->logits, &vaews->output) != 0) return -1;

    return 0;
}

int vae_backward(VAE *vae, const VAEWorkspace *vae_for_ws, VAEWorkspace *vae_bac_ws, VAELoss *loss) {
    int status = -1;
    loss->recon_loss = 0.0f;
    loss->kl_loss = 0.0f;
    Tensor logvar_bw = {0};
    if (tensor_alloc_2d(&logvar_bw, vae_for_ws->enc_hidden.shape[0], vae_for_ws->enc_hidden.shape[1]) != 0) return -1;

    // Safe BCE with logits
    for (size_t i=0; i<vae_for_ws->output.numel; i++) {
        float x = vae_for_ws->input.data[i];
        float z = vae_for_ws->logits.data[i];
        float p = vae_for_ws->output.data[i];

        // Safe L = log(1 + exp(z)) - xz
        loss->recon_loss += fmaxf(z, 0.0f) - x * z + log1pf(expf(-fabsf(z)));
        vae_bac_ws->logits.data[i] = (p - x) / vae_for_ws->output.shape[0];
    }

    for (size_t i=0; i<vae_for_ws->mu.numel; i++) {
        loss->kl_loss += vae_for_ws->mu.data[i] * vae_for_ws->mu.data[i] + expf(vae_for_ws->logvar.data[i]) - 1.0f - vae_for_ws->logvar.data[i];
    }
    loss->recon_loss /= vae_for_ws->output.shape[0];
    loss->kl_loss = 0.5 * loss->kl_loss / vae_for_ws->output.shape[0];
    loss->loss = loss->recon_loss + loss->kl_loss;

    // Decoder
    if (linear_backward(&vae->fc3, &vae_for_ws->dec_hidden, &vae_bac_ws->logits, &vae_bac_ws->dec_hidden) != 0) goto cleanup;
    if (relu_backward(&vae_for_ws->dec_pre, &vae_bac_ws->dec_hidden, &vae_bac_ws->dec_pre) != 0) goto cleanup;
    if (linear_backward(&vae->fc2, &vae_for_ws->z, &vae_bac_ws->dec_pre, &vae_bac_ws->z) != 0) goto cleanup;

    // Reparameterization
    for (size_t i=0; i<vae_for_ws->z.numel; i++) {
        vae_bac_ws->mu.data[i] = vae_bac_ws->z.data[i] + vae_for_ws->mu.data[i] / vae_for_ws->z.shape[0];
        vae_bac_ws->logvar.data[i] = vae_bac_ws->z.data[i] * vae_for_ws->eps.data[i] * 0.5f * expf(0.5f * vae_for_ws->logvar.data[i]) + 0.5f * (expf(vae_for_ws->logvar.data[i]) - 1.0f) / vae_for_ws->z.shape[0];
    }

    // Encoder
    if (linear_backward(&vae->fc_mu, &vae_for_ws->enc_hidden, &vae_bac_ws->mu, &vae_bac_ws->enc_hidden) != 0 || linear_backward(&vae->fc_logvar, &vae_for_ws->enc_hidden, &vae_bac_ws->logvar, &logvar_bw) != 0) goto cleanup;
    for (size_t i=0; i<vae_bac_ws->enc_hidden.numel; i++) {
        vae_bac_ws->enc_hidden.data[i] += logvar_bw.data[i];
    }
    if (relu_backward(&vae_for_ws->enc_pre, &vae_bac_ws->enc_hidden, &vae_bac_ws->enc_pre) != 0) goto cleanup;
    if (linear_backward(&vae->fc1, &vae_for_ws->input, &vae_bac_ws->enc_pre, &vae_bac_ws->input) != 0) goto cleanup;

    status = 0;

cleanup:
    tensor_free(&logvar_bw);
    return status;
}

void vae_parameters(VAE *vae, Parameter parameters[VAE_PARAMETERS]) {
    Linear *layers[] = {&vae->fc1, &vae->fc_mu, &vae->fc_logvar, &vae->fc2, &vae->fc3};

    for (size_t i=0; i<5; i++) {
        parameters[2 * i] = (Parameter) {.name = names[2 * i], .value = &layers[i]->W, .grad = &layers[i]->dW};
        parameters[2 * i + 1] = (Parameter) {.name = names[2 * i + 1], .value = &layers[i]->b, .grad = &layers[i]->db};
    }
}
