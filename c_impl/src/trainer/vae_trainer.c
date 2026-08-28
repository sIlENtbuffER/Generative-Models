#include "trainer/vae_trainer.h"

#include <stdio.h>

int vae_train_batch(VAE *vae, Optimizer *optimizer, RNG *rng, VAEWorkspace *vae_for_ws, VAEWorkspace *vae_bac_ws, VAELoss *loss) {
    if (vae_forward(vae, rng, vae_for_ws) != 0) return -1;
    if (vae_backward(vae, vae_for_ws, vae_bac_ws, loss) != 0) return -1;
    if (optimizer_step(optimizer) != 0) return -1;
    return 0;
}

int vae_train_epoch(VAE *vae, Optimizer *optimizer, RNG *rng, size_t batch_size, size_t epoch, Data *data, Data *samples, const char *sample_dir) {
    int status = -1;
    char sample_path[1024];
    VAELoss loss = {0};
    size_t seen = 0;
    VAEWorkspace vae_for_ws = {0};
    VAEWorkspace vae_bac_ws = {0};
    if (vae_workspace_alloc(vae, batch_size, &vae_for_ws) != 0 || vae_workspace_alloc(vae, batch_size, &vae_bac_ws) != 0) goto cleanup;

    for (size_t start=0; start+batch_size<=data->count; start+=batch_size) {
        VAELoss batch_loss;

        if (data_batch(data, start, &vae_for_ws.input) != 0) goto cleanup;
        if (vae_train_batch(vae, optimizer, rng, &vae_for_ws, &vae_bac_ws, &batch_loss) != 0) goto cleanup;

        loss.loss += batch_loss.loss * batch_size;
        loss.recon_loss += batch_loss.recon_loss * batch_size;
        loss.kl_loss += batch_loss.kl_loss * batch_size;
        seen += batch_size;
    }

    loss.loss /= seen;
    loss.recon_loss /= seen;
    loss.kl_loss /= seen;
    
    printf("Epoch %zu | loss: %.4f | recon: %.4f | kl: %.4f\n", epoch, loss.loss, loss.recon_loss, loss.kl_loss);
    
    if (vae_sample(vae, rng, samples) != 0) goto cleanup;
    snprintf(sample_path, sizeof sample_path, "%s/epoch_%zu.png", sample_dir, epoch);
    
    if (data_write_png_grid(samples, sample_path) != 0) goto cleanup;

    status = 0;
    
cleanup:
    vae_workspace_free(&vae_for_ws);
    vae_workspace_free(&vae_bac_ws);
    return status;
}

int vae_sample(const VAE *vae, RNG *rng, Data *output){
    int status = -1;

    VAEWorkspace vaews = {0};
    if (vae_workspace_alloc(vae, output->count, &vaews) != 0) goto cleanup;

    for (size_t i=0; i<vaews.z.numel; i++) {
        vaews.z.data[i] = rng_normal(rng);
    }
    
    if (linear_forward(&vae->fc2, &vaews.z, &vaews.dec_pre) != 0 || relu_forward(&vaews.dec_pre, &vaews.dec_hidden) != 0) goto cleanup;
    if (linear_forward(&vae->fc3, &vaews.dec_hidden, &vaews.logits) != 0 || sigmoid_forward(&vaews.logits, &vaews.output) != 0) goto cleanup;
    
    for (size_t i=0; i<output->count * output->rows * output->cols; i++) {
        output->pixels[i] = (uint8_t)(vaews.output.data[i] * 255.0f);
    }

    status = 0;

cleanup:
    vae_workspace_free(&vaews);
    return status;
}
