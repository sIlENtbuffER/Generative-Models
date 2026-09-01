#include "trainer/trainer.cuh"

#include "trainer/vae_trainer.cuh"
#include "optimizers/optimizer.cuh"

#include <stdio.h>

int trainer_train(Model *model, Optimizer *optimizer, uint64_t *seed, size_t batch_size, size_t start_epoch, size_t epochs, Data *data, size_t num_samples, const char *sample_dir, const char *checkpoint_dir) {
    Data samples = {};
    char checkpoint_path[1024];
    int status = -1;

    if (data_alloc(&samples, num_samples, data->channels, data->rows, data->cols) != 0) return -1;

    for (size_t epoch=start_epoch; epoch <= epochs; epoch++) {
        if (model->type == MODEL_VAE) {
            if (vae_train_epoch((VAE*)model->implementation, optimizer, seed, batch_size, epoch, data, &samples, sample_dir) != 0) goto cleanup;
        } else {
            goto cleanup;
        }

        snprintf(checkpoint_path, sizeof checkpoint_path, "%s/epoch_%zu.safetensors", checkpoint_dir, epoch);
        if (checkpoint_save(checkpoint_path, model, optimizer, epoch) != 0) goto cleanup;
    }

    status = 0;

cleanup:
    data_free(&samples);
    return status;
}
