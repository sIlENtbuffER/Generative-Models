#include "trainer/trainer.cuh"

#include "trainer/vae_trainer.cuh"
#include "trainer/gan_trainer.cuh"
#include "models/gan.cuh"
#include "optimizers/optimizer.cuh"

#include <stdio.h>

static int build_optimizers(const Model *model, const Config *config, Optimizer *optimizers, size_t *num_optimizers) {
    if (model->type == MODEL_VAE) {
        if (optimizer_build(&optimizers[0], config->optimizer_name, model->parameters, model->num_parameters, config->learning_rate, config->beta1, config->beta2, config->eps) != 0) return -1;
        *num_optimizers = 1;
        return 0;
    }

    if (model->type == MODEL_GAN) {
        GAN *gan = (GAN*)model->implementation;
        size_t num_generator = generator_num_parameters(&gan->generator);
        size_t num_discriminator = discriminator_num_parameters(&gan->discriminator);

        if (optimizer_build(&optimizers[0], config->optimizer_name, model->parameters, num_generator, config->learning_rate, config->beta1, config->beta2, config->eps) != 0) return -1;
        if (optimizer_build(&optimizers[1], config->optimizer_name, &model->parameters[num_generator], num_discriminator, config->learning_rate, config->beta1, config->beta2, config->eps) != 0) return -1;

        snprintf(optimizers[0].label, sizeof optimizers[0].label, "optimizer.generator");
        snprintf(optimizers[1].label, sizeof optimizers[1].label, "optimizer.discriminator");
        *num_optimizers = 2;
        return 0;
    }

    fprintf(stderr, "No trainer for model type %d\n", (int)model->type);
    return -1;
}

int trainer_train(Model *model, const Config *config, uint64_t *seed, Data *data) {
    Data samples = {};
    Optimizer optimizers[2] = {};
    Optimizer *optimizer_list[2] = {&optimizers[0], &optimizers[1]};
    size_t num_optimizers = 0;
    size_t completed_epoch = 0;
    char checkpoint_path[1024];
    int status = -1;

    if (build_optimizers(model, config, optimizers, &num_optimizers) != 0) goto cleanup;
    if (data_alloc(&samples, config->num_samples, data->channels, data->rows, data->cols) != 0) goto cleanup;

    if (config->load_checkpoint) {
        if (checkpoint_load(config->checkpoint_path, model, optimizer_list, num_optimizers, &completed_epoch) != 0) goto cleanup;
        printf("Loaded checkpoint: %s, epoch: %zu\n", config->checkpoint_path, completed_epoch);
    }

    for (size_t epoch=completed_epoch+1; epoch <= config->epochs; epoch++) {
        if (model->type == MODEL_VAE) {
            if (vae_train_epoch((VAE*)model->implementation, &optimizers[0], seed, config->batch_size, epoch, data, &samples, config->sample_dir) != 0) goto cleanup;
        } else {
            if (gan_train_epoch((GAN*)model->implementation, &optimizers[0], &optimizers[1], seed, config->batch_size, epoch, data, &samples, config->sample_dir) != 0) goto cleanup;
        }

        snprintf(checkpoint_path, sizeof checkpoint_path, "%s/epoch_%zu.safetensors", config->checkpoint_dir, epoch);
        if (checkpoint_save(checkpoint_path, model, optimizer_list, num_optimizers, epoch) != 0) goto cleanup;
    }

    status = 0;

cleanup:
    data_free(&samples);
    for (size_t i=0; i<num_optimizers; i++) {
        optimizer_free(&optimizers[i]);
    }
    return status;
}
