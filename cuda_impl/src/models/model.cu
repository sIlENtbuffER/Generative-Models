#include "models/model.cuh"
#include "models/vae.cuh"
#include "models/gan.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int build_vae(Model *model, size_t input_dim, const cJSON *model_cfg, uint64_t *seed) {
    size_t hidden_dim;
    size_t latent_dim;
    if (config_get_size(model_cfg, "hidden_dim", &hidden_dim) != 0) return -1;
    if (config_get_size(model_cfg, "latent_dim", &latent_dim) != 0) return -1;

    VAE *vae = (VAE*)calloc(1, sizeof *vae);
    if (vae == NULL) return -1;
    if (vae_alloc(vae, input_dim, hidden_dim, latent_dim) != 0) {
        free(vae);
        return -1;
    }
    if (vae_init(vae, seed) != 0) {
        vae_free(vae);
        free(vae);
        return -1;
    }

    model->parameters = (Parameter*)malloc(VAE_PARAMETERS * sizeof *model->parameters);
    if (model->parameters == NULL) {
        vae_free(vae);
        free(vae);
        return -1;
    }
    vae_parameters(vae, model->parameters);

    model->type = MODEL_VAE;
    model->implementation = vae;
    model->num_parameters = VAE_PARAMETERS;
    return 0;
}

static int build_gan(Model *model, size_t input_dim, const cJSON *model_cfg, uint64_t *seed) {
    int status = -1;
    size_t latent_dim;
    size_t *g_hidden_dims = NULL;
    size_t *d_hidden_dims = NULL;
    size_t num_g_hidden = 0;
    size_t num_d_hidden = 0;
    size_t num_generator = 0;
    size_t num_discriminator = 0;
    GAN *gan = NULL;

    if (config_get_size(model_cfg, "latent_dim", &latent_dim) != 0) goto cleanup;
    if (config_get_size_array(model_cfg, "g_hidden_dims", &g_hidden_dims, &num_g_hidden) != 0) goto cleanup;
    if (config_get_size_array(model_cfg, "d_hidden_dims", &d_hidden_dims, &num_d_hidden) != 0) goto cleanup;

    gan = (GAN*)calloc(1, sizeof *gan);
    if (gan == NULL) goto cleanup;
    if (gan_alloc(gan, input_dim, latent_dim, g_hidden_dims, num_g_hidden, d_hidden_dims, num_d_hidden) != 0) goto cleanup;
    if (gan_init(gan, seed) != 0) goto cleanup;

    num_generator = generator_num_parameters(&gan->generator);
    num_discriminator = discriminator_num_parameters(&gan->discriminator);
    model->parameters = (Parameter*)malloc((num_generator + num_discriminator) * sizeof *model->parameters);
    if (model->parameters == NULL) goto cleanup;
    generator_parameters(&gan->generator, model->parameters);
    discriminator_parameters(&gan->discriminator, &model->parameters[num_generator]);

    model->type = MODEL_GAN;
    model->implementation = gan;
    model->num_parameters = num_generator + num_discriminator;
    status = 0;

cleanup:
    free(g_hidden_dims);
    free(d_hidden_dims);
    if (status != 0 && gan != NULL) {
        gan_free(gan);
        free(gan);
    }
    return status;
}

int model_build(Model *model, const char *name, size_t input_dim, const cJSON *model_cfg, uint64_t *seed) {
    *model = (Model){};

    if (strcmp(name, "vae") == 0) return build_vae(model, input_dim, model_cfg, seed);
    if (strcmp(name, "gan") == 0) return build_gan(model, input_dim, model_cfg, seed);

    fprintf(stderr, "Unknown model: %s\n", name);
    return -1;
}

void model_free(Model *model) {
    if (model->type == MODEL_VAE) {
        VAE *vae = (VAE*)model->implementation;
        vae_free(vae);
        free(vae);
    } else if (model->type == MODEL_GAN) {
        GAN *gan = (GAN*)model->implementation;
        gan_free(gan);
        free(gan);
    }

    free(model->parameters);
    *model = (Model){};
}

int model_save_checkpoint(const Model *model, Checkpoint *checkpoint) {
    char name[CHECKPOINT_TENSOR_NAME_SIZE];
    Tensor host = {};

    for (size_t i=0; i<model->num_parameters; i++) {
        Parameter *parameter = &model->parameters[i];
        snprintf(name, sizeof(name), "model.%s", parameter->name);

        if (tensor_alloc(&host, parameter->value->ndim, parameter->value->shape) != 0) goto fail;
        if (tensor_device_to_host(parameter->value, host.data) != 0) goto fail;
        if (checkpoint_take_tensor(checkpoint, name, &host) != 0) goto fail;
    }
    return 0;

fail:
    tensor_free(&host);
    return -1;
}

int model_load_checkpoint(Model *model, const Checkpoint *checkpoint) {
    char name[CHECKPOINT_TENSOR_NAME_SIZE];

    for (size_t i=0; i<model->num_parameters; i++) {
        Parameter *parameter = &model->parameters[i];
        snprintf(name, sizeof name, "model.%s", parameter->name);

        const Tensor *cpt = checkpoint_get_tensor(checkpoint, name);
        if (cpt == NULL || tensor_host_to_device(cpt->data, parameter->value) != 0) return -1;
    }
    return 0;
}
