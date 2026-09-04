#include "models/model.h"
#include "models/vae.h"
#include "models/gan.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int build_vae(Model *model, size_t input_dim, const cJSON *model_cfg, RNG *rng) {
    size_t hidden_dim;
    size_t latent_dim;
    if (config_get_size(model_cfg, "hidden_dim", &hidden_dim) != 0) return -1;
    if (config_get_size(model_cfg, "latent_dim", &latent_dim) != 0) return -1;

    VAE *vae = calloc(1, sizeof *vae);
    if (vae == NULL) return -1;
    if (vae_alloc(vae, input_dim, hidden_dim, latent_dim) != 0) {
        free(vae);
        return -1;
    }
    if (vae_init(vae, rng) != 0) {
        vae_free(vae);
        free(vae);
        return -1;
    }

    model->parameters = malloc(VAE_PARAMETERS * sizeof *model->parameters);
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

static int build_gan(Model *model, size_t input_dim, const cJSON *model_cfg, RNG *rng) {
    int status = -1;
    size_t latent_dim;
    size_t *g_hidden_dims = NULL;
    size_t *d_hidden_dims = NULL;
    size_t num_g_hidden = 0;
    size_t num_d_hidden = 0;
    GAN *gan = NULL;

    if (config_get_size(model_cfg, "latent_dim", &latent_dim) != 0) goto cleanup;
    if (config_get_size_array(model_cfg, "g_hidden_dims", &g_hidden_dims, &num_g_hidden) != 0) goto cleanup;
    if (config_get_size_array(model_cfg, "d_hidden_dims", &d_hidden_dims, &num_d_hidden) != 0) goto cleanup;

    gan = calloc(1, sizeof *gan);
    if (gan == NULL) goto cleanup;
    if (gan_alloc(gan, input_dim, latent_dim, g_hidden_dims, num_g_hidden, d_hidden_dims, num_d_hidden) != 0) goto cleanup;
    if (gan_init(gan, rng) != 0) goto cleanup;

    size_t num_generator = generator_num_parameters(&gan->generator);
    size_t num_discriminator = discriminator_num_parameters(&gan->discriminator);
    model->parameters = malloc((num_generator + num_discriminator) * sizeof *model->parameters);
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

int model_build(Model *model, const char *name, size_t input_dim, const cJSON *model_cfg, RNG *rng) {
    *model = (Model){0};

    if (strcmp(name, "vae") == 0) return build_vae(model, input_dim, model_cfg, rng);
    if (strcmp(name, "gan") == 0) return build_gan(model, input_dim, model_cfg, rng);

    fprintf(stderr, "Unknown model: %s\n", name);
    return -1;
}

void model_free(Model *model) {
    if (model->type == MODEL_VAE) {
        VAE *vae = model->implementation;
        vae_free(vae);
        free(vae);
    } else if (model->type == MODEL_GAN) {
        GAN *gan = model->implementation;
        gan_free(gan);
        free(gan);
    }

    free(model->parameters);
    *model = (Model){0};
}

int model_save_checkpoint(const Model *model, Checkpoint *checkpoint) {
    char name[128];
    
    for (size_t i=0; i<model->num_parameters; i++) {
        Parameter *parameter = &model->parameters[i];
        snprintf(name, sizeof(name), "model.%s", parameter->name);
        
        if (checkpoint_add_tensor(checkpoint, name, parameter->value) != 0) return -1;
    }
    return 0;
}

int model_load_checkpoint(Model *model, const Checkpoint *checkpoint) {
    char name[CHECKPOINT_TENSOR_NAME_SIZE];

    for (size_t i=0; i<model->num_parameters; i++) {
        Parameter *parameter = &model->parameters[i];
        snprintf(name, sizeof name, "model.%s", parameter->name);
        const Tensor *cpt = checkpoint_get_tensor(checkpoint, name);
        if (cpt == NULL) return -1;
        memcpy(parameter->value->data, cpt->data, cpt->numel * sizeof *cpt->data);
    }
    return 0;
}
