#include "models/model.cuh"
#include "models/vae.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int model_build(Model *model, const char *name, size_t input_dim, size_t hidden_dim, size_t latent_dim, uint64_t *seed) {
    *model = (Model){};

    if (strcmp(name, "vae") == 0) {
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

    fprintf(stderr, "Unknown model: %s\n", name);
    return -1;
}

void model_free(Model *model) {
    if (model->type == MODEL_VAE) {
        VAE *vae = (VAE*)model->implementation;
        vae_free(vae);
        free(vae);
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
