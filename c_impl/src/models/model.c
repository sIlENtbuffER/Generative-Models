#include "models/model.h"
#include "models/vae.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int model_build(Model *model, const char *name, size_t input_dim, size_t hidden_dim, size_t latent_dim, RNG *rng) {
    *model = (Model){0};

    if (strcmp(name, "vae") == 0) {
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

    fprintf(stderr, "Unknown model: %s\n", name);
    return -1;
}

void model_free(Model *model) {
    if (model->type == MODEL_VAE) {
        VAE *vae = model->implementation;
        vae_free(vae);
        free(vae);
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
