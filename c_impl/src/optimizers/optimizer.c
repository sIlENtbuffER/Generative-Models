#include "optimizers/optimizer.h"
#include "optimizers/adam.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int optimizer_build(Optimizer *optimizer, const char *name, const Parameter *parameters, size_t num_parameters, float learning_rate, float beta1, float beta2, float eps) {
    *optimizer = (Optimizer){0};

    if (strcmp(name, "adam") == 0) {
        Adam *adam = calloc(1, sizeof *adam);
        if (adam == NULL) return -1;
        if (adam_alloc(adam, parameters, num_parameters, learning_rate, beta1, beta2, eps) != 0) {
            free(adam);
            return -1;
        }

        optimizer->type = OPTIMIZER_ADAM;
        optimizer->implementation = adam;
        return 0;
    }

    fprintf(stderr, "Unknown optimizer: %s\n", name);
    return -1;
}

void optimizer_free(Optimizer *optimizer) {
    if (optimizer->type == OPTIMIZER_ADAM) {
        Adam *adam = optimizer->implementation;
        adam_free(adam);
        free(adam);
    }

    *optimizer = (Optimizer){0};
}

int optimizer_step(Optimizer *optimizer) {
    if (optimizer->type == OPTIMIZER_ADAM) {
        return adam_step(optimizer->implementation);
    }

    return -1;
}

int optimizer_save_checkpoint(Optimizer *optimizer, Checkpoint *checkpoint) {
    if (optimizer->type == OPTIMIZER_ADAM) {
        return adam_save_checkpoint(optimizer->implementation, checkpoint);
    }
    return -1;
}

int optimizer_load_checkpoint(Optimizer *optimizer, const Checkpoint * checkpoint) {
    if (optimizer->type == OPTIMIZER_ADAM) {
        return adam_load_checkpoint(optimizer->implementation, checkpoint);
    }
    return -1;
}
