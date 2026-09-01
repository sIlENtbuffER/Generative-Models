#include "tools/config.h"
#include "tools/checkpoint.h"
#include "data/data.h"
#include "models/model.cuh"
#include "trainer/trainer.cuh"
#include "optimizers/optimizer.cuh"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    int status = EXIT_FAILURE;
    const char *config_path = argc > 1 ? argv[1] : "configs/default.json";

    Config config = {};
    Data data = {};
    uint64_t seed = 0;
    Model model = {};
    Optimizer optimizer = {};
    size_t completed_epoch = 0;

    if (config_load(&config, config_path) != 0) goto cleanup;

    seed = config.seed;
    if (data_load(&data, config.dataset_name, config.data_dir) != 0) goto cleanup;
    if (model_build(&model, config.model_name, data.channels * data.rows * data.cols, config.hidden_dim, config.latent_dim, &seed) != 0) goto cleanup;
    if (optimizer_build(&optimizer, config.optimizer_name, model.parameters, model.num_parameters, config.learning_rate, config.beta1, config.beta2, config.eps) != 0) goto cleanup;

    if (config.load_checkpoint) {
        if (checkpoint_load(config.checkpoint_path, &model, &optimizer, &completed_epoch) != 0) goto cleanup;
        printf("Loaded checkpoint: %s, epoch: %zu\n", config.checkpoint_path, completed_epoch);
    }

    if (config.training_enabled && completed_epoch < config.epochs) {
        if (trainer_train(&model, &optimizer, &seed, config.batch_size, completed_epoch + 1, config.epochs, &data, config.num_samples, config.sample_dir, config.checkpoint_dir) != 0) goto cleanup;
    }

    status = EXIT_SUCCESS;

cleanup:
    optimizer_free(&optimizer);
    model_free(&model);
    data_free(&data);
    return status;
}
