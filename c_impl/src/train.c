#include "core/rng.h"
#include "tools/config.h"
#include "tools/checkpoint.h"
#include "data/data.h"
#include "models/model.h"
#include "trainer/trainer.h"
#include "optimizers/optimizer.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    int status = EXIT_FAILURE;
    const char *config_path = argc > 1 ? argv[1] : "configs/default.json";

    Config config = {0};
    RNG rng;
    Data data = {0};
    Model model = {0};
    Optimizer optimizer = {0};
    size_t completed_epoch = 0;

    if (config_load(&config, config_path) != 0) goto cleanup;

    rng_seed(&rng, config.seed, 1);
    if (data_load(&data, config.dataset_name, config.data_dir) != 0) goto cleanup;
    if (model_build(&model, config.model_name, data.rows * data.cols, config.hidden_dim, config.latent_dim, &rng) != 0) goto cleanup;
    if (optimizer_build(&optimizer, config.optimizer_name, model.parameters, model.num_parameters, config.learning_rate, config.beta1, config.beta2, config.eps) != 0) goto cleanup;

    if (config.load_checkpoint) {
        if (checkpoint_load(config.checkpoint_path, &model, &optimizer, &completed_epoch) != 0) goto cleanup;
        printf("Loaded checkpoint: %s, epoch: %zu\n", config.checkpoint_path, completed_epoch);
    }

    if (config.training_enabled && completed_epoch < config.epochs) {
        if (trainer_train(&model, &optimizer, &rng, config.batch_size, completed_epoch + 1, config.epochs, &data, config.num_samples, config.sample_dir, config.checkpoint_dir) != 0) goto cleanup;
    }

    status = EXIT_SUCCESS;

cleanup:
    optimizer_free(&optimizer);
    model_free(&model);
    data_free(&data);
    return status;
}
