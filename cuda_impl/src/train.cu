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

    if (config_load(&config, config_path) != 0) goto cleanup;

    seed = config.seed;
    if (data_load(&data, config.dataset_name, config.data_dir) != 0) goto cleanup;
    if (model_build(&model, config.model_name, data.channels * data.rows * data.cols, config.model, &seed) != 0) goto cleanup;

    if (config.training_enabled) {
        if (trainer_train(&model, &config, &seed, &data) != 0) goto cleanup;
    }

    status = EXIT_SUCCESS;

cleanup:
    model_free(&model);
    data_free(&data);
    config_free(&config);
    return status;
}
