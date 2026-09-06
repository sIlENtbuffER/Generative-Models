#include "core/rng.h"
#include "tools/config.h"
#include "data/data.h"
#include "models/model.h"
#include "trainer/trainer.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    int status = EXIT_FAILURE;
    const char *config_path = argc > 1 ? argv[1] : "configs/default.json";

    Config config = {0};
    RNG rng;
    Data data = {0};
    Model model = {0};

    if (config_load(&config, config_path) != 0) goto cleanup;

    rng_seed(&rng, config.seed, 1);
    if (data_load(&data, config.dataset_name, config.data_dir) != 0) goto cleanup;
    if (model_build(&model, config.model_name, data.channels * data.rows * data.cols, config.model, &rng) != 0) goto cleanup;

    if (config.training_enabled) {
        if (trainer_train(&model, &config, &rng, &data) != 0) goto cleanup;
    }

    status = EXIT_SUCCESS;

cleanup:
    model_free(&model);
    data_free(&data);
    config_free(&config);
    return status;
}
