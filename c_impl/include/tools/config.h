#ifdef __cplusplus
extern "C" {
#endif

#ifndef CONFIG_H
#define CONFIG_H

#include "third_party/cjson/cJSON.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct
{
    uint64_t seed;

    char dataset_name[32];
    char data_dir[512];

    char model_name[32];
    const cJSON *model;

    char optimizer_name[32];
    float learning_rate;
    float beta1;
    float beta2;
    float eps;

    bool training_enabled;
    size_t epochs;
    size_t batch_size;

    size_t num_samples;
    char sample_dir[512];

    char checkpoint_dir[512];
    bool load_checkpoint;
    char checkpoint_path[512];

    cJSON *root;
} Config;

int config_load
(
    Config *config,
    const char *path
);

void config_free(Config *config);

int config_get_size
(
    const cJSON *node,
    const char *key,
    size_t *value
);

int config_get_size_array
(
    const cJSON *node,
    const char *key,
    size_t **values,
    size_t *count
);

#endif

#ifdef __cplusplus
}
#endif
