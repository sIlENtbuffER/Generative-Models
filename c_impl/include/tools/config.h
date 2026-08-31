#ifdef __cplusplus
extern "C" {
#endif

#ifndef CONFIG_H
#define CONFIG_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct
{
    uint64_t seed;

    char dataset_name[32];
    char data_dir[512];

    char model_name[32];
    size_t hidden_dim;
    size_t latent_dim;

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
} Config;

int config_load
(
    Config *config,
    const char *path
);

#endif

#ifdef __cplusplus
}
#endif
