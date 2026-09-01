#include "tools/config.h"
#include "third_party/cjson/cJSON.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static char *read_file(const char *path) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) return NULL;

    if (fseek(file, 0, SEEK_END) != 0) goto cleanup;
    long file_size = ftell(file);
    if (file_size < 0 || fseek(file, 0, SEEK_SET) != 0) goto cleanup;

    char *buffer = malloc((size_t)file_size + 1);
    if (buffer == NULL) goto cleanup;
    size_t bytes_read = fread(buffer, 1, (size_t)file_size, file);
    if (bytes_read != (size_t)file_size) {
        free(buffer);
        goto cleanup;
    }
    buffer[bytes_read] = '\0';

    fclose(file);
    return buffer;

cleanup:
    fclose(file);
    return NULL;
}

static int creat_dir(const char *path) {
    char buffer[512];
    snprintf(buffer, sizeof buffer, "%s", path);

    // Create each parent in turn by cutting the path short at every separator
    for (char *slash = strchr(buffer + 1, '/'); slash != NULL; slash = strchr(slash + 1, '/')) {
        *slash = '\0';
        if (mkdir(buffer, 0755) != 0 && errno != EEXIST) goto failed;
        *slash = '/';
    }
    if (mkdir(buffer, 0755) != 0 && errno != EEXIST) goto failed;

    return 0;

failed:
    fprintf(stderr, "Cannot create '%s': %s\n", buffer, strerror(errno));
    return -1;
}

int config_load(Config *config, const char *path) {
    int status = -1;
    char *text = read_file(path);
    cJSON *root = NULL;
    if (text == NULL) {
        fprintf(stderr, "Cannot read JSON config file\n");
        goto cleanup;
    }

    root = cJSON_Parse(text);
    if (root == NULL) {
        fprintf(stderr, "Invalid JSON config file\n");
        goto cleanup;
    }

    const cJSON *dataset = cJSON_GetObjectItemCaseSensitive(root, "dataset");
    const cJSON *model = cJSON_GetObjectItemCaseSensitive(root, "model");
    const cJSON *optimizer = cJSON_GetObjectItemCaseSensitive(root, "optimizer");
    const cJSON *training = cJSON_GetObjectItemCaseSensitive(root, "training");
    const cJSON *sampling = cJSON_GetObjectItemCaseSensitive(root, "sampling");
    const cJSON *checkpoint = cJSON_GetObjectItemCaseSensitive(root, "checkpoint");

    config->seed = (uint64_t)cJSON_GetObjectItemCaseSensitive(root, "seed")->valuedouble;
    snprintf(config->dataset_name, sizeof(config->dataset_name), "%s", cJSON_GetObjectItemCaseSensitive(dataset, "name")->valuestring);
    snprintf(config->data_dir, sizeof(config->data_dir), "%s", cJSON_GetObjectItemCaseSensitive(dataset, "data_dir")->valuestring);
    snprintf(config->model_name, sizeof(config->model_name), "%s", cJSON_GetObjectItemCaseSensitive(model, "name")->valuestring);
    config->hidden_dim = (size_t)cJSON_GetObjectItemCaseSensitive(model, "hidden_dim")->valuedouble;
    config->latent_dim = (size_t)cJSON_GetObjectItemCaseSensitive(model, "latent_dim")->valuedouble;
    snprintf(config->optimizer_name, sizeof(config->optimizer_name), "%s", cJSON_GetObjectItemCaseSensitive(optimizer, "name")->valuestring);
    config->learning_rate = (float)cJSON_GetObjectItemCaseSensitive(optimizer, "lr")->valuedouble;
    config->beta1 = (float)cJSON_GetObjectItemCaseSensitive(optimizer, "beta1")->valuedouble;
    config->beta2 = (float)cJSON_GetObjectItemCaseSensitive(optimizer, "beta2")->valuedouble;
    config->eps = (float)cJSON_GetObjectItemCaseSensitive(optimizer, "eps")->valuedouble;
    config->training_enabled = cJSON_IsTrue(cJSON_GetObjectItemCaseSensitive(training, "enabled"));
    config->epochs = (size_t)cJSON_GetObjectItemCaseSensitive(training, "epochs")->valuedouble;
    config->batch_size = (size_t)cJSON_GetObjectItemCaseSensitive(training, "batch_size")->valuedouble;
    config->num_samples = (size_t)cJSON_GetObjectItemCaseSensitive(sampling, "num_samples")->valuedouble;
    snprintf(config->sample_dir, sizeof(config->sample_dir), "%s", cJSON_GetObjectItemCaseSensitive(sampling, "sample_dir")->valuestring);
    snprintf(config->checkpoint_dir, sizeof(config->checkpoint_dir), "%s", cJSON_GetObjectItemCaseSensitive(checkpoint, "checkpoint_dir")->valuestring);
    config->load_checkpoint = cJSON_IsTrue(cJSON_GetObjectItemCaseSensitive(checkpoint, "load_checkpoint"));
    snprintf(config->checkpoint_path, sizeof(config->checkpoint_path), "%s", cJSON_GetObjectItemCaseSensitive(checkpoint, "load_checkpoint_path")->valuestring);

    if (creat_dir(config->checkpoint_dir) != 0 || creat_dir(config->sample_dir) != 0) goto cleanup;
    status = 0;

cleanup:
    cJSON_Delete(root);
    free(text);
    return status;
}
