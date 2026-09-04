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

static const cJSON *get_node(const cJSON *parent, const char *key) {
    const cJSON *node = cJSON_GetObjectItemCaseSensitive(parent, key);
    if (node == NULL) fprintf(stderr, "Config is missing '%s'\n", key);
    return node;
}

static int get_string(const cJSON *parent, const char *key, char *value, size_t size) {
    const cJSON *node = get_node(parent, key);
    if (!cJSON_IsString(node)) return -1;
    snprintf(value, size, "%s", node->valuestring);
    return 0;
}

static int get_number(const cJSON *parent, const char *key, double *value) {
    const cJSON *node = get_node(parent, key);
    if (!cJSON_IsNumber(node)) return -1;
    *value = node->valuedouble;
    return 0;
}

static int get_bool(const cJSON *parent, const char *key, bool *value) {
    const cJSON *node = get_node(parent, key);
    if (!cJSON_IsBool(node)) return -1;
    *value = cJSON_IsTrue(node);
    return 0;
}

int config_get_size(const cJSON *node, const char *key, size_t *value) {
    double number;
    if (get_number(node, key, &number) != 0) return -1;
    *value = (size_t)number;
    return 0;
}

int config_get_size_array(const cJSON *node, const char *key, size_t **values, size_t *count) {
    const cJSON *array = get_node(node, key);
    if (!cJSON_IsArray(array)) return -1;

    *count = (size_t)cJSON_GetArraySize(array);
    *values = malloc(*count * sizeof **values);
    if (*values == NULL) return -1;

    for (size_t i=0; i<*count; i++) {
        const cJSON *item = cJSON_GetArrayItem(array, (int)i);
        if (!cJSON_IsNumber(item)) {
            free(*values);
            *values = NULL;
            return -1;
        }
        (*values)[i] = (size_t)item->valuedouble;
    }
    return 0;
}

void config_free(Config *config) {
    cJSON_Delete(config->root);
    config->root = NULL;
    config->model = NULL;
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

    const cJSON *dataset = get_node(root, "dataset");
    const cJSON *model = get_node(root, "model");
    const cJSON *optimizer = get_node(root, "optimizer");
    const cJSON *training = get_node(root, "training");
    const cJSON *sampling = get_node(root, "sampling");
    const cJSON *checkpoint = get_node(root, "checkpoint");
    if (dataset == NULL || model == NULL || optimizer == NULL || training == NULL || sampling == NULL || checkpoint == NULL) goto cleanup;

    double seed;
    double learning_rate, beta1, beta2, eps;
    if (get_number(root, "seed", &seed) != 0) goto cleanup;
    if (get_string(dataset, "name", config->dataset_name, sizeof config->dataset_name) != 0) goto cleanup;
    if (get_string(dataset, "data_dir", config->data_dir, sizeof config->data_dir) != 0) goto cleanup;
    if (get_string(model, "name", config->model_name, sizeof config->model_name) != 0) goto cleanup;
    if (get_string(optimizer, "name", config->optimizer_name, sizeof config->optimizer_name) != 0) goto cleanup;
    if (get_number(optimizer, "lr", &learning_rate) != 0) goto cleanup;
    if (get_number(optimizer, "beta1", &beta1) != 0) goto cleanup;
    if (get_number(optimizer, "beta2", &beta2) != 0) goto cleanup;
    if (get_number(optimizer, "eps", &eps) != 0) goto cleanup;
    if (get_bool(training, "enabled", &config->training_enabled) != 0) goto cleanup;
    if (config_get_size(training, "epochs", &config->epochs) != 0) goto cleanup;
    if (config_get_size(training, "batch_size", &config->batch_size) != 0) goto cleanup;
    if (config_get_size(sampling, "num_samples", &config->num_samples) != 0) goto cleanup;
    if (get_string(sampling, "sample_dir", config->sample_dir, sizeof config->sample_dir) != 0) goto cleanup;
    if (get_string(checkpoint, "checkpoint_dir", config->checkpoint_dir, sizeof config->checkpoint_dir) != 0) goto cleanup;
    if (get_bool(checkpoint, "load_checkpoint", &config->load_checkpoint) != 0) goto cleanup;
    if (get_string(checkpoint, "load_checkpoint_path", config->checkpoint_path, sizeof config->checkpoint_path) != 0) goto cleanup;

    config->seed = (uint64_t)seed;
    config->learning_rate = (float)learning_rate;
    config->beta1 = (float)beta1;
    config->beta2 = (float)beta2;
    config->eps = (float)eps;
    config->root = root;
    config->model = model;

    if (creat_dir(config->checkpoint_dir) != 0 || creat_dir(config->sample_dir) != 0) goto cleanup;

    free(text);
    return 0;

cleanup:
    cJSON_Delete(root);
    free(text);
    return status;
}
