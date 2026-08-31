#include "tools/checkpoint.h"
#include "optimizers/optimizer.h"
#include "models/model.h"
#include "third_party/cjson/cJSON.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

static int write_u64_le(FILE *file, uint64_t value) {
    unsigned char bytes[8];

    for (size_t i = 0; i < 8; i++) {
        bytes[i] = (unsigned char)((value >> (8 * i)) & 0xff);
    }

    return fwrite(bytes, 1, 8, file) == 8 ? 0 : -1;
}

static int read_u64_le(FILE *file, uint64_t *value) {
    unsigned char bytes[8];
    if (fread(bytes, 1, 8, file) != 8) return -1;

    uint64_t result = 0;
    for (size_t i = 0; i < 8; i++) {
        result |= (uint64_t)bytes[i] << (8 * i);
    }

    *value = result;
    return 0;
}

void checkpoint_free(Checkpoint *checkpoint) {
    for (size_t i = 0; i < checkpoint->num_tensors; i++) {
        tensor_free(&checkpoint->tensors[i].tensor);
    }

    free(checkpoint->tensors);
    free(checkpoint->metadata);
    *checkpoint = (Checkpoint){0};
}

int checkpoint_add_tensor(Checkpoint *checkpoint, const char *name, const Tensor *tensor) {
    Tensor copy = {0};

    if (tensor_alloc(&copy, tensor->ndim, tensor->shape) != 0) return -1;
    memcpy(copy.data, tensor->data, tensor->numel * sizeof *tensor->data);

    if (checkpoint_take_tensor(checkpoint, name, &copy) != 0) {
        tensor_free(&copy);
        return -1;
    }

    return 0;
}

int checkpoint_take_tensor(Checkpoint *checkpoint, const char *name, Tensor *tensor) {
    size_t new_num_tensors = checkpoint->num_tensors + 1;
    CheckpointTensor *new_tensors = realloc(checkpoint->tensors, new_num_tensors * sizeof *new_tensors);
    if (new_tensors == NULL) return -1;

    checkpoint->tensors = new_tensors;

    CheckpointTensor *entry = &checkpoint->tensors[checkpoint->num_tensors];
    *entry = (CheckpointTensor){0};
    snprintf(entry->name, sizeof entry->name, "%s", name);

    entry->tensor = *tensor;
    *tensor = (Tensor){0};
    checkpoint->num_tensors = new_num_tensors;

    return 0;
}

const Tensor *checkpoint_get_tensor(const Checkpoint *checkpoint, const char *name) {
    for (size_t i=0; i<checkpoint->num_tensors; i++) {
        if (strcmp(checkpoint->tensors[i].name, name) == 0) {
            return &checkpoint->tensors[i].tensor;
        }
    }

    return NULL;
}

int checkpoint_set_metadata(Checkpoint *checkpoint, const char *key, const char *value) {
    for (size_t i=0; i<checkpoint->num_metadata; i++) {
        CheckpointMetadata *entry = &checkpoint->metadata[i];

        if (strcmp(entry->key, key) == 0) {
            snprintf(entry->value, sizeof entry->value, "%s", value);
            return 0;
        }
    }

    size_t new_num_metadata = checkpoint->num_metadata + 1;
    CheckpointMetadata *new_metadata = realloc(checkpoint->metadata, new_num_metadata * sizeof *new_metadata);
    if (new_metadata == NULL) return -1;

    checkpoint->metadata = new_metadata;
    checkpoint->num_metadata = new_num_metadata;

    CheckpointMetadata *entry = &checkpoint->metadata[new_num_metadata - 1];
    snprintf(entry->key, sizeof entry->key, "%s", key);
    snprintf(entry->value, sizeof entry->value, "%s", value);

    return 0;
}

const char *checkpoint_get_metadata(const Checkpoint *checkpoint, const char *key) {
    for (size_t i=0; i<checkpoint->num_metadata; i++) {
        if (strcmp(checkpoint->metadata[i].key, key) == 0) return checkpoint->metadata[i].value;
    }
    return NULL;
}

int checkpoint_write(const Checkpoint *checkpoint, const char *path) {
    int status = -1;
    FILE *file = NULL;
    cJSON *root = NULL;
    char *header_text = NULL;
    size_t data_offset = 0;

    root = cJSON_CreateObject();
    if (root == NULL) goto cleanup;

    cJSON *metadata = cJSON_CreateObject();
    if (metadata == NULL) goto cleanup;
    cJSON_AddItemToObject(root, "__metadata__", metadata);

    for (size_t i=0; i<checkpoint->num_metadata; i++) {
        const CheckpointMetadata *entry = &checkpoint->metadata[i];
        if (cJSON_AddStringToObject(metadata, entry->key, entry->value) == NULL) goto cleanup;
    }

    for (size_t i=0; i<checkpoint->num_tensors; i++) {
        const CheckpointTensor *entry = &checkpoint->tensors[i];
        const Tensor *tensor = &entry->tensor;

        if (tensor->numel > SIZE_MAX / sizeof(float)) goto cleanup;
        size_t data_size = tensor->numel * sizeof(float);
        if (data_offset > SIZE_MAX - data_size) goto cleanup;

        cJSON *tensor_json = cJSON_CreateObject();
        cJSON *shape = cJSON_CreateArray();
        cJSON *offsets = cJSON_CreateArray();
        if (tensor_json == NULL || shape == NULL || offsets == NULL) {
            cJSON_Delete(tensor_json);
            cJSON_Delete(shape);
            cJSON_Delete(offsets);
            goto cleanup;
        }

        cJSON_AddStringToObject(tensor_json, "dtype", "F32");

        for (size_t dim=0; dim<tensor->ndim; dim++) {
            cJSON_AddItemToArray(shape, cJSON_CreateNumber((double)tensor->shape[dim]));
        }

        cJSON_AddItemToArray(offsets, cJSON_CreateNumber((double)data_offset));
        cJSON_AddItemToArray(offsets, cJSON_CreateNumber((double)(data_offset + data_size)));

        cJSON_AddItemToObject(tensor_json, "shape", shape);
        cJSON_AddItemToObject(tensor_json, "data_offsets", offsets);

        cJSON_AddItemToObject(root, entry->name, tensor_json);

        data_offset += data_size;
    }

    header_text = cJSON_PrintUnformatted(root);
    if (header_text == NULL) goto cleanup;

    size_t header_size = strlen(header_text);
    size_t padding = (8 - header_size % 8) % 8;
    if (header_size > SIZE_MAX - padding) goto cleanup;

    size_t padded_header_size = header_size + padding;

    file = fopen(path, "wb");
    if (file == NULL) goto cleanup;

    if (write_u64_le(file, (uint64_t)padded_header_size) != 0) goto cleanup;
    if (fwrite(header_text, 1, header_size, file) != header_size) goto cleanup;

    for (size_t i=0; i<padding; i++) {
        if (fputc(' ', file) == EOF) goto cleanup;
    }

    for (size_t i=0; i<checkpoint->num_tensors; i++) {
        const Tensor *tensor = &checkpoint->tensors[i].tensor;
        if (fwrite(tensor->data, sizeof(float), tensor->numel, file) != tensor->numel) goto cleanup;
    }

    status = 0;

cleanup:
    if (file != NULL) fclose(file);
    cJSON_free(header_text);
    cJSON_Delete(root);
    return status;
}

int checkpoint_read(Checkpoint *checkpoint, const char *path) {
    int status = -1;
    FILE *file = NULL;
    char *header_text = NULL;
    cJSON *root = NULL;
    uint64_t header_size_u64;

    file = fopen(path, "rb");
    if (file == NULL) goto cleanup;

    if (read_u64_le(file, &header_size_u64) != 0 || header_size_u64 > SIZE_MAX - 1) goto cleanup;

    size_t header_size = (size_t)header_size_u64;
    header_text = malloc(header_size + 1);
    if (header_text == NULL || fread(header_text, 1, header_size, file) != header_size) goto cleanup;
    header_text[header_size] = '\0';

    root = cJSON_Parse(header_text);
    if (root == NULL) goto cleanup;

    cJSON *metadata = cJSON_GetObjectItemCaseSensitive(root, "__metadata__");

    if (cJSON_IsObject(metadata)) {
        cJSON *item = NULL;

        cJSON_ArrayForEach(item, metadata) {
            if (!cJSON_IsString(item)) goto cleanup;
            if (checkpoint_set_metadata(checkpoint, item->string, item->valuestring) != 0) goto cleanup;
        }
    }

    size_t tensor_data_start = 8 + header_size;

    for (size_t i=0; i<checkpoint->num_tensors; i++) {
        CheckpointTensor *entry = &checkpoint->tensors[i];
        Tensor *tensor = &entry->tensor;
        cJSON *tensor_json = cJSON_GetObjectItemCaseSensitive(root, entry->name);

        if (!cJSON_IsObject(tensor_json)) goto cleanup;

        cJSON *dtype = cJSON_GetObjectItemCaseSensitive(tensor_json, "dtype");
        cJSON *shape = cJSON_GetObjectItemCaseSensitive(tensor_json, "shape");
        cJSON *offsets = cJSON_GetObjectItemCaseSensitive(tensor_json, "data_offsets");

        if (!cJSON_IsString(dtype) || strcmp(dtype->valuestring, "F32") != 0) goto cleanup;
        if (!cJSON_IsArray(shape) || (size_t)cJSON_GetArraySize(shape) != tensor->ndim) goto cleanup;
        for (size_t dim=0; dim<tensor->ndim; dim++) {
            cJSON *size_json = cJSON_GetArrayItem(shape, (int)dim);
            if (!cJSON_IsNumber(size_json) || (size_t)size_json->valuedouble != tensor->shape[dim]) goto cleanup;
        }
        if (!cJSON_IsArray(offsets) || cJSON_GetArraySize(offsets) != 2) goto cleanup;

        cJSON *begin_json = cJSON_GetArrayItem(offsets, 0);
        cJSON *end_json = cJSON_GetArrayItem(offsets, 1);
        if (!cJSON_IsNumber(begin_json) || !cJSON_IsNumber(end_json)) goto cleanup;

        size_t begin = (size_t)begin_json->valuedouble;
        size_t end = (size_t)end_json->valuedouble;
        size_t expected_size = tensor->numel * sizeof(float);
        if (end < begin || end - begin != expected_size) goto cleanup;

        size_t file_offset = tensor_data_start + begin;
        if (file_offset > LONG_MAX) goto cleanup;

        if (fseek(file, (long)file_offset, SEEK_SET) != 0) goto cleanup;
        if (fread(tensor->data, sizeof(float), tensor->numel, file) != tensor->numel) goto cleanup;
    }

    status = 0;

cleanup:
    cJSON_Delete(root);
    free(header_text);
    if (file != NULL) fclose(file);
    return status;
}

int checkpoint_save(const char *path, struct Model *model, struct Optimizer *optimizer, size_t epoch) {
    Checkpoint checkpoint = {0};
    int status = -1;
    char epoch_text[32];

    snprintf(epoch_text, sizeof epoch_text, "%zu", epoch);
    if (checkpoint_set_metadata(&checkpoint, "epoch", epoch_text) != 0) goto cleanup;

    if (model_save_checkpoint(model, &checkpoint) != 0) goto cleanup;
    if (optimizer_save_checkpoint(optimizer, &checkpoint) != 0) goto cleanup;

    if (checkpoint_write(&checkpoint, path) != 0) goto cleanup;

    status = 0;

cleanup:
    checkpoint_free(&checkpoint);
    return status;
}

int checkpoint_load(const char *path, struct Model *model, struct Optimizer *optimizer, size_t *epoch) {
    Checkpoint checkpoint = {0};
    int status = -1;

    if (model_save_checkpoint(model, &checkpoint) != 0) goto cleanup;
    if (optimizer_save_checkpoint(optimizer, &checkpoint) != 0) goto cleanup;

    if (checkpoint_read(&checkpoint, path) != 0) goto cleanup;

    const char *epoch_text = checkpoint_get_metadata(&checkpoint, "epoch");
    if (epoch_text == NULL) goto cleanup;
    *epoch = (size_t)strtoull(epoch_text, NULL, 10);

    if (model_load_checkpoint(model, &checkpoint) != 0) goto cleanup;
    if (optimizer_load_checkpoint(optimizer, &checkpoint) != 0) goto cleanup;
    
    status = 0;

cleanup:
    checkpoint_free(&checkpoint);
    return status;
}
