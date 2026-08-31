#ifdef __cplusplus
extern "C" {
#endif

#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include "core/tensor.h"

#include <stddef.h>

#define CHECKPOINT_TENSOR_NAME_SIZE 128
#define CHECKPOINT_METADATA_KEY_SIZE 64
#define CHECKPOINT_METADATA_VALUE_SIZE 128

struct Model;
struct Optimizer;

typedef struct
{
    char name[CHECKPOINT_TENSOR_NAME_SIZE];
    Tensor tensor;
} CheckpointTensor;

typedef struct
{
    char key[64];
    char value[128];
} CheckpointMetadata;

typedef struct
{
    CheckpointTensor *tensors;
    size_t num_tensors;
    CheckpointMetadata *metadata;
    size_t num_metadata;
} Checkpoint;

void checkpoint_free(Checkpoint *checkpoint);

int checkpoint_add_tensor
(
    Checkpoint *checkpoint,
    const char *name,
    const Tensor *tensor
);

int checkpoint_take_tensor
(
    Checkpoint *checkpoint,
    const char *name,
    Tensor *tensor
);

const Tensor *checkpoint_get_tensor
(
    const Checkpoint *checkpoint,
    const char *name
);

int checkpoint_set_metadata
(
    Checkpoint *checkpoint,
    const char *key,
    const char *value
);

const char *checkpoint_get_metadata
(
    const Checkpoint *checkpoint,
    const char *key
);

int checkpoint_write
(
    const Checkpoint *checkpoint,
    const char *path
);

int checkpoint_read
(
    Checkpoint *checkpoint,
    const char *path
);

int checkpoint_save
(
    const char *path,
    struct Model *model,
    struct Optimizer *optimizer,
    size_t epoch
);

int checkpoint_load
(
    const char *path,
    struct Model *model,
    struct Optimizer *optimizer,
    size_t *epoch
);

#endif

#ifdef __cplusplus
}
#endif
