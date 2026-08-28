#ifndef DATA_H
#define DATA_H

#include "core/tensor.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct
{
    size_t count;
    size_t rows;
    size_t cols;
    uint8_t *pixels;
} Data;

int data_alloc
(
    Data *data,
    size_t count,
    size_t rows,
    size_t cols
);

int data_load
(
    Data *data,
    const char *name,
    const char *data_dir
);

int data_batch
(
    const Data *data,
    size_t start,
    Tensor *output
);

void data_free(Data *data);

int data_write_png_grid
(
    const Data *data,
    const char *path
);

#endif
