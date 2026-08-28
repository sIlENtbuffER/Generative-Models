#include "data/data.h"
#include "data/mnist.h"
#include "third_party/stb/stb_image_write.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int data_alloc(Data *data, size_t count, size_t rows, size_t cols) {
    *data = (Data){0};

    data->pixels = calloc(count * rows * cols, sizeof *data->pixels);
    if (data->pixels == NULL) return -1;

    data->count = count;
    data->rows = rows;
    data->cols = cols;

    return 0;
}

int data_load(Data *data, const char *name, const char *data_dir) {
    if (strcmp(name, "mnist") == 0) return mnist_load(data, data_dir);
    
    fprintf(stderr, "Unknown dataset: %s\n", name);
    return -1;
}

void data_free(Data *data) {
    if (data == NULL) return;
    free(data->pixels);
    *data = (Data){0};
}

int data_batch(const Data *data, size_t start, Tensor *output) {
    size_t image_size = data->rows * data->cols;
    if (output->shape[1] != image_size || start >= data->count || output->shape[0] > data->count - start) return -1;

    for (size_t r=0; r<output->shape[0]; r++) {
        size_t src_row = (start + r) * image_size;
        size_t des_row = r * output->shape[1];
        
        for (size_t c=0; c<image_size; c++) {
            output->data[des_row + c] = data->pixels[src_row + c] / 255.0f;
        }
    }
    return 0;
}

int data_write_png_grid(const Data *data, const char *path) {
    if (data->rows > SIZE_MAX / data->cols) return -1;
    size_t image_size = data->rows * data->cols;

    // Initialize grid
    size_t grid_cols = 1;
    while (grid_cols < data->count / grid_cols || grid_cols * grid_cols < data->count) {
        grid_cols++;
    }
    size_t grid_rows = data->count / grid_cols + (data->count % grid_cols != 0);
    if (grid_cols > SIZE_MAX / data->cols || grid_rows > SIZE_MAX / data->rows) return -1; // Check overflow

    size_t grid_width = grid_cols * data->cols;
    size_t grid_height = grid_rows * data->rows;
    if (grid_width > INT_MAX || grid_height > INT_MAX || grid_height > SIZE_MAX / grid_width) return -1; // The width, height, and stride in stb are int

    uint8_t *grid = calloc(grid_width * grid_height, sizeof *grid);
    if (grid == NULL) return -1;

    // Copy image to grid
    for (size_t i=0; i<data->count; i++) {
        size_t grid_row = i / grid_cols;
        size_t grid_col = i % grid_cols;

        for (size_t r=0; r<data->rows; r++) {
            const uint8_t *src = data->pixels + i * image_size + r * data->cols;
            uint8_t *des = grid + (grid_row * data->rows + r) * grid_width + grid_col * data->cols;
            memcpy(des, src, data->cols);
        }
    }

    int written = stbi_write_png(path, (int)grid_width, (int)grid_height, 1, grid, (int)grid_width);
    
    free(grid);
    
    if (!written) {
        fprintf(stderr, "Failed to write PNG\n");
        return -1;
    }
    return 0;
}

