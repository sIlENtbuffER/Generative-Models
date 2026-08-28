#include "data/mnist.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int read_u32_be(FILE *file, uint32_t *value) {
    uint8_t bytes[4];
    if (fread(bytes, 1, 4, file) != 4) return -1;
    *value = ((uint32_t)bytes[0] << 24) | ((uint32_t)bytes[1] << 16) | ((uint32_t)bytes[2] << 8) | (uint32_t)bytes[3];
    
    return 0;
}

int mnist_load(Data *images, const char *data_dir) {
    if (images == NULL || data_dir == NULL) return -1;
    *images = (Data){0};
    FILE *file = NULL;
    uint8_t *pixels = NULL;
    int status = -1;
    uint32_t magic;
    uint32_t count;
    uint32_t rows;
    uint32_t cols;
    char path[1024];

    snprintf(path, sizeof(path), "%s/train-images-idx3-ubyte", data_dir);
    file = fopen(path, "rb");
    if (file == NULL) goto cleanup;
    if (read_u32_be(file, &magic) != 0 || read_u32_be(file, &count) != 0 || read_u32_be(file, &rows) != 0 || read_u32_be(file, &cols) != 0) goto cleanup;
    if (magic != 2051 || count == 0 || rows == 0 || cols == 0) goto cleanup; // MNIST image magic number

    size_t image_size = (size_t)rows * (size_t)cols;
    size_t total_size = (size_t)count * image_size;
    // Check overflow
    if (image_size / (size_t)rows != (size_t)cols || total_size / (size_t)count != image_size) goto cleanup;

    pixels = malloc(total_size);
    if (pixels == NULL) goto cleanup;
    if (fread(pixels, 1, total_size, file) != total_size) goto cleanup;

    images->count = count;
    images->rows = rows;
    images->cols = cols;
    images->pixels = pixels;

    pixels = NULL;
    status = 0;

cleanup:
    free(pixels);
    if (file != NULL) fclose(file);
    return status;
}
