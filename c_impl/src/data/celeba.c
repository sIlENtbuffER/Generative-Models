#include "data/celeba.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int celeba_load(Data *images, const char *data_dir) {
    if (images == NULL || data_dir == NULL) return -1;
    *images = (Data){0};
    FILE *file = NULL;
    uint8_t *pixels = NULL;
    int status = -1;
    uint32_t header[4]; // count, channels, rows, cols
    char path[1024];

    snprintf(path, sizeof(path), "%s/celeba.bin", data_dir);
    file = fopen(path, "rb");
    if (file == NULL) goto cleanup;
    if (fread(header, sizeof *header, 4, file) != 4) goto cleanup;

    size_t count = header[0];
    size_t channels = header[1];
    size_t rows = header[2];
    size_t cols = header[3];
    if (count == 0 || channels == 0 || rows == 0 || cols == 0) goto cleanup;

    size_t image_size = channels * rows * cols;
    size_t total_size = count * image_size;
    // Check overflow
    if (image_size / channels / rows != cols || total_size / count != image_size) goto cleanup;

    pixels = malloc(total_size);
    if (pixels == NULL) goto cleanup;
    if (fread(pixels, 1, total_size, file) != total_size) goto cleanup;

    images->count = count;
    images->channels = channels;
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
