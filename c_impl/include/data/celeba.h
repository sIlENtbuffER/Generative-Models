#ifndef DATA_CELEBA_H
#define DATA_CELEBA_H

#include "core/tensor.h"
#include "data/data.h"

#include <stddef.h>
#include <stdint.h>

int celeba_load
(
    Data *images,
    const char *data_dir
);

#endif
