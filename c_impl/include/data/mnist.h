#ifndef DATA_MNIST_H
#define DATA_MNIST_H

#include "core/tensor.h"
#include "data/data.h"

#include <stddef.h>
#include <stdint.h>

int mnist_load
(
    Data *images,
    const char *data_dir
);

#endif
