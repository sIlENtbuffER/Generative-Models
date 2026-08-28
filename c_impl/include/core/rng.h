#ifndef CORE_RNG_H
#define CORE_RNG_H

#include <stdbool.h>
#include <stdint.h>

typedef struct
{
    uint64_t state;
    uint64_t increment;
    bool has_spare;
    float spare;
} RNG;

void rng_seed
(
    RNG *rng,
    uint64_t seed,
    uint64_t sequence
);

uint32_t rng_next_u32(RNG *rng);

float rng_uniform(RNG *rng);

float rng_normal(RNG *rng);

#endif
