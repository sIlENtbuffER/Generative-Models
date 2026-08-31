#ifndef RNG_CUH
#define RNG_CUH

#include <stdbool.h>
#include <stdint.h>

typedef struct
{
    uint64_t state;
    uint64_t increment;
    bool has_spare;
    float spare;
} RNG;

__device__ void rng_seed
(
    RNG *rng,
    uint64_t seed,
    uint64_t sequence
);

__device__ uint32_t rng_next_u32(RNG *rng);

__device__ float rng_uniform(RNG *rng);

__device__ float rng_normal(RNG *rng);

#endif
