#include "core/rng.h"

#include <math.h>

uint32_t rng_next_u32(RNG *rng) {
    // PCG32
    uint64_t old_state = rng->state;
    rng->state = old_state * 6364136223846793005ULL + rng->increment;
    uint32_t xsh = (uint32_t) (((old_state >> 18u) ^ old_state) >> 27u);
    uint32_t rr = (uint32_t) (old_state >> 59u);
    return (xsh >> rr) | (xsh << ((-rr) & 31));
}

void rng_seed(RNG *rng, uint64_t seed, uint64_t sequence) {
    rng->state = 0u;
    rng->increment = (sequence << 1u) | 1u;
    rng->has_spare = false;
    rng->spare = 0.0f;

    rng_next_u32(rng);
    rng->state += seed;
    rng_next_u32(rng);
}

float rng_uniform(RNG *rng) {
    // 24 bits for float mantissa
    uint32_t value = rng_next_u32(rng) >> 8;
    return (float)value * (1.0f / 16777216.0f);
}

float rng_normal(RNG *rng) {
    if (rng->has_spare) {
        rng->has_spare = false;
        return rng->spare;
    }
    
    // Box-Muller transform
    float u1 = 1.0f - rng_uniform(rng);
    float u2 = rng_uniform(rng);
    float radius = sqrtf(-2.0f * logf(u1)); // Rayleigh distribution
    float angle = 2.0f * 3.14159265358979323846f * u2;
    float z0 = radius * cosf(angle);
    float z1 = radius * sinf(angle);

    rng->spare = z1;
    rng->has_spare = true;
    return z0;
}
