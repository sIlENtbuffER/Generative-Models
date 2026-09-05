#ifndef GAN_H
#define GAN_H
#define GAN_LEAKY_RELU_SLOPE 0.2f
#define GAN_PARAMETER_NAME_SIZE 32

#include "core/rng.h"
#include "core/tensor.h"
#include "core/parameter.h"
#include "modules/linear.h"
#include "modules/activations.h"

#include <stddef.h>

typedef struct
{
    size_t input_dim;
    size_t latent_dim;
    Linear *layers;
    size_t num_layers;
    char (*parameter_names)[GAN_PARAMETER_NAME_SIZE];
} Generator;

typedef struct
{
    size_t input_dim;
    Linear *layers;
    size_t num_layers;
    char (*parameter_names)[GAN_PARAMETER_NAME_SIZE];
} Discriminator;

typedef struct
{
    Generator generator;
    Discriminator discriminator;
} GAN;

typedef struct
{
    Tensor input;
    Tensor *pre;
    Tensor *hidden;
    size_t num_layers;
} StackWorkspace;

typedef struct
{
    StackWorkspace g_for;
    StackWorkspace g_bac;
    StackWorkspace d_all_for;
    StackWorkspace d_all_bac;
    StackWorkspace d_fake_for;
    StackWorkspace d_fake_bac;

    Tensor z;
    Tensor x_all;
    Tensor y;
} GANWorkspace;

typedef struct
{
    float d_loss;
    float g_loss;
} GANLoss;

int gan_alloc
(
    GAN *gan,
    size_t input_dim,
    size_t latent_dim,
    const size_t *g_hidden_dims,
    size_t num_g_hidden,
    const size_t *d_hidden_dims,
    size_t num_d_hidden
);

void gan_free(GAN *gan);

int stack_workspace_alloc
(
    StackWorkspace *stack,
    const Linear *layers,
    size_t num_layers,
    size_t batch_size
);

void stack_workspace_free(StackWorkspace *stack);

int gan_workspace_alloc
(
    GANWorkspace *ganws,
    const GAN *gan,
    size_t batch_size
);

void gan_workspace_free(GANWorkspace *ganws);

int gan_init
(
    GAN *gan,
    RNG *rng
);

int generator_forward
(
    const Generator *generator,
    const Tensor *z,
    StackWorkspace *ws
);

int generator_backward
(
    Generator *generator,
    const StackWorkspace *fw_ws,
    StackWorkspace *bw_ws
);

int discriminator_forward
(
    const Discriminator *discriminator,
    const Tensor *x,
    StackWorkspace *ws
);

int discriminator_backward
(
    Discriminator *discriminator,
    const StackWorkspace *fw_ws,
    StackWorkspace *bw_ws
);

size_t generator_num_parameters(const Generator *generator);

size_t discriminator_num_parameters(const Discriminator *discriminator);

void generator_parameters
(
    Generator *generator,
    Parameter *parameters
);

void discriminator_parameters
(
    Discriminator *discriminator,
    Parameter *parameters
);

#endif
