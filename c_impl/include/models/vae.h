#ifndef VAE_H
#define VAE_H
#define VAE_PARAMETERS 10

#include "core/rng.h"
#include "core/tensor.h"
#include "core/parameter.h"
#include "modules/linear.h"
#include "modules/activations.h"

typedef struct 
{
    size_t input_dim;
    size_t hidden_dim;
    size_t latent_dim;

    Linear fc1;
    Linear fc_mu;
    Linear fc_logvar;
    Linear fc2;
    Linear fc3;
} VAE;

typedef struct
{
    Tensor input;
    Tensor enc_pre;
    Tensor enc_hidden;
    Tensor mu;
    Tensor logvar;
    Tensor eps;
    Tensor z;
    Tensor dec_pre;
    Tensor dec_hidden;
    Tensor logits;
    Tensor output;
} VAEWorkspace;

typedef struct
{
    float loss;
    float recon_loss;
    float kl_loss;
} VAELoss;


int vae_alloc
(
    VAE *vae,
    size_t input_dim,
    size_t hidden_dim,
    size_t latent_dim
);

int vae_free(VAE *vae);

int vae_workspace_alloc
(
    const VAE *vae,
    size_t batch_size,
    VAEWorkspace *vaews
);

int vae_workspace_free(VAEWorkspace *vaews);

int vae_init
(
    VAE *vae,
    RNG *rng
);

int vae_forward
(
    const VAE *vae,
    RNG *rng,
    VAEWorkspace *vaews
);

int vae_backward
(
    VAE *vae,
    const VAEWorkspace *vae_for_ws,
    VAEWorkspace *vae_bac_ws,
    VAELoss *loss
);

void vae_parameters
(
    VAE *vae,
    Parameter parameters[VAE_PARAMETERS]
);

#endif
