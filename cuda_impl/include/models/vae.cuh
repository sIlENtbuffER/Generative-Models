#ifndef VAE_CUH
#define VAE_CUH

#define VAE_PARAMETERS 10

#include "core/rng.cuh"
#include "core/tensor.cuh"
#include "core/parameter.cuh"
#include "modules/linear.cuh"
#include "modules/activations.cuh"

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

    DeviceTensor d_logvar;
    DeviceTensor d_recon_loss;
    DeviceTensor d_kl_loss;
} VAE;

typedef struct
{
    DeviceTensor input;
    DeviceTensor enc_pre;
    DeviceTensor enc_hidden;
    DeviceTensor mu;
    DeviceTensor logvar;
    DeviceTensor eps;
    DeviceTensor z;
    DeviceTensor dec_pre;
    DeviceTensor dec_hidden;
    DeviceTensor logits;
    DeviceTensor output;
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
    uint64_t *seed
);

int vae_forward
(
    const VAE *vae,
    uint64_t *seed,
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
