#include "models/gan.h"
#include "modules/activations.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int stack_alloc(Linear **layers, char (**names)[GAN_PARAMETER_NAME_SIZE], const size_t *dims, size_t num_layers, const char *prefix) {
    *layers = calloc(num_layers, sizeof(Linear));
    *names = calloc(2 * num_layers, sizeof **names);
    if (*layers == NULL || *names == NULL) return -1;

    for (size_t i=0; i<num_layers; i++) {
        if (linear_alloc(&(*layers)[i], dims[i], dims[i+1]) != 0) return -1;
        snprintf((*names)[2 * i], GAN_PARAMETER_NAME_SIZE, "%s.fc%zu_W", prefix, i + 1);
        snprintf((*names)[2 * i + 1], GAN_PARAMETER_NAME_SIZE, "%s.fc%zu_b", prefix, i + 1);
    }

    return 0;
}

int gan_alloc(GAN *gan, size_t input_dim, size_t latent_dim, const size_t *g_hidden_dims, size_t num_g_hidden, const size_t *d_hidden_dims, size_t num_d_hidden) {
    size_t *dims = NULL;
    size_t num_dims = (num_g_hidden > num_d_hidden ? num_g_hidden : num_d_hidden) + 2;

    *gan = (GAN){0};
    gan->generator.input_dim = input_dim;
    gan->generator.latent_dim = latent_dim;
    gan->generator.num_layers = num_g_hidden + 1;
    gan->discriminator.input_dim = input_dim;
    gan->discriminator.num_layers = num_d_hidden + 1;

    dims = calloc(num_dims, sizeof(size_t));
    if (dims == NULL) goto cleanup;

    dims[0] = latent_dim;
    memcpy(&dims[1], g_hidden_dims, num_g_hidden * sizeof(size_t));
    dims[num_g_hidden + 1] = input_dim;
    if (stack_alloc(&gan->generator.layers, &gan->generator.parameter_names, dims, gan->generator.num_layers, "generator") != 0) goto cleanup;

    dims[0] = input_dim;
    memcpy(&dims[1], d_hidden_dims, num_d_hidden * sizeof(size_t));
    dims[num_d_hidden + 1] = 1;
    if (stack_alloc(&gan->discriminator.layers, &gan->discriminator.parameter_names, dims, gan->discriminator.num_layers, "discriminator") != 0) goto cleanup;

    free(dims);
    return 0;

cleanup:
    free(dims);
    gan_free(gan);
    return -1;
}

void gan_free(GAN *gan) {
    for (size_t i=0; i<gan->generator.num_layers && gan->generator.layers != NULL; i++) {
        linear_free(&gan->generator.layers[i]);
    }
    for (size_t i=0; i<gan->discriminator.num_layers && gan->discriminator.layers != NULL; i++) {
        linear_free(&gan->discriminator.layers[i]);
    }

    free(gan->generator.layers);
    free(gan->generator.parameter_names);
    free(gan->discriminator.layers);
    free(gan->discriminator.parameter_names);

    *gan = (GAN){0};
}

static int stack_workspace_alloc(StackWorkspace *stack, const Linear *layers, size_t num_layers, size_t batch_size) {
    *stack = (StackWorkspace){0};
    stack->num_layers = num_layers;

    stack->pre = calloc(num_layers, sizeof(Tensor));
    stack->hidden = calloc(num_layers, sizeof(Tensor));
    if (stack->pre == NULL || stack->hidden == NULL) return -1;

    if (tensor_alloc_2d(&stack->input, batch_size, layers[0].in_dim) != 0) return -1;
    for (size_t i=0; i<num_layers; i++) {
        if (tensor_alloc_2d(&stack->pre[i], batch_size, layers[i].out_dim) != 0) return -1;
        if (tensor_alloc_2d(&stack->hidden[i], batch_size, layers[i].out_dim) != 0) return -1;
    }

    return 0;
}

int stack_gn_workspace_alloc(StackWorkspace *stack, const Generator *generator, const size_t batch_size) {
    return stack_workspace_alloc(stack, generator->layers, generator->num_layers, batch_size);
}

int stack_dc_workspace_alloc(StackWorkspace *stack, const Discriminator *discriminator, const size_t batch_size) {
    return stack_workspace_alloc(stack, discriminator->layers, discriminator->num_layers, batch_size);
}

void stack_workspace_free(StackWorkspace *stack) {
    tensor_free(&stack->input);
    for (size_t i=0; i<stack->num_layers; i++) {
        if (stack->pre != NULL) tensor_free(&stack->pre[i]);
        if (stack->hidden != NULL) tensor_free(&stack->hidden[i]);
    }

    free(stack->pre);
    free(stack->hidden);

    *stack = (StackWorkspace){0};
}

int gan_workspace_alloc(GANWorkspace *ganws, const GAN *gan, size_t batch_size) {
    *ganws = (GANWorkspace){0};

    if (tensor_alloc_2d(&ganws->z, batch_size, gan->generator.latent_dim) != 0) goto cleanup;
    if (tensor_alloc_2d(&ganws->x_all, 2 * batch_size, gan->generator.input_dim) != 0) goto cleanup;
    if (tensor_alloc_2d(&ganws->y, 2 * batch_size, 1) != 0) goto cleanup;

    if (stack_gn_workspace_alloc(&ganws->g_for, &gan->generator, batch_size) != 0) goto cleanup;
    if (stack_gn_workspace_alloc(&ganws->g_bac, &gan->generator, batch_size) != 0) goto cleanup;
    if (stack_dc_workspace_alloc(&ganws->d_all_for, &gan->discriminator, 2 * batch_size) != 0) goto cleanup;
    if (stack_dc_workspace_alloc(&ganws->d_all_bac, &gan->discriminator, 2 * batch_size) != 0) goto cleanup;
    if (stack_dc_workspace_alloc(&ganws->d_fake_for, &gan->discriminator, batch_size) != 0) goto cleanup;
    if (stack_dc_workspace_alloc(&ganws->d_fake_bac, &gan->discriminator, batch_size) != 0) goto cleanup;

    return 0;

cleanup:
    gan_workspace_free(ganws);
    return -1;
}

void gan_workspace_free(GANWorkspace *ganws) {
    tensor_free(&ganws->z);
    tensor_free(&ganws->x_all);
    tensor_free(&ganws->y);

    stack_workspace_free(&ganws->g_for);
    stack_workspace_free(&ganws->g_bac);
    stack_workspace_free(&ganws->d_all_for);
    stack_workspace_free(&ganws->d_all_bac);
    stack_workspace_free(&ganws->d_fake_for);
    stack_workspace_free(&ganws->d_fake_bac);

    *ganws = (GANWorkspace){0};
}

int gan_init(GAN *gan, RNG *rng) {
    for (size_t i=0; i<gan->generator.num_layers; i++) {
        if (linear_he_init(&gan->generator.layers[i], rng) != 0) return -1;
    }
    for (size_t i=0; i<gan->discriminator.num_layers; i++) {
        if (linear_he_init(&gan->discriminator.layers[i], rng) != 0) return -1;
    }

    return 0;
}

int generator_forward(const Generator *generator, const Tensor *z, StackWorkspace *ws) {
    if (!tensor_is_same_shape(z, &ws->input)) return -1;
    memcpy(ws->input.data, z->data, z->numel * sizeof(float));

    if (linear_forward(&generator->layers[0], &ws->input, &ws->pre[0]) != 0) return -1;
    for (size_t i=1; i<generator->num_layers; i++) {
        if (leaky_relu_forward(&ws->pre[i-1], GAN_LEAKY_RELU_SLOPE, &ws->hidden[i-1]) != 0) return -1;
        if (linear_forward(&generator->layers[i], &ws->hidden[i-1], &ws->pre[i]) != 0) return -1;
    }

    const size_t last = generator->num_layers - 1;
    return tanh_forward(&ws->pre[last], &ws->hidden[last]);
}

int generator_backward(Generator *generator, const StackWorkspace *fw_ws, StackWorkspace *bw_ws) {
    const size_t last = generator->num_layers - 1;

    if (tanh_backward(&fw_ws->hidden[last], &bw_ws->hidden[last], &bw_ws->pre[last]) != 0) return -1;

    for (size_t i=last; i>=1; i--) {
        if (linear_backward(&generator->layers[i], &fw_ws->hidden[i-1], &bw_ws->pre[i], &bw_ws->hidden[i-1]) != 0) return -1;
        if (leaky_relu_backward(&fw_ws->pre[i-1], &bw_ws->hidden[i-1], GAN_LEAKY_RELU_SLOPE, &bw_ws->pre[i-1]) != 0) return -1;
    }

    return linear_backward(&generator->layers[0], &fw_ws->input, &bw_ws->pre[0], &bw_ws->input);
}

int discriminator_forward(const Discriminator *discriminator, const Tensor *x, StackWorkspace *ws) {
    if (!tensor_is_same_shape(x, &ws->input)) return -1;
    memcpy(ws->input.data, x->data, x->numel * sizeof(float));

    if (linear_forward(&discriminator->layers[0], &ws->input, &ws->pre[0]) != 0) return -1;
    for (size_t i=1; i<discriminator->num_layers; i++) {
        if (leaky_relu_forward(&ws->pre[i-1], GAN_LEAKY_RELU_SLOPE, &ws->hidden[i-1]) != 0) return -1;
        if (linear_forward(&discriminator->layers[i], &ws->hidden[i-1], &ws->pre[i]) != 0) return -1;
    }

    return 0;
}

int discriminator_backward(Discriminator *discriminator, const StackWorkspace *fw_ws, StackWorkspace *bw_ws) {
    const size_t last = discriminator->num_layers - 1;

    for (size_t i=last; i>=1; i--) {
        if (linear_backward(&discriminator->layers[i], &fw_ws->hidden[i-1], &bw_ws->pre[i], &bw_ws->hidden[i-1]) != 0) return -1;
        if (leaky_relu_backward(&fw_ws->pre[i-1], &bw_ws->hidden[i-1], GAN_LEAKY_RELU_SLOPE, &bw_ws->pre[i-1]) != 0) return -1;
    }

    return linear_backward(&discriminator->layers[0], &fw_ws->input, &bw_ws->pre[0], &bw_ws->input);
}

size_t generator_num_parameters(const Generator *generator) {
    return 2 * generator->num_layers;
}

size_t discriminator_num_parameters(const Discriminator *discriminator) {
    return 2 * discriminator->num_layers;
}

void generator_parameters(Generator *generator, Parameter *parameters) {
    for (size_t i=0; i<generator->num_layers; i++) {
        parameters[2 * i] = (Parameter){.name = generator->parameter_names[2 * i], .value = &generator->layers[i].W, .grad = &generator->layers[i].dW};
        parameters[2 * i + 1] = (Parameter){.name = generator->parameter_names[2 * i + 1], .value = &generator->layers[i].b, .grad = &generator->layers[i].db};
    }
}

void discriminator_parameters(Discriminator *discriminator, Parameter *parameters) {
    for (size_t i=0; i<discriminator->num_layers; i++) {
        parameters[2 * i] = (Parameter){.name = discriminator->parameter_names[2 * i], .value = &discriminator->layers[i].W, .grad = &discriminator->layers[i].dW};
        parameters[2 * i + 1] = (Parameter){.name = discriminator->parameter_names[2 * i + 1], .value = &discriminator->layers[i].b, .grad = &discriminator->layers[i].db};
    }
}
