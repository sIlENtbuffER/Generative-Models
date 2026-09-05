#include "trainer/gan_trainer.h"
#include "models/gan.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static float sigmoid(float z) {
    return z >= 0.0f ? 1.0f / (1.0f + expf(-z)) : expf(z) / (1.0f + expf(z));
}

int gan_train_batch(GAN *gan, Optimizer *g_optimizer, Optimizer *d_optimizer, RNG *rng, const Tensor *x_real, GANWorkspace *ganws, GANLoss *loss) {
    const size_t batch_size = x_real->shape[0];
    const size_t g_last = gan->generator.num_layers - 1;
    const size_t d_last = gan->discriminator.num_layers - 1;

    loss->d_loss = 0.0f;
    loss->g_loss = 0.0f;

    for (size_t i=0; i<ganws->z.numel; i++) {
        ganws->z.data[i] = rng_normal(rng);
    }
    if (generator_forward(&gan->generator, &ganws->z, &ganws->g_for) != 0) return -1;

    const Tensor *x_fake = &ganws->g_for.hidden[g_last];

    // D step
    memcpy(ganws->x_all.data, x_fake->data, x_fake->numel * sizeof(float));
    memcpy(&ganws->x_all.data[x_fake->numel], x_real->data, x_real->numel * sizeof(float));
    for (size_t i=0; i<ganws->y.numel; i++) {
        ganws->y.data[i] = i < batch_size ? 0.0f : 1.0f;
    }

    if (discriminator_forward(&gan->discriminator, &ganws->x_all, &ganws->d_all_for) != 0) return -1;

    const Tensor *d_logits = &ganws->d_all_for.pre[d_last];
    for (size_t i=0; i<d_logits->numel; i++) {
        float z = d_logits->data[i];
        float y = ganws->y.data[i];
        loss->d_loss += fmaxf(z, 0.0f) - y * z + log1pf(expf(-fabsf(z)));
        ganws->d_all_bac.pre[d_last].data[i] = (sigmoid(z) - y) / (float)(2 * batch_size);
    }
    loss->d_loss /= (float)(2 * batch_size);

    if (discriminator_backward(&gan->discriminator, &ganws->d_all_for, &ganws->d_all_bac) != 0) return -1;
    if (optimizer_step(d_optimizer) != 0) return -1;

    // G step
    if (discriminator_forward(&gan->discriminator, x_fake, &ganws->d_fake_for) != 0) return -1;

    const Tensor *g_logits = &ganws->d_fake_for.pre[d_last];
    for (size_t i=0; i<g_logits->numel; i++) {
        float z = g_logits->data[i];
        loss->g_loss += fmaxf(-z, 0.0f) + log1pf(expf(-fabsf(z)));
        ganws->d_fake_bac.pre[d_last].data[i] = (sigmoid(z) - 1.0f) / (float)batch_size;
    }
    loss->g_loss /= (float)batch_size;

    if (discriminator_backward(&gan->discriminator, &ganws->d_fake_for, &ganws->d_fake_bac) != 0) return -1;
    memcpy(ganws->g_bac.hidden[g_last].data, ganws->d_fake_bac.input.data, ganws->d_fake_bac.input.numel * sizeof(float));
    if (generator_backward(&gan->generator, &ganws->g_for, &ganws->g_bac) != 0) return -1;
    if (optimizer_step(g_optimizer) != 0) return -1;

    return 0;
}

int gan_train_epoch(GAN *gan, Optimizer *g_optimizer, Optimizer *d_optimizer, RNG *rng, size_t batch_size, size_t epoch, Data *data, Data *samples, const char *sample_dir) {
    int status = -1;
    char sample_path[1024];
    GANLoss loss = {0};
    size_t seen = 0;
    GANWorkspace ganws = {0};
    Tensor x_real = {0};

    if (gan_workspace_alloc(&ganws, gan, batch_size) != 0) goto cleanup;
    if (tensor_alloc_2d(&x_real, batch_size, gan->discriminator.input_dim) != 0) goto cleanup;

    for (size_t start=0; start+batch_size<=data->count; start+=batch_size) {
        GANLoss batch_loss;

        if (data_batch(data, start, &x_real) != 0) goto cleanup;
        // data_batch normalises to [0, 1]; the generator's tanh lives in [-1, 1]
        for (size_t i=0; i<x_real.numel; i++) {
            x_real.data[i] = x_real.data[i] * 2.0f - 1.0f;
        }

        if (gan_train_batch(gan, g_optimizer, d_optimizer, rng, &x_real, &ganws, &batch_loss) != 0) goto cleanup;

        loss.d_loss += batch_loss.d_loss * batch_size;
        loss.g_loss += batch_loss.g_loss * batch_size;
        seen += batch_size;
    }

    loss.d_loss /= seen;
    loss.g_loss /= seen;

    printf("Epoch %zu | d_loss: %.4f | g_loss: %.4f\n", epoch, loss.d_loss, loss.g_loss);

    if (gan_sample(&gan->generator, rng, samples) != 0) goto cleanup;
    snprintf(sample_path, sizeof sample_path, "%s/epoch_%zu.png", sample_dir, epoch);

    if (data_write_png_grid(samples, sample_path) != 0) goto cleanup;

    status = 0;

cleanup:
    gan_workspace_free(&ganws);
    tensor_free(&x_real);
    return status;
}

int gan_sample(const Generator *generator, RNG *rng, Data *output) {
    int status = -1;
    StackWorkspace ws = {0};
    Tensor z = {0};

    if (stack_workspace_alloc(&ws, generator->layers, generator->num_layers, output->count) != 0) goto cleanup;
    if (tensor_alloc_2d(&z, output->count, generator->latent_dim) != 0) goto cleanup;

    for (size_t i=0; i<z.numel; i++) {
        z.data[i] = rng_normal(rng);
    }
    if (generator_forward(generator, &z, &ws) != 0) goto cleanup;

    const Tensor *x_fake = &ws.hidden[generator->num_layers - 1];
    for (size_t i=0; i<output->count * output->channels * output->rows * output->cols; i++) {
        output->pixels[i] = (uint8_t)((x_fake->data[i] + 1.0f) * 127.5f);
    }

    status = 0;

cleanup:
    stack_workspace_free(&ws);
    tensor_free(&z);
    return status;
}
