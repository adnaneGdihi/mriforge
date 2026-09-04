# Example: Using Configurable Latent Architecture

## Quick Start

```python
from spectramr.models.latent.configurable_vae import ConfigurableVAE, create_configurable_vae

# 1. Simple VAE
vae_simple = create_configurable_vae(preset="simple", latent_dim=256)

# 2. ResNet VAE with Attention
vae_advanced = create_configurable_vae(
    preset="resnet_attention",
    latent_dim=512,
    beta=0.5,  # KL weight
)

# 3. Custom configuration
from spectramr.models.latent import EncoderConfig, DecoderConfig, LatentConfig

encoder_cfg = EncoderConfig(
    arch_type="resnet",
    hidden_dims=[64, 128, 256, 512],
    use_attention=True,
    attention_layers=[3],
    use_batch_norm=True,
    dropout_rate=0.1,
    activation="silu",
)

decoder_cfg = DecoderConfig(
    arch_type="resnet",
    hidden_dims=[512, 256, 128, 64],
    use_attention=True,
    attention_layers=[0],
    output_activation="tanh",
)

latent_cfg = LatentConfig(
    latent_dim=512,
    distribution="gaussian",
)

vae_custom = ConfigurableVAE(encoder_cfg, decoder_cfg, latent_cfg, beta=1.0)
```

## Training

```python
import torch
import torch.optim as optim

# Create model
model = create_configurable_vae(preset="resnet_attention", latent_dim=512)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
model.train()
for batch in dataloader:
    optimizer.zero_grad()

    # Compute loss
    losses = model.compute_loss(batch)
    total_loss = losses["total_loss"]

    # Backprop
    total_loss.backward()
    optimizer.step()

    print(
        f"Loss: {total_loss.item():.4f}, "
        f"Recon: {losses['recon_loss'].item():.4f}, "
        f"KL: {losses['kl_loss'].item():.4f}"
    )
```

## Available Presets

1. **simple**: Basic convolutional VAE
   - Conv encoder/decoder
   - Hidden dims: [32, 64, 128, 256]
   - Latent dim: 256

2. **resnet**: ResNet-based VAE
   - ResNet blocks with skip connections
   - Hidden dims: [64, 128, 256, 512]
   - Latent dim: 512

3. **attention**: Conv + Self-Attention
   - Hybrid architecture
   - Attention at deeper layers
   - Hidden dims: [64, 128, 256, 512]

4. **resnet_attention**: ResNet + Attention
   - Best of both worlds
   - Attention at bottleneck
   - Hidden dims: [64, 128, 256, 512]

5. **deep**: Very deep ResNet
   - 6-layer encoder/decoder
   - Hidden dims: [32, 64, 128, 256, 512, 1024]
   - Latent dim: 1024

## Advanced Features

### VQ-VAE
```python
latent_cfg = LatentConfig(
    latent_dim=256,
    distribution="vq",
    num_embeddings=512,
    commitment_cost=0.25,
)

vae = ConfigurableVAE(
    encoder_config=encoder_cfg,
    decoder_config=decoder_cfg,
    latent_config=latent_cfg,
)
```

### Custom Architecture Components
```python
encoder_cfg = EncoderConfig(
    arch_type="resnet",  # or "conv", "attention", "hybrid"
    hidden_dims=[64, 128, 256, 512],
    use_batch_norm=True,  # Batch normalization
    use_layer_norm=False,  # Layer normalization (mutually exclusive)
    use_spectral_norm=False,  # Spectral normalization
    dropout_rate=0.1,
    activation="silu",  # or "relu", "gelu", "leaky_relu"
    use_attention=True,
    num_heads=8,
    attention_layers=[2, 3],  # Add attention at these layer indices
    use_residual=True,  # Residual connections
)
```

## Integration with Existing Code

```python
# Register in the model registry (the @register_* decorator seam)
from spectramr.models.registry import register_model


@register_model(name="configurable_vae", training_mode="vae")
class ConfigurableVAERegistered(ConfigurableVAE):
    pass
```

## YAML Configuration

```yaml
model:
  model_type: configurable_vae
  preset: resnet_attention
  latent_dim: 512
  beta: 0.5

  # Or custom config
  encoder_config:
    arch_type: resnet
    hidden_dims: [64, 128, 256, 512]
    use_attention: true
    attention_layers: [3]

  decoder_config:
    arch_type: resnet
    hidden_dims: [512, 256, 128, 64]
    use_attention: true
    attention_layers: [0]
```
