#!/bin/bash
# Example training script with different configurations

echo "Vision Token Compression - Training Examples"
echo "=============================================="

# Example 1: Quick test with subset
echo -e "\n1. Quick test with data subset (for testing setup)"
python train.py \
    data.use_subset=true \
    data.subset_size=1000 \
    training.epochs=5 \
    training.batch_size=16 \
    experiment.name="quick_test"

# Example 2: Standard training with 6x6 compression
echo -e "\n2. Standard training (6x6 compression)"
python train.py \
    model.compressor.output_grid_size=6 \
    training.epochs=100 \
    training.batch_size=32 \
    experiment.name="compression_6x6"

# Example 3: Higher compression (4x4 grid)
echo -e "\n3. Higher compression (4x4 grid)"
python train.py \
    model.compressor.output_grid_size=4 \
    training.epochs=100 \
    training.batch_size=32 \
    experiment.name="compression_4x4"

# Example 4: Lower compression (8x8 grid)
echo -e "\n4. Lower compression (8x8 grid)"
python train.py \
    model.compressor.output_grid_size=8 \
    training.epochs=100 \
    training.batch_size=32 \
    experiment.name="compression_8x8"

# Example 5: Custom loss weights
echo -e "\n5. Custom loss weights (higher AE weight)"
python train.py \
    loss.weights.ae=2.0 \
    loss.weights.wgan=1.0 \
    experiment.name="high_ae_weight"

# Example 6: Attention-based decoder
echo -e "\n6. Attention-based decoder"
python train.py \
    model.autoencoder.use_attention=true \
    experiment.name="attention_decoder"

# Example 7: Different CUDA device
echo -e "\n7. Train on CUDA device 1"
python train.py \
    hardware.cuda_device=1 \
    experiment.name="cuda_device_1"

echo -e "\n=============================================="
echo "Examples completed!"
