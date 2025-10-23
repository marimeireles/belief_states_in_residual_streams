"""
Train Transformer on Composite Process

Train a small transformer on the Mess3 ⊗ Bloch Walk composite process,
then probe its residual stream to see if it discovers the belief geometry.

Experiment parameters (from paper):
- Small model: 2 layers, ~100k parameters
- Context length: 6-8 tokens
- Training: ~5000 steps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime
from transformers import LlamaConfig, LlamaForCausalLM
from composite_process import CompositeProcess

# Device setup
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

"""## Hyperparameters"""

# Data generation
VOCAB_SIZE = 12  # Composite tokens: 3 (Mess3) × 4 (Bloch Walk)
N_CTX = 8  # Context length (sequence length)

# Model architecture
D_MODEL = 64  # Increased from 64 for more capacity
N_LAYERS = 4  # Increased from 2 for deeper model
N_HEADS = 4  # Number of attention heads

# Training
BATCH_SIZE = 64
N_STEPS = 15000
VAL_SIZE = 200
LOG_EVERY_STEPS = 100

# Auxiliary loss for explicit belief prediction
USE_AUXILIARY_LOSS = True  # Set to True to add belief prediction heads
ALPHA = 0.1  # Weight for Mess3 belief loss
BETA = 0.1   # Weight for Bloch belief loss

print("\n" + "=" * 60)
print("Experiment Configuration")
print("=" * 60)
print(f"Composite Process: Mess3 ⊗ Bloch Walk")
print(f"  Vocabulary size: {VOCAB_SIZE}")
print(f"  Context length: {N_CTX}")
print(f"\nModel Architecture:")
print(f"  Hidden dimension: {D_MODEL}")
print(f"  Number of layers: {N_LAYERS}")
print(f"  Attention heads: {N_HEADS}")
print(f"\nTraining:")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Training steps: {N_STEPS}")
print(f"  Device: {device}")
print(f"\nAuxiliary Loss:")
print(f"  Enabled: {USE_AUXILIARY_LOSS}")
if USE_AUXILIARY_LOSS:
    print(f"  Mess3 weight (α): {ALPHA}")
    print(f"  Bloch weight (β): {BETA}")

"""## Data Generation (Deterministic)"""

# Create composite process
composite_process = CompositeProcess(seed=SEED)

def generate_all_sequences(max_length, n_samples_per_length=5000, pad_token=-1):
    """
    Generate a fixed dataset of sequences with VARIED LENGTHS (1 to max_length).
    This gives richer training data showing how beliefs evolve over time.

    Uses pad_token=-1 (outside vocab 0-11) to distinguish from real tokens.
    """
    print("\nGenerating complete deterministic dataset...")
    all_sequences = []

    # Generate sequences of varied lengths
    for length in range(1, max_length + 1):
        for seed_offset in range(n_samples_per_length):
            np.random.seed(SEED + seed_offset + length * 10000)
            random.seed(SEED + seed_offset + length * 10000)
            tokens, _ = composite_process.generate_sequence(length)

            # Pad to max_length with special pad_token
            padded = np.pad(tokens, (0, max_length - length), constant_values=pad_token)
            all_sequences.append(padded)

    print(f"Generated {len(all_sequences)} varied-length sequences (lengths 1-{max_length})")
    return np.array(all_sequences)

def get_mixed_batch(dataset, batch_size, composite_process, seq_len, pad_token=-1):
    """
    HYBRID APPROACH: Mix fixed dataset samples with fresh random generation.

    - 25% from fixed dataset (comprehensive coverage, varied lengths)
    - 75% freshly generated (diversity, prevents memorization, full length)

    Properly masks padding with -100 in labels.
    """
    quarter_batch = batch_size // 4

    # 25% from fixed dataset (varied lengths with padding)
    fixed_indices = np.random.choice(len(dataset), size=quarter_batch, replace=False)
    fixed_samples = dataset[fixed_indices]

    # 75% freshly generated (full length, no padding)
    fresh_samples = []
    for _ in range(batch_size - quarter_batch):
        tokens, _ = composite_process.generate_sequence(seq_len)
        fresh_samples.append(tokens)
    fresh_samples = np.array(fresh_samples)

    # Combine
    batch = np.concatenate([fixed_samples, fresh_samples], axis=0)

    # Create mask for padding positions
    padding_mask = (batch == pad_token)

    # Input tokens: replace pad_token with 0 for valid embedding lookup
    batch_clean = np.where(batch == pad_token, 0, batch)
    tokens = torch.tensor(batch_clean, dtype=torch.long, device=device)

    # Labels: mask padding positions with -100 (ignored in loss)
    labels = tokens.clone()
    labels[torch.tensor(padding_mask, device=device)] = -100

    return tokens, labels

print("\n" + "=" * 60)
print("Generating Dataset")
print("=" * 60)

# Generate complete deterministic dataset (2x data for better coverage)
train_dataset = generate_all_sequences(max_length=N_CTX, n_samples_per_length=10000)
print(f"Dataset shape: {train_dataset.shape}")

# Test batch retrieval
test_tokens, test_labels = get_mixed_batch(train_dataset, batch_size=2, composite_process=composite_process, seq_len=N_CTX)
print(f"\nExample batch shapes:")
print(f"  Tokens: {test_tokens.shape}")
print(f"  Labels: {test_labels.shape}")
print(f"Example sequences (varied lengths, padded to {N_CTX}):")
for i in range(2):
    seq = test_tokens[i].cpu().numpy()
    lab = test_labels[i].cpu().numpy()
    # Count non-padding tokens
    real_length = np.sum(lab != -100)
    print(f"  Sequence {i+1}: {seq}")
    print(f"    Real length: {real_length}, Labels: {lab}")

"""## Model Setup"""

config = LlamaConfig(
    hidden_size=D_MODEL,
    intermediate_size=D_MODEL * 4,  # Standard MLP expansion
    num_attention_heads=N_HEADS,
    num_key_value_heads=N_HEADS,
    num_hidden_layers=N_LAYERS,
    vocab_size=VOCAB_SIZE,
    max_position_embeddings=N_CTX,
    # No dropout - model is small enough that we need full capacity
    # Disable some features we don't need
    tie_word_embeddings=False,
    use_cache=False,
)

model = LlamaForCausalLM(config).to(device)

# Auxiliary prediction heads (only used if USE_AUXILIARY_LOSS=True)
if USE_AUXILIARY_LOSS:
    # Mess3 belief predictor: D_MODEL -> 2 (predict first 2 simplex dims, 3rd is 1-sum)
    mess3_head = nn.Sequential(
        nn.Linear(D_MODEL, 64),
        nn.ReLU(),
        nn.Linear(64, 2)
    ).to(device)

    # Bloch belief predictor: D_MODEL -> 3 (Bloch vector)
    bloch_head = nn.Sequential(
        nn.Linear(D_MODEL, 64),
        nn.ReLU(),
        nn.Linear(64, 3)
    ).to(device)

    print("\n✓ Auxiliary belief prediction heads created")
else:
    mess3_head = None
    bloch_head = None

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

if USE_AUXILIARY_LOSS:
    aux_params = sum(p.numel() for p in mess3_head.parameters()) + sum(p.numel() for p in bloch_head.parameters())
    total_params += aux_params
    trainable_params += aux_params

print("\n" + "=" * 60)
print("Model Architecture")
print("=" * 60)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"\nTarget: ~100k parameters (we have {total_params/1000:.1f}k)")

"""## Training"""

model.train()

# Optimizer includes auxiliary heads if enabled
if USE_AUXILIARY_LOSS:
    all_params = list(model.parameters()) + list(mess3_head.parameters()) + list(bloch_head.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=1e-3, weight_decay=0.01)
    mess3_head.train()
    bloch_head.train()
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=200
)


def compute_belief_targets(tokens_batch, composite_process):
    """
    Compute ground truth belief states for a batch of token sequences.

    Returns:
        mess3_targets: (batch, seq_len, 2) - first 2 simplex dims
        bloch_targets: (batch, seq_len, 3) - Bloch vector
        mask: (batch, seq_len) - True for real tokens, False for padding
    """
    batch_size, seq_len = tokens_batch.shape
    mess3_targets = []
    bloch_targets = []
    masks = []

    for seq in tokens_batch.cpu().numpy():
        seq_mess3 = []
        seq_bloch = []
        seq_mask = []

        for pos in range(seq_len):
            # Get history up to this position
            history = seq[:pos+1].tolist()

            # Check if this is padding (token 0 at padded position)
            # We detect padding if all remaining tokens are 0
            if pos < seq_len - 1 and all(t == 0 for t in seq[pos+1:]):
                # This might be padding, check if it's actually token 0 or padding
                # For simplicity, we'll compute beliefs for all positions and rely on loss masking
                pass

            # Compute beliefs
            mess3_belief, bloch_belief = composite_process.compute_belief_state(history)
            seq_mess3.append(mess3_belief[:2])  # Only first 2 dims (3rd is 1-sum)
            seq_bloch.append(bloch_belief)
            seq_mask.append(True)  # Will be properly masked by labels later

        mess3_targets.append(seq_mess3)
        bloch_targets.append(seq_bloch)
        masks.append(seq_mask)

    return (
        torch.tensor(np.array(mess3_targets), dtype=torch.float32, device=device),
        torch.tensor(np.array(bloch_targets), dtype=torch.float32, device=device),
        torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)  # Simplified mask
    )

# Track best model
best_model_state_dict = model.state_dict()
best_val_loss = float("inf")

# Create validation set from dataset (use last VAL_SIZE sequences)
print("\n" + "=" * 60)
print("Creating Validation Set")
print("=" * 60)
val_data = train_dataset[-VAL_SIZE:]
pad_token = -1

# Create mask for padding positions
val_padding_mask = (val_data == pad_token)

# Input: replace pad_token with 0 for embedding lookup
val_data_clean = np.where(val_data == pad_token, 0, val_data)
val_tokens = torch.tensor(val_data_clean, dtype=torch.long, device=device)

# Labels: mask padding with -100
val_labels = val_tokens.clone()
val_labels[torch.tensor(val_padding_mask, device=device)] = -100

print(f"Validation set size: {VAL_SIZE} sequences (varied lengths)")
print(f"Training set size: {len(train_dataset) - VAL_SIZE} sequences")

# Training loop
print("\n" + "=" * 60)
print("Training")
print("=" * 60)

train_losses = []
val_losses = []
steps = []

for step in range(N_STEPS):
    # Get MIXED batch: 25% fixed dataset, 75% fresh generation (with proper padding mask)
    tokens, labels = get_mixed_batch(train_dataset[:-VAL_SIZE], BATCH_SIZE, composite_process, N_CTX)

    # Forward pass (labels have -100 for padding, which is ignored in loss)
    outputs = model(tokens, labels=labels, return_dict=True, output_hidden_states=USE_AUXILIARY_LOSS)
    loss = outputs.loss

    # Add auxiliary losses if enabled
    if USE_AUXILIARY_LOSS:
        # Extract last layer activations
        hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, D_MODEL)

        # Predict beliefs from activations
        mess3_pred = mess3_head(hidden_states)  # (batch, seq_len, 2)
        bloch_pred = bloch_head(hidden_states)  # (batch, seq_len, 3)

        # Compute ground truth beliefs
        mess3_target, bloch_target, mask = compute_belief_targets(tokens, composite_process)

        # Mask out padding positions (where labels == -100)
        non_padding_mask = (labels != -100)

        # Compute MSE loss only on non-padded positions
        mess3_loss = F.mse_loss(
            mess3_pred[non_padding_mask],
            mess3_target[non_padding_mask]
        )
        bloch_loss = F.mse_loss(
            bloch_pred[non_padding_mask],
            bloch_target[non_padding_mask]
        )

        # Combine losses
        loss = loss + ALPHA * mess3_loss + BETA * bloch_loss

    # Backward pass
    loss.backward()
    if USE_AUXILIARY_LOSS:
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(mess3_head.parameters()) + list(bloch_head.parameters()), 1.0)
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()

    # Logging
    if step % LOG_EVERY_STEPS == 0 or step == N_STEPS - 1:
        with torch.no_grad():
            val_outputs = model(val_tokens, labels=val_labels, return_dict=True)
            val_loss = val_outputs.loss

            # Track best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            # Update learning rate based on validation loss
            scheduler.step(val_loss)

            # Log
            if USE_AUXILIARY_LOSS:
                print(f"Step {step:4d}: train_loss={loss.item():.4f}  val_loss={val_loss.item():.4f}  best_val={best_val_loss.item():.4f}  [mess3_aux={mess3_loss.item():.4f} bloch_aux={bloch_loss.item():.4f}]")
            else:
                print(f"Step {step:4d}: train_loss={loss.item():.4f}  val_loss={val_loss.item():.4f}  best_val={best_val_loss.item():.4f}")

            # Track for plotting
            train_losses.append(loss.item())
            val_losses.append(val_loss.item())
            steps.append(step)

print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)
print(f"Best validation loss: {best_val_loss.item():.4f}")

# Load best model
model.load_state_dict({k: v.to(device) for k, v in best_model_state_dict.items()})
model.eval()

# Save model with timestamp and val loss in filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
val_loss_str = f"{best_val_loss.item():.4f}".replace('.', '_')
model_filename = f'transformer_model_{timestamp}_val{val_loss_str}.pt'
model_path = f'/Users/3l3ktr4/dev/simplex/{model_filename}'

save_dict = {
    'model_state_dict': best_model_state_dict,
    'config': config,
    'best_val_loss': best_val_loss,
    'timestamp': timestamp,
    'use_auxiliary_loss': USE_AUXILIARY_LOSS,
}

if USE_AUXILIARY_LOSS:
    save_dict['mess3_head_state_dict'] = mess3_head.state_dict()
    save_dict['bloch_head_state_dict'] = bloch_head.state_dict()

torch.save(save_dict, model_path)
print(f"\n✓ Model saved to: {model_path}")

# Also save as "latest" for easy probing
latest_path = '/Users/3l3ktr4/dev/simplex/transformer_model.pt'
torch.save(save_dict, latest_path)
print(f"✓ Also saved as: {latest_path} (for probing)")

"""## Training Visualization"""

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.plot(steps, train_losses, 'b-', label='Train Loss', alpha=0.7)
ax.plot(steps, val_losses, 'r-', label='Val Loss', alpha=0.7)
ax.axhline(y=best_val_loss.item(), color='r', linestyle='--', alpha=0.5, label=f'Best Val ({best_val_loss.item():.4f})')

ax.set_xlabel('Training Step', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Transformer Training on Composite Process', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
curve_filename = f'training_curve_{timestamp}_val{val_loss_str}.png'
curve_path = f'/Users/3l3ktr4/dev/simplex/{curve_filename}'
plt.savefig(curve_path, dpi=150, bbox_inches='tight')
print(f"✓ Training curve saved to: {curve_filename}")
plt.show()

"""## Test Predictions"""

print("\n" + "=" * 60)
print("Testing Next-Token Prediction")
print("=" * 60)

# Get a few test sequences from validation set
test_tokens = val_tokens[:5]

with torch.no_grad():
    logits = model(test_tokens).logits

# Show predictions vs ground truth
print("\nExample predictions (showing last token):")
for i in range(min(5, test_tokens.shape[0])):
    # Get sequence
    seq = test_tokens[i].cpu().numpy()

    # Decode
    decoded = [composite_process.decode_composite_token(t) for t in seq]

    # Get model's prediction for last token
    pred_logits = logits[i, -2, :]  # Predict position -1 from position -2
    pred_token = pred_logits.argmax().item()
    true_token = seq[-1]

    # Decode prediction and true
    pred_m, pred_b = composite_process.decode_composite_token(pred_token)
    true_m, true_b = composite_process.decode_composite_token(true_token)

    pred_str = f"({composite_process.mess3_token_names[pred_m]},{pred_b})"
    true_str = f"({composite_process.mess3_token_names[true_m]},{true_b})"

    match = "✓" if pred_token == true_token else "✗"

    print(f"\n  Sequence {i+1}:")
    print(f"    Context:    {seq[:-1]}")
    print(f"    Predicted:  {pred_token:2d} {pred_str}  {match}")
    print(f"    Ground truth: {true_token:2d} {true_str}")
    print(f"    Confidence: {F.softmax(pred_logits, dim=0)[pred_token].item():.3f}")

print("\n" + "=" * 60)
print("Ready for Probing!")
print("=" * 60)
print("\nNext steps:")
print("  1. Extract activations from residual stream")
print("  2. Train probes to predict Mess3 and Bloch beliefs")
print("  3. Visualize discovered geometry vs ground truth")
print("  4. Analyze if geometry is separable or entangled")
print("=" * 60)
