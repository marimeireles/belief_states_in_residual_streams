"""
Probe Transformer Activations for Belief Geometry

Extract activations from the residual stream and probe for:
1. Mess3 belief geometry (2-simplex)
2. Bloch Walk belief geometry (Bloch disk)

Usage:
    python probe_activations.py                              # Use default model
    python probe_activations.py --model path/to/model.pt     # Use specific model
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from transformers import LlamaForCausalLM, LlamaConfig
from composite_process import CompositeProcess

# Parse command line arguments
parser = argparse.ArgumentParser(description='Probe transformer activations for belief geometry')
parser.add_argument('--model', type=str,
                    default='/Users/3l3ktr4/dev/simplex/transformer_model.pt',
                    help='Path to trained model checkpoint (default: transformer_model.pt)')
args = parser.parse_args()

# Device setup
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

# Load composite process
composite = CompositeProcess(seed=42)

# Model config (must match training)
VOCAB_SIZE = 12
D_MODEL = 64  # Updated to match new training config
N_CTX = 8
N_LAYERS = 4  # Updated to match new training config

print("\n" + "=" * 60)
print("Loading Trained Model")
print("=" * 60)

# Load model
model_path = args.model
print(f"Model path: {model_path}")
checkpoint = torch.load(model_path, map_location=device, weights_only=False)

config = LlamaConfig(
    hidden_size=D_MODEL,
    intermediate_size=D_MODEL * 4,
    num_attention_heads=4,
    num_key_value_heads=4,
    num_hidden_layers=N_LAYERS,
    vocab_size=VOCAB_SIZE,
    max_position_embeddings=N_CTX,
    tie_word_embeddings=False,
    use_cache=False,
)

model = LlamaForCausalLM(config).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✓ Model loaded (best val loss: {checkpoint['best_val_loss']:.4f})")

"""## Extract Activations from Residual Stream"""

def get_all_sequences(max_length):
    """
    Generate all possible token sequences up to max_length.
    Returns sequences, mess3 beliefs, and bloch beliefs.
    """
    sequences = []
    mess3_beliefs_list = []
    bloch_beliefs_list = []

    # Start with empty sequence
    sequences.append([])
    mess3_beliefs_list.append(composite.mess3.stationary_dist)
    bloch_beliefs_list.append(composite.bloch.density_to_bloch(composite.bloch.initial_density))

    # Build up all sequences
    for length in range(1, max_length + 1):
        for prev_seq in [s for s in sequences if len(s) == length - 1]:
            for token in range(VOCAB_SIZE):
                new_seq = prev_seq + [token]

                # Compute belief states
                mess3_belief, bloch_belief = composite.compute_belief_state(new_seq)

                # Check if reachable (non-zero probability)
                prob = composite.compute_history_probability(new_seq)
                if prob > 1e-10:
                    sequences.append(new_seq)
                    mess3_beliefs_list.append(mess3_belief)
                    bloch_beliefs_list.append(bloch_belief)

    return sequences, np.array(mess3_beliefs_list), np.array(bloch_beliefs_list)

def compute_history_probability(composite, history):
    """Compute probability of a composite token history."""
    if len(history) == 0:
        return 1.0

    # Decode into mess3 and bloch histories
    mess3_history = []
    bloch_history = []
    for token in history:
        m, b = composite.decode_composite_token(token)
        mess3_history.append(m)
        bloch_history.append(b)

    # Compute probabilities independently
    mess3_prob = composite.mess3.compute_history_probability(mess3_history)
    bloch_prob = composite.bloch.compute_history_probability(bloch_history)

    return mess3_prob * bloch_prob

# Add this method to CompositeProcess if not there
composite.compute_history_probability = lambda history: compute_history_probability(composite, history)

print("\n" + "=" * 60)
print("Generating All Sequences")
print("=" * 60)

sequences, mess3_beliefs_gt, bloch_beliefs_gt = get_all_sequences(max_length=4)
print(f"Total sequences: {len(sequences)}")
print(f"Mess3 beliefs shape: {mess3_beliefs_gt.shape}")
print(f"Bloch beliefs shape: {bloch_beliefs_gt.shape}")

def extract_activations(sequences):
    """Extract residual stream activations for all sequences."""
    activations = []

    with torch.no_grad():
        for seq in sequences:
            if len(seq) == 0:
                # For empty sequence, use a dummy token and take position 0
                tokens = torch.tensor([[0]], dtype=torch.long, device=device)
                outputs = model(tokens, output_hidden_states=True)
                # Take first position embedding
                act = outputs.hidden_states[-1][0, 0].cpu().numpy()
            else:
                tokens = torch.tensor([seq], dtype=torch.long, device=device)
                outputs = model(tokens, output_hidden_states=True)
                # Take last position (last token in sequence)
                act = outputs.hidden_states[-1][0, -1].cpu().numpy()

            activations.append(act)

    return np.array(activations)

print("\n" + "=" * 60)
print("Extracting Activations")
print("=" * 60)

activations = extract_activations(sequences)
print(f"Activations shape: {activations.shape}")

"""## Train Probes"""

print("\n" + "=" * 60)
print("Training Probes (MLP)")
print("=" * 60)

# Train probe for Mess3 geometry (3D simplex → use first 2 dims since sum=1)
print("\nMess3 Probe (2-Simplex):")
print("  Training MLP probe...")
mess3_probe = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.2,
    random_state=42,
    verbose=False
)
mess3_probe.fit(activations, mess3_beliefs_gt[:, :2])  # First 2 dims (3rd is determined)

# Evaluate
mess3_pred = mess3_probe.predict(activations)
mess3_pred_full = np.column_stack([mess3_pred, 1 - mess3_pred.sum(axis=1)])  # Reconstruct 3rd dim
mess3_mse = np.mean((mess3_beliefs_gt - mess3_pred_full) ** 2)
mess3_rmse = np.sqrt(mess3_mse)
print(f"  RMSE: {mess3_rmse:.6f}")

# Train probe for Bloch Walk geometry (3D Bloch vector)
print("\nBloch Walk Probe (Bloch Disk):")
print("  Training MLP probe...")
bloch_probe = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.2,
    random_state=42,
    verbose=False
)
bloch_probe.fit(activations, bloch_beliefs_gt)

# Evaluate
bloch_pred = bloch_probe.predict(activations)
bloch_mse = np.mean((bloch_beliefs_gt - bloch_pred) ** 2)
bloch_rmse = np.sqrt(bloch_mse)
print(f"  RMSE: {bloch_rmse:.6f}")

"""## Visualization"""

print("\n" + "=" * 60)
print("Visualizing Results")
print("=" * 60)

fig = plt.figure(figsize=(16, 12))

# Row 1: Mess3 Geometry
# Ground Truth
ax1 = fig.add_subplot(2, 3, 1)
ax1.scatter(mess3_beliefs_gt[:, 0], mess3_beliefs_gt[:, 1],
            c=range(len(mess3_beliefs_gt)), cmap='viridis', alpha=0.6, s=20)
triangle = np.array([[1, 0], [0, 1], [0, 0], [1, 0]])
ax1.plot(triangle[:, 0], triangle[:, 1], 'r-', linewidth=2, label='Simplex boundary')
ax1.scatter([1, 0, 0], [0, 1, 0], c='red', s=200, marker='*', edgecolors='black', linewidths=2)
ax1.set_xlabel('p(state 0)', fontsize=11)
ax1.set_ylabel('p(state 1)', fontsize=11)
ax1.set_title('Mess3 Ground Truth\n(2-Simplex)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')
ax1.legend()

# Predicted
ax2 = fig.add_subplot(2, 3, 2)
ax2.scatter(mess3_pred_full[:, 0], mess3_pred_full[:, 1],
            c=range(len(mess3_pred_full)), cmap='viridis', alpha=0.6, s=20)
ax2.plot(triangle[:, 0], triangle[:, 1], 'r-', linewidth=2)
ax2.scatter([1, 0, 0], [0, 1, 0], c='red', s=200, marker='*', edgecolors='black', linewidths=2)
ax2.set_xlabel('p(state 0)', fontsize=11)
ax2.set_ylabel('p(state 1)', fontsize=11)
ax2.set_title(f'Mess3 from Activations\nRMSE={mess3_rmse:.4f}', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

# Overlay
ax3 = fig.add_subplot(2, 3, 3)
# Plot predicted FIRST (bottom layer)
ax3.scatter(mess3_pred_full[:, 0], mess3_pred_full[:, 1],
            c='red', alpha=0.3, s=20, label='Predicted', zorder=1)
# Plot ground truth ON TOP (top layer)
ax3.scatter(mess3_beliefs_gt[:, 0], mess3_beliefs_gt[:, 1],
            c='blue', alpha=0.8, s=15, label='Ground Truth', zorder=2)
ax3.plot(triangle[:, 0], triangle[:, 1], 'k-', linewidth=2)
ax3.set_xlabel('p(state 0)', fontsize=11)
ax3.set_ylabel('p(state 1)', fontsize=11)
ax3.set_title('Mess3 Overlay', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.set_aspect('equal')

# Row 2: Bloch Walk Geometry (x-z plane)
# Ground Truth
ax4 = fig.add_subplot(2, 3, 4)
ax4.scatter(bloch_beliefs_gt[:, 0], bloch_beliefs_gt[:, 2],
            c=range(len(bloch_beliefs_gt)), cmap='plasma', alpha=0.6, s=20)
theta = np.linspace(0, 2*np.pi, 100)
circle_x = np.cos(theta)
circle_z = np.sin(theta)
ax4.plot(circle_x, circle_z, 'r-', linewidth=2, label='Bloch boundary')
ax4.set_xlabel('b_x', fontsize=11)
ax4.set_ylabel('b_z', fontsize=11)
ax4.set_title('Bloch Walk Ground Truth\n(x-z Disk)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_aspect('equal')
ax4.legend()

# Predicted
ax5 = fig.add_subplot(2, 3, 5)
ax5.scatter(bloch_pred[:, 0], bloch_pred[:, 2],
            c=range(len(bloch_pred)), cmap='plasma', alpha=0.6, s=20)
ax5.plot(circle_x, circle_z, 'r-', linewidth=2)
ax5.set_xlabel('b_x', fontsize=11)
ax5.set_ylabel('b_z', fontsize=11)
ax5.set_title(f'Bloch Walk from Activations\nRMSE={bloch_rmse:.4f}', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.set_aspect('equal')

# Overlay
ax6 = fig.add_subplot(2, 3, 6)
# Plot predicted FIRST (bottom layer)
ax6.scatter(bloch_pred[:, 0], bloch_pred[:, 2],
            c='red', alpha=0.3, s=20, label='Predicted', zorder=1)
# Plot ground truth ON TOP (top layer)
ax6.scatter(bloch_beliefs_gt[:, 0], bloch_beliefs_gt[:, 2],
            c='blue', alpha=0.8, s=15, label='Ground Truth', zorder=2)
ax6.plot(circle_x, circle_z, 'k-', linewidth=2)
ax6.set_xlabel('b_x', fontsize=11)
ax6.set_ylabel('b_z', fontsize=11)
ax6.set_title('Bloch Walk Overlay', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.legend()
ax6.set_aspect('equal')

plt.tight_layout()
plt.savefig('/Users/3l3ktr4/dev/simplex/probe_results.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved to: probe_results.png")
plt.show()

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"Mess3 RMSE: {mess3_rmse:.6f}")
print(f"Bloch RMSE: {bloch_rmse:.6f}")
print(f"\nInterpretation:")
print(f"  - Lower RMSE = better match to ground truth geometry")
print(f"  - Check visualizations to see if geometries are preserved")
print("=" * 60)
