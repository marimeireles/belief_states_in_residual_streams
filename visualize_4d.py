"""
Visualize 4D Composite Belief Geometry
Using pairs plot to show all 2D projections
"""

import numpy as np
import matplotlib.pyplot as plt
from composite_process import CompositeProcess

print("=" * 60)
print("Generating 4D Composite Belief Geometry")
print("=" * 60)

# Create composite process
composite = CompositeProcess(seed=42)

# Get all belief states
print("\nComputing belief geometry (max_length=5)...")
histories, mess3_beliefs, bloch_beliefs = composite.get_belief_geometry(max_length=5)
print(f"Total sequences: {len(histories)}")

# Construct 4D joint belief space
# Dim 0: Mess3 p(state 0)
# Dim 1: Mess3 p(state 1)
# Dim 2: Bloch b_x
# Dim 3: Bloch b_z
joint_4d = np.column_stack([
    mess3_beliefs[:, 0],  # Mess3 dim 0
    mess3_beliefs[:, 1],  # Mess3 dim 1
    bloch_beliefs[:, 0],  # Bloch b_x
    bloch_beliefs[:, 2],  # Bloch b_z
])

dim_names = ['Mess3\np(s0)', 'Mess3\np(s1)', 'Bloch\nb_x', 'Bloch\nb_z']

print(f"Joint 4D shape: {joint_4d.shape}")

# --- add these helpers after you build joint_4d and dim_names ---

def _bounds_for(name):
    # Mess3 ∈ [0,1], Bloch components ∈ [-1, 1]
    return (0.0, 1.0) if "Mess3" in name else (-1.0, 1.0)

def plot_1d_barcode(ax, x, name):
    """1D colored rug: same colormap as your 2-D panels."""
    n = len(x)
    ax.scatter(x, np.zeros(n), c=np.arange(n), cmap='viridis', s=6, alpha=0.7)
    lo, hi = _bounds_for(name)
    ax.set_xlim(lo, hi)
    ax.set_yticks([])
    ax.set_xlabel(name, fontsize=10, fontweight='bold')
    # draw bounds as faint guides
    ax.vlines([lo, hi], 0, 1, colors='r', alpha=0.25, linewidth=1)
    ax.set_title("1D barcode (samples)", fontsize=9)
    ax.grid(True, axis='x', alpha=0.2)

def plot_1d_heatline(ax, x, name, bins=256):
    """1D density as an image (heatmap collapsed to a line)."""
    lo, hi = _bounds_for(name)
    hist, edges = np.histogram(x, bins=bins, range=(lo, hi))
    heat = hist[np.newaxis, :]  # shape (1, bins)
    ax.imshow(
        heat, aspect='auto', origin='lower', cmap='viridis',
        extent=[edges[0], edges[-1], 0, 1]
    )
    ax.set_yticks([])
    ax.set_xlim(lo, hi)
    ax.set_xlabel(name, fontsize=10, fontweight='bold')
    ax.vlines([lo, hi], 0, 1, colors='r', alpha=0.25, linewidth=1)
    ax.set_title("1D heatline (density)", fontsize=9)
    ax.grid(False)

def summarize_support(x, name, k=12, round_decimals=6):
    """Optional: print the most frequent exact values (reveals discrete support)."""
    vals, cnt = np.unique(np.round(x, round_decimals), return_counts=True)
    order = np.argsort(-cnt)
    print(f"\n{name} — top {min(k, len(vals))} values (rounded {round_decimals}dp):")
    for v, c in zip(vals[order][:k], cnt[order][:k]):
        print(f"  {v:.{round_decimals}f}  × {c}")

# --- create the one-dimensional figure ---

fig1d, axes1d = plt.subplots(4, 2, figsize=(12, 8), constrained_layout=True)
for d, name in enumerate(dim_names):
    x = joint_4d[:, d]
    plot_1d_barcode(axes1d[d, 0], x, name)
    plot_1d_heatline(axes1d[d, 1], x, name, bins=256)
    # Optional: print the discrete support summary to the console
    summarize_support(x, name, k=12, round_decimals=6)

fig1d.suptitle("4D Composite Belief — One-Dimensional Views", fontsize=14, fontweight='bold')
plt.savefig('/Users/3l3ktr4/dev/simplex/belief_4d_1d_views.png', dpi=150, bbox_inches='tight')
print("\n✓ 1D views saved to: belief_4d_1d_views.png")


# Create pairs plot
fig, axes = plt.subplots(4, 4, figsize=(16, 16))

for i in range(4):
    for j in range(4):
        ax = axes[i, j]

        if i == j:
            # Diagonal: histogram
            ax.hist(joint_4d[:, i], bins=30, color='steelblue', alpha=0.7)
            ax.set_ylabel('Count', fontsize=9)
            if i == 3:
                ax.set_xlabel(dim_names[i], fontsize=10, fontweight='bold')
        else:
            # Off-diagonal: scatter plot
            ax.scatter(joint_4d[:, j], joint_4d[:, i],
                      c=range(len(joint_4d)), cmap='viridis',
                      alpha=0.5, s=5)

            # Add boundaries for Mess3 (simplex)
            if i < 2 and j < 2:
                triangle = np.array([[1, 0], [0, 1], [0, 0], [1, 0]])
                if j == 0:  # x-axis is p(s0)
                    ax.plot(triangle[:, 0], triangle[:, 1], 'r-', linewidth=1, alpha=0.3)
                else:  # x-axis is p(s1)
                    ax.plot(triangle[:, 1], triangle[:, 0], 'r-', linewidth=1, alpha=0.3)

            # Add boundary for Bloch (circle)
            if i >= 2 and j >= 2:
                theta = np.linspace(0, 2*np.pi, 100)
                circle_x = np.cos(theta)
                circle_z = np.sin(theta)
                ax.plot(circle_x, circle_z, 'r-', linewidth=1, alpha=0.3)
                ax.set_aspect('equal')

        # Show labels on all plots for clarity
        ax.set_ylabel(dim_names[i], fontsize=8)
        ax.set_xlabel(dim_names[j], fontsize=8)

        # Show all tick labels
        ax.tick_params(axis='both', labelsize=7)
        ax.grid(True, alpha=0.2)

plt.suptitle('4D Composite Belief Geometry - Pairs Plot\n(All 2D Projections)',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('/Users/3l3ktr4/dev/simplex/belief_4d_pairs.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Pairs plot saved to: belief_4d_pairs.png")
plt.show()

print("\n" + "=" * 60)
print("Interpretation Guide:")
print("=" * 60)
print("Diagonal: Distribution of each dimension")
print("Off-diagonal: All pairwise relationships")
print("\nKey observations to look for:")
print("  - Top-left 2x2: Mess3 geometry (should show simplex)")
print("  - Bottom-right 2x2: Bloch geometry (should show disk)")
print("  - Top-right & bottom-left: Cross-correlations")
print("    → If sparse/uncorrelated: dimensions are SEPARABLE")
print("    → If structured: dimensions are ENTANGLED")
print("=" * 60)
