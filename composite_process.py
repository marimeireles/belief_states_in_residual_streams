"""
Composite Process Generator

Combines Mess3 (classical HMM) and Bloch Walk (quantum process)
via independent generation and Cartesian product of token vocabularies.

"""

import numpy as np
from typing import Tuple, List
from mess3_process import Mess3Process
from bloch_walk_process import BlochWalkProcess


class CompositeProcess:
    """
    Composite process: Mess3 ⊗ Bloch Walk

    Token vocabulary: {0, 1, ..., 11} representing all pairs (m, b) where:
    - m ∈ {0, 1, 2} (Mess3 token)
    - b ∈ {0, 1, 2, 3} (Bloch Walk token)

    Encoding: composite_token = m * 4 + b

    Belief state: (mess3_belief, bloch_belief)
    - mess3_belief: 3D vector (point in 2-simplex)
    - bloch_belief: 3D vector (point in Bloch ball)
    - Joint: 6D vector with 4 degrees of freedom (2 + 2)
    """

    def __init__(self, seed: int = 42):
        """
        Initialize composite process.

        Args:
            seed: Random seed for reproducibility
        """
        # Create both subprocesses with same seed for reproducibility
        self.mess3 = Mess3Process(seed=seed)
        self.bloch = BlochWalkProcess(seed=seed)

        # Token vocabulary
        self.n_mess3_tokens = 3
        self.n_bloch_tokens = 4
        self.n_composite_tokens = self.n_mess3_tokens * self.n_bloch_tokens  # 12

        # For interpretability
        self.mess3_token_names = ['a', 'b', 'c']
        self.bloch_token_names = ['0', '1', '2', '3']

    def encode_composite_token(self, mess3_token: int, bloch_token: int) -> int:
        """
        Encode pair (mess3_token, bloch_token) as single integer.

        Args:
            mess3_token: Token from Mess3 ∈ {0, 1, 2}
            bloch_token: Token from Bloch Walk ∈ {0, 1, 2, 3}

        Returns:
            composite_token: Integer ∈ {0, 1, ..., 11}
        """
        return mess3_token * self.n_bloch_tokens + bloch_token

    def decode_composite_token(self, composite_token: int) -> Tuple[int, int]:
        """
        Decode composite token back to (mess3_token, bloch_token) pair.

        Args:
            composite_token: Integer ∈ {0, 1, ..., 11}

        Returns:
            (mess3_token, bloch_token): The original pair
        """
        mess3_token = composite_token // self.n_bloch_tokens
        bloch_token = composite_token % self.n_bloch_tokens
        return mess3_token, bloch_token

    def generate_sequence(self, length: int) -> Tuple[np.ndarray, Tuple[List, List]]:
        """
        Generate sequence from composite process via INDEPENDENT generation.

        The key insight: Mess3 and Bloch Walk run in parallel without
        influencing each other. This creates a separable belief geometry.

        Args:
            length: Number of tokens to generate

        Returns:
            composite_tokens: Array of shape (length,) with tokens ∈ {0, ..., 11}
            states: Tuple of (mess3_states, bloch_states)
                mess3_states: List of length 'length' with Mess3 hidden states
                bloch_states: List of length 'length' with Bloch density matrices
        """
        # Generate from Mess3 (classical HMM)
        mess3_tokens, mess3_states = self.mess3.generate_sequence(length)

        # Generate from Bloch Walk (quantum process)
        bloch_tokens, bloch_states = self.bloch.generate_sequence(length)

        # Combine into composite tokens
        composite_tokens = np.array([
            self.encode_composite_token(m, b)
            for m, b in zip(mess3_tokens, bloch_tokens)
        ])

        return composite_tokens, (mess3_states, bloch_states)

    def compute_belief_state(self, composite_history: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute belief state given composite token history.

        Since processes are independent, we can compute beliefs separately:
        - Decode composite tokens into (mess3, bloch) pairs
        - Run Mess3 Bayesian filtering on mess3 tokens
        - Run Bloch Walk quantum filtering on bloch tokens

        Args:
            composite_history: List of composite tokens ∈ {0, ..., 11}

        Returns:
            mess3_belief: 3D vector (point in 2-simplex)
            bloch_belief: 3D Bloch vector (point in Bloch ball)
        """
        # Decode composite tokens
        mess3_history = []
        bloch_history = []

        for composite_token in composite_history:
            m, b = self.decode_composite_token(composite_token)
            mess3_history.append(m)
            bloch_history.append(b)

        # Compute beliefs independently
        mess3_belief = self.mess3.compute_belief_state(mess3_history)
        bloch_belief = self.bloch.compute_belief_state(bloch_history)

        return mess3_belief, bloch_belief

    def get_belief_geometry(self, max_length: int = 6) -> Tuple[List[List[int]], np.ndarray, np.ndarray]:
        """
        Compute belief states for all possible composite histories.

        This gives us the "ground truth" 4D belief geometry:
        - Mess3 contributes 2D (simplex)
        - Bloch Walk contributes 2D (Bloch disk)
        - Joint is 4D (Cartesian product)

        Args:
            max_length: Maximum length of token histories to consider

        Returns:
            histories: List of composite token histories
            mess3_beliefs: Array of shape (n_histories, 3) - Mess3 simplex points
            bloch_beliefs: Array of shape (n_histories, 3) - Bloch ball points
        """
        # We could generate all 12^max_length histories, but that's huge!
        # Instead, we'll generate by combining Mess3 and Bloch histories

        print(f"Computing ground truth belief geometry (max_length={max_length})...")

        # Get belief geometries for each subprocess
        print("  Getting Mess3 belief geometry...")
        mess3_histories, mess3_beliefs = self.mess3.get_belief_geometry(max_length)
        print(f"    Found {len(mess3_histories)} Mess3 histories")

        print("  Getting Bloch Walk belief geometry...")
        bloch_histories, bloch_beliefs = self.bloch.get_belief_geometry(max_length)
        print(f"    Found {len(bloch_histories)} Bloch histories")

        # Combine them (Cartesian product of histories)
        print("  Computing Cartesian product...")
        composite_histories = []
        composite_mess3_beliefs = []
        composite_bloch_beliefs = []

        for mess3_hist, mess3_belief in zip(mess3_histories, mess3_beliefs):
            for bloch_hist, bloch_belief in zip(bloch_histories, bloch_beliefs):
                # Combine histories (interleaved by time)
                # Make sure they're the same length
                if len(mess3_hist) == len(bloch_hist):
                    composite_hist = [
                        self.encode_composite_token(m, b)
                        for m, b in zip(mess3_hist, bloch_hist)
                    ]
                    composite_histories.append(composite_hist)
                    composite_mess3_beliefs.append(mess3_belief)
                    composite_bloch_beliefs.append(bloch_belief)

        print(f"  Total composite histories: {len(composite_histories)}")
        print(f"  (Product: {len(mess3_histories)} × {len(bloch_histories)} would be {len(mess3_histories) * len(bloch_histories)},")
        print(f"   but we only keep same-length pairs)")

        return (composite_histories,
                np.array(composite_mess3_beliefs),
                np.array(composite_bloch_beliefs))


# Test the implementation
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    print("=" * 60)
    print("Testing Composite Process (Mess3 ⊗ Bloch Walk)")
    print("=" * 60)

    # Create composite process
    composite = CompositeProcess()

    print(f"\nToken vocabulary:")
    print(f"  Mess3 tokens: {composite.n_mess3_tokens} (a, b, c)")
    print(f"  Bloch tokens: {composite.n_bloch_tokens} (0, 1, 2, 3)")
    print(f"  Composite tokens: {composite.n_composite_tokens}")
    print(f"\nEncoding examples:")
    for m in range(3):
        for b in range(4):
            c = composite.encode_composite_token(m, b)
            m_name = composite.mess3_token_names[m]
            b_name = composite.bloch_token_names[b]
            print(f"  ({m_name}, {b_name}) → {c:2d}")

    # Generate sample sequence
    print(f"\nGenerating sample sequence of length 20:")
    tokens, (mess3_states, bloch_states) = composite.generate_sequence(20)

    print(f"  Composite tokens: {tokens}")

    # Decode to show structure
    print(f"\n  Decoded as (Mess3, Bloch) pairs:")
    decoded = [composite.decode_composite_token(t) for t in tokens]
    mess3_seq = ''.join([composite.mess3_token_names[m] for m, b in decoded])
    bloch_seq = ''.join([composite.bloch_token_names[b] for m, b in decoded])
    print(f"    Mess3:      {mess3_seq}")
    print(f"    Bloch Walk: {bloch_seq}")

    # Test belief computation
    print(f"\nBelief state examples:")
    print(f"  (Showing independent belief updates)")

    test_histories = [
        [],
        [0],  # (a, 0)
        [0, 5],  # (a, 0), (b, 1)
    ]

    for comp_hist in test_histories:
        mess3_belief, bloch_belief = composite.compute_belief_state(comp_hist)

        # Decode for display
        if comp_hist:
            decoded = [composite.decode_composite_token(t) for t in comp_hist]
            hist_str = ' '.join([f"({composite.mess3_token_names[m]},{b})" for m, b in decoded])
        else:
            hist_str = "(empty)"

        print(f"\n  After {hist_str}:")
        print(f"    Mess3 belief:  {mess3_belief} (simplex point)")
        print(f"    Bloch belief:  {bloch_belief} (Bloch vector)")

    # Get full belief geometry (use smaller max_length since it's Cartesian product!)
    print(f"\n" + "=" * 60)
    print("Computing full belief geometry...")
    print("=" * 60)

    histories, mess3_beliefs, bloch_beliefs = composite.get_belief_geometry(max_length=4)

    print(f"\nBelief geometry shape:")
    print(f"  Mess3 beliefs: {mess3_beliefs.shape} (points in 2-simplex)")
    print(f"  Bloch beliefs: {bloch_beliefs.shape} (points in Bloch ball)")
    print(f"  Joint dimension: {mess3_beliefs.shape[1] + bloch_beliefs.shape[1]}D")
    print(f"  Degrees of freedom: 2 (Mess3) + 2 (Bloch) = 4")

    # Visualize the composite geometry
    print(f"\n" + "=" * 60)
    print("Visualizing Composite Belief Geometry")
    print("=" * 60)

    fig = plt.figure(figsize=(16, 7))

    # Plot 1: Mess3 simplex (2D view)
    ax1 = fig.add_subplot(121)
    scatter1 = ax1.scatter(mess3_beliefs[:, 0], mess3_beliefs[:, 1],
                           c=range(len(mess3_beliefs)), cmap='viridis',
                           alpha=0.6, s=20)

    # Draw simplex boundary
    triangle = np.array([[1, 0], [0, 1], [0, 0], [1, 0]])
    ax1.plot(triangle[:, 0], triangle[:, 1], 'r-', linewidth=2, label='Simplex boundary')

    ax1.set_xlabel('p(state 0)', fontsize=12)
    ax1.set_ylabel('p(state 1)', fontsize=12)
    ax1.set_title('Mess3 Component\n(2-Simplex)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_aspect('equal')

    # Plot 2: Bloch Walk disk (x-z plane)
    ax2 = fig.add_subplot(122)
    scatter2 = ax2.scatter(bloch_beliefs[:, 0], bloch_beliefs[:, 2],
                           c=range(len(bloch_beliefs)), cmap='plasma',
                           alpha=0.6, s=20)

    # Draw unit circle (boundary)
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = np.cos(theta)
    circle_z = np.sin(theta)
    ax2.plot(circle_x, circle_z, 'r-', linewidth=2, label='Bloch sphere boundary')

    ax2.set_xlabel('b_x', fontsize=12)
    ax2.set_ylabel('b_z', fontsize=12)
    ax2.set_title('Bloch Walk Component\n(Bloch Disk)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('/Users/3l3ktr4/dev/simplex/composite_ground_truth.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: composite_ground_truth.png")
    plt.show()

    print("\n" + "=" * 60)
    print("Composite Process implementation complete!")
    print("\nKey insights:")
    print("  - Composite token vocabulary: 12 tokens (3 × 4)")
    print("  - Belief geometry: 4D (2D simplex ⊗ 2D Bloch disk)")
    print("  - Processes are INDEPENDENT (run in parallel)")
    print("  - This is the ground truth the transformer should learn!")
    print("=" * 60)
