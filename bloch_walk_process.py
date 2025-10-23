"""
Bloch Walk Process Generator

From Paper Appendix D.2:
A quantum process that can be generated with a single qubit of quantum memory,
but has NO finite classical HMM representation.

The belief states form a 2D slice through the Bloch sphere (x-z plane).
"""

import numpy as np
from typing import Tuple, List


class BlochWalkProcess:
    """
    Bloch Walk: Quantum process with 1 qubit memory

    Hidden states: Quantum states (density matrices on Bloch sphere)
    Observable tokens: {0, 1, 2, 3}

    Parameters from paper experiments:
    - α (alpha) = 1
    - β (beta) = √51
    """

    def __init__(self, alpha: float = 1.0, beta: float = np.sqrt(51), seed: int = 42):
        """
        Initialize Bloch Walk process.

        Args:
            alpha: Parameter controlling dynamics
            beta: Parameter controlling dynamics
            seed: Random seed for reproducibility
        """
        self.alpha = alpha
        self.beta = beta

        # Derived normalization constant
        # γ = 1/(2√(α² + β²))
        self.gamma = 1.0 / (2.0 * np.sqrt(alpha**2 + beta**2))

        self.n_tokens = 4
        self.token_names = ['0', '1', '2', '3']

        # Pauli matrices (basis for qubit operations)
        self.I = np.array([[1, 0], [0, 1]], dtype=complex)  # Identity
        self.sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)  # X (bit flip)
        self.sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)  # Y
        self.sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)  # Z (phase flip)

        # Build Kraus operators (quantum measurement operators)
        self.build_kraus_operators()

        # Initial state: maximally mixed (center of Bloch ball)
        self.initial_density = self.I / 2.0

        # Random state
        self.rng = np.random.RandomState(seed)

    def build_kraus_operators(self):
        """
        Build the 4 Kraus operators from the paper (Equations D4-D7, page 17).

        Kraus operators K_n act on the quantum state (density matrix).
        They represent "quantum measurement" that produces observable token n.

        K0: Reinforces +z direction (token 0)
        K1: Reinforces -z direction (token 1)
        K2: Reinforces +x direction (token 2)
        K3: Reinforces -x direction (token 3)
        """
        α, β, γ = self.alpha, self.beta, self.gamma

        # From Equation D4: K0 = γ(αI + βσ_z)
        self.K0 = γ * (α * self.I + β * self.sigma_z)

        # From Equation D5: K1 = γ(αI - βσ_z)
        self.K1 = γ * (α * self.I - β * self.sigma_z)

        # From Equation D6: K2 = γ(αI + βσ_x)
        self.K2 = γ * (α * self.I + β * self.sigma_x)

        # From Equation D7: K3 = γ(αI - βσ_x)
        self.K3 = γ * (α * self.I - β * self.sigma_x)

        self.kraus_operators = [self.K0, self.K1, self.K2, self.K3]

        # Verify completeness: Σ K†K = I (required for valid quantum instrument)
        completeness = sum([K.conj().T @ K for K in self.kraus_operators])
        assert np.allclose(completeness, self.I), "Kraus operators must satisfy completeness"

    def density_to_bloch(self, rho: np.ndarray) -> np.ndarray:
        """
        Convert density matrix to Bloch vector representation.

        For a qubit: ρ = I/2 + (b_x·σ_x + b_y·σ_y + b_z·σ_z)/2

        Bloch vector: [b_x, b_y, b_z]
        - Pure states: |b| = 1 (on surface of Bloch sphere)
        - Mixed states: |b| < 1 (inside Bloch ball)

        Args:
            rho: 2x2 density matrix

        Returns:
            bloch: 3D Bloch vector [b_x, b_y, b_z]
        """
        # Extract Bloch vector components
        # b_i = 2 * Re(Tr(ρ · σ_i))  for real density matrices
        # But actually b_i = Tr(ρ · σ_i) works since result is real
        b_x = np.real(np.trace(rho @ self.sigma_x))
        b_y = np.real(np.trace(rho @ self.sigma_y))
        b_z = np.real(np.trace(rho @ self.sigma_z))

        return np.array([b_x, b_y, b_z])

    def bloch_to_density(self, bloch: np.ndarray) -> np.ndarray:
        """
        Convert Bloch vector to density matrix.

        ρ = I/2 + (b_x·σ_x + b_y·σ_y + b_z·σ_z)/2

        Args:
            bloch: 3D Bloch vector [b_x, b_y, b_z]

        Returns:
            rho: 2x2 density matrix
        """
        b_x, b_y, b_z = bloch
        rho = 0.5 * (self.I + b_x * self.sigma_x + b_y * self.sigma_y + b_z * self.sigma_z)
        return rho

    def apply_kraus_operator(self, rho: np.ndarray, token: int) -> Tuple[np.ndarray, float]:
        """
        Apply Kraus operator corresponding to observed token.

        This is the quantum "Bayesian update":
        ρ' = K_n ρ K_n† / Tr(K_n ρ K_n†)

        Args:
            rho: Current density matrix (2x2)
            token: Observed token (0, 1, 2, or 3)

        Returns:
            new_rho: Updated density matrix (2x2)
            prob: Probability of observing this token
        """
        K = self.kraus_operators[token]

        # Unnormalized new state
        new_rho_unnorm = K @ rho @ K.conj().T

        # Probability of this measurement outcome
        prob = np.real(np.trace(new_rho_unnorm))

        # Normalize to get valid density matrix
        if prob > 1e-10:
            new_rho = new_rho_unnorm / prob
        else:
            # If probability is essentially zero, return original state
            new_rho = rho

        return new_rho, prob

    def generate_sequence(self, length: int, initial_rho: np.ndarray = None) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Generate a sequence of observable tokens from the Bloch Walk process.

        Args:
            length: Number of tokens to generate
            initial_rho: Starting quantum state (if None, use maximally mixed)

        Returns:
            tokens: Array of shape (length,) with token IDs {0, 1, 2, 3}
            states: List of density matrices at each step
        """
        if initial_rho is None:
            current_rho = self.initial_density.copy()
        else:
            current_rho = initial_rho.copy()

        tokens = []
        states = []

        for _ in range(length):
            # Compute probability of each token given current state
            probs = []
            for token_id in range(self.n_tokens):
                K = self.kraus_operators[token_id]
                # P(token) = Tr(K ρ K†)
                prob = np.real(np.trace(K @ current_rho @ K.conj().T))
                probs.append(prob)

            probs = np.array(probs)
            probs = probs / probs.sum()  # Normalize (should already sum to 1)

            # Sample token
            token = self.rng.choice(self.n_tokens, p=probs)

            # Update quantum state given observed token
            current_rho, _ = self.apply_kraus_operator(current_rho, token)

            tokens.append(token)
            states.append(current_rho.copy())

        return np.array(tokens), states

    def compute_belief_state(self, token_history: List[int]) -> np.ndarray:
        """
        Compute the belief state (quantum density matrix) given a token history.

        This performs quantum Bayesian filtering:
        ρ_{t+1} = K_n ρ_t K_n† / Tr(K_n ρ_t K_n†)

        Args:
            token_history: List of observed tokens

        Returns:
            bloch: Bloch vector representation [b_x, b_y, b_z]
                   This is a point on/in the Bloch sphere
        """
        # Start with maximally mixed state
        rho = self.initial_density.copy()

        # Apply each observation sequentially
        for token in token_history:
            rho, _ = self.apply_kraus_operator(rho, token)

        # Convert to Bloch vector for visualization
        return self.density_to_bloch(rho)

    def get_belief_geometry(self, max_length: int = 8) -> Tuple[List[List[int]], np.ndarray]:
        """
        Compute belief states for all possible token histories up to max_length.

        For Bloch Walk, beliefs are Bloch vectors (points on/in Bloch sphere).

        Args:
            max_length: Maximum length of token histories to consider

        Returns:
            histories: List of token histories (each is a list of ints)
            beliefs: Array of shape (n_histories, 3) with Bloch vectors
        """
        # Initialize with empty history
        histories = [[]]
        beliefs = [self.density_to_bloch(self.initial_density)]

        # Build up histories incrementally
        for length in range(1, max_length + 1):
            # Get all histories of length (length-1)
            for prev_history in [h for h in histories if len(h) == length - 1]:
                # Try appending each possible token
                for token in range(self.n_tokens):
                    new_history = prev_history + [token]
                    new_belief = self.compute_belief_state(new_history)

                    # Check if reachable (has non-zero probability)
                    if self.compute_history_probability(new_history) > 1e-10:
                        histories.append(new_history)
                        beliefs.append(new_belief)

        # Convert to numpy array
        # Shape: (n_histories, 3) for 3D Bloch vectors
        return histories, np.array(beliefs)

    def compute_history_probability(self, token_history: List[int]) -> float:
        """
        Compute P(history) = probability of observing this token sequence.
        """
        rho = self.initial_density.copy()
        prob = 1.0

        for token in token_history:
            rho, p = self.apply_kraus_operator(rho, token)
            prob *= p

        return prob


# Test the implementation
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    print("=" * 60)
    print("Testing Bloch Walk Process")
    print("=" * 60)

    # Create process
    bloch = BlochWalkProcess()

    print(f"\nParameters:")
    print(f"  α = {bloch.alpha}")
    print(f"  β = {bloch.beta}")
    print(f"  γ = {bloch.gamma:.6f}")

    print(f"\nInitial state (maximally mixed):")
    print(f"  Density matrix:\n{bloch.initial_density}")
    print(f"  Bloch vector: {bloch.density_to_bloch(bloch.initial_density)}")
    print(f"  (Center of Bloch ball = maximum uncertainty)")

    # Generate sample sequence
    print(f"\nGenerating sample sequence of length 20:")
    tokens, states = bloch.generate_sequence(20)
    token_str = ' '.join([bloch.token_names[t] for t in tokens])
    print(f"  Tokens: {token_str}")

    print(f"\nBloch vectors along trajectory:")
    for i, (token, state) in enumerate(zip(tokens[:5], states[:5])):
        b = bloch.density_to_bloch(state)
        print(f"  After token {token}: Bloch = [{b[0]:6.3f}, {b[1]:6.3f}, {b[2]:6.3f}], |b|={np.linalg.norm(b):.3f}")

    # Compute belief for short histories
    print(f"\nBelief state examples (Bloch vectors):")
    print(f"  (Showing how belief changes as you observe tokens)")
    for history_tokens in [[], [0], [2], [0, 0], [2, 2]]:
        belief = bloch.compute_belief_state(history_tokens)
        history_str = ' '.join([bloch.token_names[t] for t in history_tokens]) if history_tokens else "(empty)"
        print(f"  After '{history_str:8s}': [{belief[0]:6.3f}, {belief[1]:6.3f}, {belief[2]:6.3f}]")

    # Get full belief geometry
    print(f"\nComputing full belief geometry (up to length 6)...")
    histories, beliefs = bloch.get_belief_geometry(max_length=6)
    print(f"  Total unique histories: {len(histories)}")
    print(f"  Belief states shape: {beliefs.shape}")
    print(f"  (Each row is a 3D Bloch vector!)")

    print("\n" + "=" * 60)
    print("Visualizing Ground Truth Belief Geometry")
    print("=" * 60)

    # 3D visualization of Bloch sphere
    fig = plt.figure(figsize=(14, 6))

    # Plot 1: Full 3D Bloch ball
    ax1 = fig.add_subplot(121, projection='3d')
    scatter = ax1.scatter(beliefs[:, 0], beliefs[:, 1], beliefs[:, 2],
                          c=range(len(beliefs)), cmap='viridis', alpha=0.6, s=20)

    # Draw Bloch sphere wireframe
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='cyan')

    ax1.set_xlabel('b_x', fontsize=12)
    ax1.set_ylabel('b_y', fontsize=12)
    ax1.set_zlabel('b_z', fontsize=12)
    ax1.set_title('Bloch Walk Belief Geometry\n3D Bloch Sphere', fontsize=14)

    # Plot 2: x-z plane slice (y should be ~0)
    ax2 = fig.add_subplot(122)
    scatter2 = ax2.scatter(beliefs[:, 0], beliefs[:, 2],
                           c=range(len(beliefs)), cmap='viridis', alpha=0.6, s=20)

    # Draw unit circle (boundary of Bloch disk)
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = np.cos(theta)
    circle_z = np.sin(theta)
    ax2.plot(circle_x, circle_z, 'r-', linewidth=2, label='Bloch sphere boundary')

    ax2.set_xlabel('b_x', fontsize=12)
    ax2.set_ylabel('b_z', fontsize=12)
    ax2.set_title('x-z Plane Slice\n(y component ≈ 0)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_aspect('equal')

    plt.colorbar(scatter2, ax=ax2, label='History order')
    plt.tight_layout()
    plt.savefig('/Users/3l3ktr4/dev/simplex/bloch_walk_ground_truth.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: bloch_walk_ground_truth.png")
    plt.show()

    print("\n" + "=" * 60)
    print("Bloch Walk Process implementation complete!")
    print("Key insight: Beliefs live on 2D slice of Bloch sphere (x-z plane)")
    print("This is what the neural network should learn to represent!")
    print("=" * 60)
