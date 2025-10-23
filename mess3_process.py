"""
Mess3 Process Generator

From Paper Appendix D.1:
A classical 3-state Hidden Markov Model that generates sequences
of tokens {a, b, c} with intricate temporal correlations.

The belief states form a fractal pattern in the 2-simplex (triangle).
"""

import numpy as np
from typing import Tuple, List


class Mess3Process:
    """
    Mess3: 3-state Hidden Markov Model

    Hidden states: {0, 1, 2}  (internally we use 0-indexing)
    Observable tokens: {0, 1, 2}  (representing a, b, c)

    Parameters from paper experiments:
    - x = 0.05
    - α (alpha) = 0.85
    """

    def __init__(self, x: float = 0.05, alpha: float = 0.85, seed: int = 42):
        """
        Initialize Mess3 process.

        Args:
            x: Parameter controlling token emission asymmetry
            alpha: Parameter controlling self-transition probability
            seed: Random seed for reproducibility
        """
        self.x = x
        self.alpha = alpha
        self.beta = (1 - alpha) / 2  # Derived parameter
        self.y = 1 - 2 * x           # Derived parameter

        self.n_states = 3
        self.n_tokens = 3

        # Token names for interpretability
        self.token_names = ['a', 'b', 'c']

        # Build transition matrices (one per observable token)
        # These are T^(a), T^(b), T^(c) from Equations D1, D2, D3
        self.build_transition_matrices()

        # Calculate stationary distribution
        self.stationary_dist = self.compute_stationary_distribution()

        # Random state
        self.rng = np.random.RandomState(seed)

    def build_transition_matrices(self):
        """
        Build the labeled transition matrices from the paper.

        T^(x)[i,j] = Pr(next_state=j, observe_token=x | current_state=i)

        Each row sums to probability of emitting that token from that state.
        Sum over all tokens: Σ_x T^(x) = row-stochastic transition matrix
        """
        α, β, x, y = self.alpha, self.beta, self.x, self.y

        # From Equation D1 (paper page 16)
        self.T_a = np.array([
            [α*y, β*x, β*x],
            [α*x, β*y, β*x],
            [α*x, β*x, β*y]
        ])

        # From Equation D2
        self.T_b = np.array([
            [β*y, α*x, β*x],
            [β*x, α*y, β*x],
            [β*x, α*x, β*y]
        ])

        # From Equation D3
        self.T_c = np.array([
            [β*y, β*x, α*x],
            [β*x, β*y, α*x],
            [β*x, β*x, α*y]
        ])

        # Store in list for easy indexing
        self.transition_matrices = [self.T_a, self.T_b, self.T_c]

        # Net transition matrix (sums over all tokens)
        self.T = self.T_a + self.T_b + self.T_c

        # Verify it's row-stochastic
        assert np.allclose(self.T.sum(axis=1), 1.0), "T must be row-stochastic"

    def compute_stationary_distribution(self) -> np.ndarray:
        """
        Compute the stationary distribution π where π·T = π

        This is the left eigenvector with eigenvalue 1.

        TODO: is the result of this in this case just 0.3,0.3,0.3?
        """
        eigenvalues, eigenvectors = np.linalg.eig(self.T.T)

        # Find eigenvector for eigenvalue 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()  # Normalize

        return stationary

    def generate_sequence(self, length: int, initial_state: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a sequence of observable tokens from the Mess3 process.

        Args:
            length: Number of tokens to generate
            initial_state: Starting hidden state (if None, sample from stationary dist)

        Returns:
            tokens: Array of shape (length,) with token IDs {0, 1, 2}
            states: Array of shape (length,) with hidden state IDs {0, 1, 2}
        """
        if initial_state is None:
            # Sample from stationary distribution
            current_state = self.rng.choice(self.n_states, p=self.stationary_dist)
        else:
            current_state = initial_state

        tokens = []
        states = []

        for _ in range(length):
            # Get emission probabilities for current state
            # P(token | state) is the current_state row, summed over next states
            emission_probs = np.zeros(self.n_tokens)
            for token_id in range(self.n_tokens):
                # Sum over next states for this token
                emission_probs[token_id] = self.transition_matrices[token_id][current_state, :].sum()

            # Sample token
            token = self.rng.choice(self.n_tokens, p=emission_probs)

            # Sample next state given token emitted
            # P(next_state | current_state, token) ∝ T^(token)[current_state, next_state]
            transition_probs = self.transition_matrices[token][current_state, :]
            transition_probs = transition_probs / transition_probs.sum()  # Normalize
            next_state = self.rng.choice(self.n_states, p=transition_probs)

            tokens.append(token)
            states.append(current_state)
            current_state = next_state

        return np.array(tokens), np.array(states)

    def compute_belief_state(self, token_history: List[int]) -> np.ndarray:
        """
        Compute the belief state (posterior over hidden states) given a token history.

        This performs Bayesian filtering:
        belief_{t+1} = normalize(belief_t · T^(observed_token))

        Args:
            token_history: List of observed tokens

        Returns:
            belief: Probability distribution over hidden states (sums to 1)
                   Shape: (n_states,) = (3,)
        """
        # Start with stationary distribution
        belief = self.stationary_dist.copy()

        # Update belief for each observed token
        for token in token_history:
            # Bayesian update: belief' = belief · T^(token)
            # This gives unnormalized next belief
            belief = belief @ self.transition_matrices[token]

            # Normalize to get probability distribution
            belief = belief / belief.sum()

        return belief

    def get_belief_geometry(self, max_length: int = 8) -> Tuple[List[List[int]], np.ndarray]:
        """
        Compute belief states for all possible token histories up to max_length.

        This gives us the "ground truth" belief geometry that should be
        discovered by the neural network.

        Args:
            max_length: Maximum length of token histories to consider

        Returns:
            histories: List of token histories (each is a list of ints)
            beliefs: Array of shape (n_histories, n_states) with belief states
        """
        # Initialize with the empty history (no tokens observed yet)
        histories = [[]]
        # Empty history → belief = stationary distribution
        # This is our "starting belief" before seeing any data
        beliefs = [self.stationary_dist]

        # Build up histories incrementally: length 1, then 2, then 3, etc.
        # range(1, max_length + 1) gives us [1, 2, 3, ..., max_length]
        for length in range(1, max_length + 1):

            # For each length, we extend ALL histories of the previous length
            # [h for h in histories if len(h) == length - 1]
            # ↑ This filters histories to only get ones of length (length-1)
            # Example: if length=2, we get all histories of length 1: [[0], [1], [2]]
            for prev_history in [h for h in histories if len(h) == length - 1]:

                # Try appending each possible token (0, 1, 2 for Mess3)
                # This generates ALL possible extensions
                # Example: [0] can become [0,0], [0,1], or [0,2]
                for token in range(self.n_tokens):

                    # Create new history by appending this token
                    # Example: [0, 1] + [2] = [0, 1, 2]
                    new_history = prev_history + [token]

                    # Compute the belief state after observing this history
                    # This does Bayesian filtering:
                    # "Given I saw tokens [0,1,2], what's my belief about hidden state?"
                    # Returns: [p(s0), p(s1), p(s2)] - point in the 2-simplex!
                    new_belief = self.compute_belief_state(new_history)

                    # Check if this history is actually reachable
                    # Some histories have probability ≈ 0 (essentially impossible)
                    # compute_history_probability returns P(seeing this exact sequence)
                    # We only keep histories with P > 1e-10 (very small threshold)
                    # Example: History "a a a a a a a a" might be very unlikely
                    if self.compute_history_probability(new_history) > 1e-10:

                        # This history is reachable! Add it to our collection
                        histories.append(new_history)

                        # Also store its corresponding belief state
                        # This belief is a point in the 2-simplex
                        beliefs.append(new_belief)

        # Convert list of beliefs to numpy array for easier manipulation
        # Shape will be (n_histories, 3) for Mess3
        # Each row is a 3D point: [p(s0), p(s1), p(s2)]
        # But it lives in 2D simplex because p(s0)+p(s1)+p(s2)=1
        return histories, np.array(beliefs)

    def compute_history_probability(self, token_history: List[int]) -> float:
        """
        Compute P(history) = probability of observing this token sequence.

        P(x1, x2, ..., xn) = π · T^(x1) · T^(x2) · ... · T^(xn) · 1
        where 1 is a column vector of all ones.
        """
        prob_vec = self.stationary_dist.copy()

        for token in token_history:
            prob_vec = prob_vec @ self.transition_matrices[token]

        return prob_vec.sum()


# Test the implementation
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    print("=" * 60)
    print("Testing Mess3 Process")
    print("=" * 60)

    # Create process
    mess3 = Mess3Process()

    print(f"\nParameters:")
    print(f"  x = {mess3.x}")
    print(f"  α = {mess3.alpha}")
    print(f"  β = {mess3.beta}")
    print(f"  y = {mess3.y}")

    print(f"\nStationary distribution: {mess3.stationary_dist}")
    print(f"  (This is the long-run average probability of each state)")

    # Generate a sample sequence
    print(f"\nGenerating sample sequence of length 20:")
    tokens, states = mess3.generate_sequence(20)
    token_str = ' '.join([mess3.token_names[t] for t in tokens])
    print(f"  Tokens: {token_str}")
    print(f"  Hidden states: {states}")

    # Compute belief for a short history
    print(f"\nBelief state examples:")
    print(f"  (Showing how belief changes as you observe more tokens)")
    for history_tokens in [[], [0], [0, 1], [0, 1, 2]]:
        belief = mess3.compute_belief_state(history_tokens)
        history_str = ' '.join([mess3.token_names[t] for t in history_tokens]) if history_tokens else "(empty)"
        print(f"  After '{history_str:12s}': {belief} → sum={belief.sum():.3f}")

    # Get full belief geometry
    print(f"\nComputing full belief geometry (up to length 6)...")
    histories, beliefs = mess3.get_belief_geometry(max_length=6)
    print(f"  Total unique histories: {len(histories)}")
    print(f"  Belief states shape: {beliefs.shape}")
    print(f"  (Each row is a point in the 2-simplex!)")

    print("\n" + "=" * 60)
    print("Visualizing Ground Truth Belief Geometry")
    print("=" * 60)

    # Create 2D visualization (simplex is 2D!)
    # We'll use barycentric coordinates: just plot p0 vs p1 (p2 is determined)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: 2D view (most natural for 2-simplex)
    ax = axes[0]
    ax.scatter(beliefs[:, 0], beliefs[:, 1], c=range(len(beliefs)),
               cmap='viridis', alpha=0.6, s=20)

    # Draw simplex boundary (triangle)
    # Vertices: (1,0), (0,1), (0,0)
    triangle = np.array([[1, 0], [0, 1], [0, 0], [1, 0]])
    ax.plot(triangle[:, 0], triangle[:, 1], 'r-', linewidth=2, label='Simplex boundary')

    # Mark vertices
    ax.scatter([1, 0, 0], [0, 1, 0], c='red', s=200, marker='*',
               edgecolors='black', linewidths=2, zorder=10)
    ax.text(1.05, 0, 'State 0', fontsize=12, fontweight='bold')
    ax.text(-0.15, 1, 'State 1', fontsize=12, fontweight='bold')
    ax.text(-0.15, -0.1, 'State 2', fontsize=12, fontweight='bold')

    ax.set_xlabel('p(state 0)', fontsize=12)
    ax.set_ylabel('p(state 1)', fontsize=12)
    ax.set_title('Mess3 Belief Geometry (2D Simplex)\nGround Truth', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal')

    # Plot 2: 3D view (to show it's really using 3 coordinates)
    ax2 = fig.add_subplot(122, projection='3d')
    scatter = ax2.scatter(beliefs[:, 0], beliefs[:, 1], beliefs[:, 2],
                          c=range(len(beliefs)), cmap='viridis', alpha=0.6, s=20)

    # Draw simplex as a triangle in 3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    vertices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    simplex = [[vertices[0], vertices[1], vertices[2]]]
    ax2.add_collection3d(Poly3DCollection(simplex, alpha=0.1, facecolor='red', edgecolor='red'))

    # Mark vertices
    ax2.scatter([1, 0, 0], [0, 1, 0], [0, 0, 1], c='red', s=200, marker='*',
                edgecolors='black', linewidths=2, zorder=10)

    ax2.set_xlabel('p(state 0)', fontsize=10)
    ax2.set_ylabel('p(state 1)', fontsize=10)
    ax2.set_zlabel('p(state 2)', fontsize=10)
    ax2.set_title('Same Data in 3D\n(constrained to 2D surface)', fontsize=14)

    plt.tight_layout()
    plt.savefig('/Users/3l3ktr4/dev/simplex/mess3_ground_truth.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: mess3_ground_truth.png")
    plt.show()

    print("\n" + "=" * 60)
    print("Mess3 Process implementation complete!")
    print("Key insight: Notice the FRACTAL pattern in the simplex!")
    print("This is what the neural network should learn to represent!")
    print("=" * 60)
