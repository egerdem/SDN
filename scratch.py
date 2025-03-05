import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')  # Set the backend to Qt5


def scattering_matrix_crate(increase_coef=0.2):
    c = increase_coef
    original_matrix = (2 / 5) * np.ones((5, 5)) - np.eye(5)
    # Create a new matrix by adjusting the diagonal and off-diagonal elements
    adjusted_matrix = np.copy(original_matrix)

    # Decrease diagonal elements by c
    np.fill_diagonal(adjusted_matrix, adjusted_matrix.diagonal() - c)

    # Increase only off-diagonal elements by c/4
    off_diagonal_mask = np.ones(adjusted_matrix.shape, dtype=bool)  # Create a mask for all elements
    np.fill_diagonal(off_diagonal_mask, False)  # Set diagonal elements to False

    # Add c/4 to only the off-diagonal elements
    adjusted_matrix[off_diagonal_mask] += (c / 4)
    return adjusted_matrix

def calculate_recursive_scattering(iterations, initial_input, sc):
    """
    Recursively calculate scattering matrix operations.
    
    Args:
        iterations (int): Number of iterations to perform
        initial_input (np.ndarray): Initial input vector
        
    Returns:
        tuple: (final output vector, sum of final output)
    """

    def recursive_step(input_vector, current_iter):
        if current_iter == 0:
            return input_vector, np.sum(input_vector)
        
        # Calculate output using scattering matrix
        output = np.dot(sc, input_vector)
        
        # Zero out all elements except index 1
        filtered_output = np.zeros(5)
        filtered_output[1] = output[1]
        
        # Recursive call
        return recursive_step(filtered_output, current_iter - 1)
    
    return recursive_step(initial_input, iterations)

# Example usage

initial_vector = np.array([1/2, 1/2, 1/2, 1/2, 1/2])
initial_vector2 = np.array([1/8, 2, 1/8, 1/8, 1/8])

# Arrays to store results for plotting
iterations = range(8)
results1 = []
results2 = []

# Initialize scattering matrix
sc = np.ones(5) * 2 / 5 - np.eye(5)
sc2 = scattering_matrix_crate(0.4)
print(sc, "\n", sc2)
# Calculate results for both initial vectors
for i in iterations:
    result, sum_result = calculate_recursive_scattering(i, initial_vector, sc)
    results1.append(sum_result)
    print(f"Iteration {i}:", result, f"{sum_result:.2f}")
    
    result, sum_result = calculate_recursive_scattering(i, initial_vector, sc2)
    results2.append(sum_result)
    print(f"Iteration {i}:", result, f"{sum_result:.2f}")

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(iterations, results1, 'b.-', label='Initial Vector 1 [1/5, ...]', linewidth=2, markersize=10)
plt.plot(iterations, results2, 'r.-', label='Initial Vector 2 [1/5...', linewidth=2, markersize=10)

plt.xlabel('Iteration')
plt.ylabel('Scaled Sum (sum_result)')
plt.title('Comparison of Scaled Sums for Different Initial Vectors')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.xticks(iterations)

plt.tight_layout()
plt.show()







