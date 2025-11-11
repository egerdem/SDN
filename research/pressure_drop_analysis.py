import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')  # Set the backend to Qt5

def scat():
    """
    Create the 5x5 scattering matrix S = (2/5)*ones(5,5) - I.
    """
    K = 5
    S = (2 / K) * np.ones((K, K)) - np.eye(K)
    return S


def injection_vec(c, source_pressure_injection_coef = 1):
    # Orginal sdn : source_pressure_injection_coef = 1/2, c = 1
    K = 5
    psk = ((5 - c) / 4) * source_pressure_injection_coef * np.ones(K)
    psk[0] = c * source_pressure_injection_coef
    return psk

def form_input_vector(c):
    """
    Form the input vector b based on a weighting parameter c using the new rule:
    b = ((5 - c) / 4) * (1/2) * ones(5) for all entries,
    then b[0] = c * (1/2).
    """
    K = 5
    b = ((5 - c) / 4) * (1 / 2) * np.ones(K)
    b[0] = c * (1 / 2)
    return b

def form_input_vector_new(b):
    # for b in [0, 0.5,1,1.5,2.,2.5,3.,3.5,4.,4.5,5]:
    b0 = (5-2*b)/8
    if b>b0 or b==b0:
        pr = 2/5 * (1-b)
        pr_vec = [b, b0, b0, b0, b0]
    else:
        pr = (2*b-3)/5
        pr_vec = [b0, b, b, b, b]

    print(f"c = {2*b}, b = {b}, b0 = {b0}, \n, {pr_vec}, pr = {pr}")
    outgoing_vec = np.dot(S,pr_vec)
    print("outgoing_vec", outgoing_vec)
    if np.max(np.abs(outgoing_vec)) == pr:
        print("yes! true")
    else:
        print("no!:", np.max(outgoing_vec), pr)
    return pr_vec

def out_vec_with_single_largest_variable(S, p_in, flag="max"):
    """
    Apply the scattering matrix S to the input vector b.
    Then, select the element with the largest absolute value and
    zero out all the others.

    Returns the new output vector.
    """
    out = np.dot(S, p_in)

    idx = np.argmax(np.abs(out))
    # print(out[idx])
    p_out_plus = np.zeros_like(out)
    p_out_plus[idx] = out[idx]

    return out, p_out_plus


def mic_pressure(p_out, coef=1/5):
    # original sdn: coef = 2/5
    """
    Compute the microphone pressure from the current node pressure,
    defined as (2/5) times the sum of the input vector.
    """
    return coef * np.sum(p_out)


def simulate_mic_pressures(c, num_iterations=3):
    """
    Simulate the first num_iterations mic pressures using the scattering
    process, given a weighting parameter c.

    Process:
      1. Form the initial input vector using the new rule.
      2. Compute the first mic pressure from this vector.
      3. For each subsequent iteration, apply the scattering matrix,
         filter the output (keeping only the largest element),
         and compute the new mic pressure.

    Returns a list of mic pressures.
    """
    S = scat()
    pressures = []
    # current = form_input_vector(c)
    current = injection_vec(c, source_pressure_injection_coef=1)
    pressures.append(mic_pressure(current))

    for _ in range(1, num_iterations):
        out, current = out_vec_with_single_largest_variable(S, current, flag="max") # singled out outgoing vector
        pressures.append(mic_pressure(current))
    return pressures


def main():
    # Test for a range of weighting parameters c from 1 to 6.
    c_values = [1, 2, 3, 4, 5]
    num_iterations = 10  # first three mic pressures
    results = {}

    for c in c_values:
        mic_pressures = simulate_mic_pressures(c, num_iterations)
        results[c] = mic_pressures
        print(f"Weighting c = {c}, Mic pressures (iterations): {mic_pressures}")

        res = [results]

    for i, res in enumerate(res):
        # Plot the results.
        plt.figure(figsize=(10, 6))
        iterations = np.arange(num_iterations)
        for c in c_values:
            plt.plot(iterations, res[c], marker='o', label=f'c = {c}')

        plt.xlabel("Iteration (Reflection Number)")
        plt.ylabel("Mic Pressure (scaled)")
        plt.title("Mic Pressure for First Three Reflections vs. Weighting Parameter")
        plt.xticks(iterations)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # main()
    for c in [-3,-2,-1,0,1,2,3,4,5,6,7]:
        print("\nc=", c, "injection:", injection_vec(c))
        print(np.dot(scat(),injection_vec(c)))   

    """S = scat()
    form_input_vector_new(2)

    rank = np.linalg.matrix_rank(S)
    num_vectors = S.shape[0]  # Assuming S is square
    is_independent = rank == num_vectors
    # print("S has a basis of linearly independent vectors:", is_independent)
    eigenvalues, eigenvectors = np.linalg.eig(S)  # Compute eigenvalues and eigenvectors of the scattering matrix
    print("Eigenvalues of S:", eigenvalues)
    print("Eigenvectors of S:", eigenvectors)
    pi = form_input_vector(1) # [0.5, 0.5, 0.5, 0.5, 0.5]
    po = np.dot(S,pi) # [0.5, 0.5, 0.5, 0.5, 0.5]
    out, pi_1st,_ = out_vec_with_single_largest_variable(S, pi) # [0.5, 0. , 0. , 0. , 0. ]
    po_1st = np.dot(S,pi_1st) # [-0.3,  0.2,  0.2,  0.2,  0.2]"""

    """# Quadratic form: p1 = pi_1st^T * S * pi_1st
    # Ensure pi_1st is a column vector for correct multiplication
    pi_1st_col = pi_1st.reshape(-1, 1)  # Convert to column vector
    p1 = np.dot(pi_1st_col.T, np.dot(S, pi_1st_col))  # p1 is a scalar

    # Quadratic form: p2 = po_1st^T * S * po_1st
    # Ensure po_1st is a column vector for correct multiplication
    po_1st_col = po_1st.reshape(-1, 1)  # Convert to column vector
    p2 = np.dot(po_1st_col.T, np.dot(S, po_1st_col))  # p2 is a scalar"""

    # print("p1:", p1)
    # print("p2:", p2)

    # print("pi_1st:", pi_1st)
    # u = np.array([1, 0, 0, 0, 0]).reshape(5, 1)
    # S_specular = np.eye(5)-2*np.dot(u,ut)
    # alfa = 0.9
    # S_hybrid = alfa*S + (1-alfa)*S_specular
