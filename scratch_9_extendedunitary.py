import numpy as np


def injection_vec(c, source_pressure_injection_coef=1/2):
    K = 5
    cn = (5 - c) / 4.0 # Use float division
    psk = cn * source_pressure_injection_coef * np.ones(K)
    psk[0] = c * source_pressure_injection_coef
    return psk

size = 5
S = (2 / size) * np.ones((size, size)) - np.eye(size)
K = size
b = (2 / 5) * np.ones((1,K))
c = (1 / 2) * np.ones((K,1))

# d: Direct source-to-microphone transfer
d0 = np.array([[1]])
d1 = np.array([[0]])

def unify(c, d, S = S):

    u = np.block([[S, c],
              [b, d]])
    return u

# Construct the full scattering matrix
S_extended0  = np.block([[S, c],
                       [b, d0]])

S_extended1 = np.block([[S, c],
                       [b, d1]])
# print(S)
# print("6x6")
# print(S_extended)
svd = np.linalg.svd(S_extended0)
print(np.round(svd.S,3))

svd1 = np.linalg.svd(S_extended1)
print(np.round(svd1.S,3))

# np.set_printoptions(precision=3, suppress=True)
#
# b = np.ones((K,1))*1/2
#
# attenuation_values = [0.89 for i in range(5)]
# att = np.diag(attenuation_values)
# S_eff = np.dot(S,att)
# print(np.dot( np.dot(S,att), b))
# print(np.dot(S_eff, b))

"""for weight in range(1, 7):
    b =  (5 - weight) / 4  * (1 / 2) * np.ones((K,1))
    b[0] = weight * (1 / 2)
    S_extended = np.block([[S, b],
                           [c, d]])
    print("new matrix, weight =", weight)
    # print(S_extended)
    svd = np.linalg.svd(S_extended)
    singular = svd.S
    print(f"svd = {singular}\n")
    print(svd)"""

"""
# Example scattering matrix for the SDN sub-network (for K = 5 nodes)

# ---- Approach 1: Approximately Unitary with Fixed d = 1 ----
# In this approach, we fix the direct path gain d = 1.
# To minimize the deviation from unitarity, we choose b with a small norm,
# and then we set c = -S @ b.
# If the norm of b is very small, the extra energy injected is negligible.
# Note: Here, unitarity is approximate because  d^2 + ||b||^2 will not equal 1 exactly,
# unless ||b|| is negligible.
b_approx = 0.4 * np.ones((size, 1))  # b with small magnitude; you can also enforce zero-sum if desired.
d_approx = np.array([[1]])           # Fixed direct path gain

# Compute c such that c = -S * b
c_approx = -S @ b_approx

# Construct the extended scattering matrix:
# S_extended = [ S,    b ]
#              [ c,    d ]
S_extended_approx = np.block([[S, b_approx],
                              [c_approx.T, d_approx]])
S_extended_approx_inv = np.linalg.inv(S_extended_approx)  # Calculate the inverse of S_extended_approx

print("Approach 1: Approximately Unitary with Fixed d=1")
print("S:")
print(S)
print("Extended S matrix (approximate):")
print(S_extended_approx)
print("unitary check\n", S_extended_approx.T, "\n", S_extended_approx_inv)

print("Norm of b (should be small):", np.linalg.norm(b_approx))
print("Frobenius norm difference from unitary condition (approx):", np.linalg.norm(S_extended_approx.T @ S_extended_approx - np.eye(size+1)))

# ---- Approach 2: Exactly Unitary Extended Block ----
# In this approach, we allow d to be adjusted.
# We choose b (again, preferably with a small norm), and then set:
#   d = sqrt(1 - ||b||^2)
#   c = -S @ b / d
# This ensures that the bottom-right block (i.e. the combination of b and d)
# satisfies b^T*b + d^2 = 1 exactly.
b_exact = 0.05 * np.ones((size, 1))  # Again, choose a small b vector.
norm_b = np.linalg.norm(b_exact)
d_exact = np.array([[np.sqrt(1 - norm_b**2)]])  # Adjust d to guarantee energy conservation.
c_exact = -S @ b_exact / d_exact  # Compute c accordingly.

# Construct the extended scattering matrix:
S_extended_exact = np.block([[S, b_exact],
                             [c_exact.T, d_exact]])
S_extended_exact_inv = np.linalg.inv(S_extended_exact)  # Calculate the inverse of S_extended_approx
print("unitary check\n", S_extended_exact.T, "\n", S_extended_exact_inv)

print("\nApproach 2: Exactly Unitary Extended Block")
print("S:")
print(S)
print("Extended S matrix (exact unitary extension):")
print(S_extended_exact)
# Verify unitarity: S_extended_exact.T @ S_extended_exact should be identity.
print("Frobenius norm difference from unitary condition (exact):", np.linalg.norm(S_extended_exact.T @ S_extended_exact - np.eye(size+1)))

def is_unitary(S):
    return np.allclose(S.T @ S, np.eye(S.shape[0]))"""