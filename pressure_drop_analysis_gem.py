import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

def scat():
    K = 5
    S = (2 / K) * np.ones((K, K)) - np.eye(K)
    return S

def injection_vec(c, source_pressure_injection_coef=1):
    K = 5
    cn = (5 - c) / 4.0 # Use float division
    psk = cn * source_pressure_injection_coef * np.ones(K)
    psk[0] = c * source_pressure_injection_coef
    return psk

def mic_pressure(p_in_sparse_vec, coef=1/5):
    # Mic pressure from a sparse input vector (only one non-zero element)
    # representing the wave arriving from the previous step.
    return coef * np.sum(p_in_sparse_vec) # Sum is just the single non-zero value

def simulate_mic_pressures_two_paths(c, num_iterations=10):
    S = scat()
    pressures_path1 = [] # Path originating from V_dom = 2-c
    pressures_path2 = [] # Path originating from V_non_dom = 2-cn

    # Initial step (Source injection)
    p_S_vec = injection_vec(c, source_pressure_injection_coef=1)
    initial_node_pressure = (1/5) * np.sum(p_S_vec) # Should be 1
    # We typically measure pressure *after* a reflection, so start tracking from iteration 1

    # First outgoing vector
    out_first = np.dot(S, p_S_vec)
    V_dom = out_first[0] # Component 2-c
    V_non_dom = out_first[1] # Component 2-cn (any index from 1 to 4)

    # Initialize the sparse vectors for the two paths for the first reflection step
    current_path1 = np.zeros_like(out_first); current_path1[0] = V_dom
    current_path2 = np.zeros_like(out_first); current_path2[1] = V_non_dom # Use index 1 for non-dom

    # --- Simulate Path 1 (originated from V_dom) ---
    temp_current_path1 = current_path1.copy()
    # Calculate pressure arriving *from* this first step
    pressures_path1.append(initial_node_pressure)
    pressures_path1.append(mic_pressure(temp_current_path1))
    for _ in range(1, num_iterations):
        # Scatter the incoming wave (which is sparse)
        out_path1 = np.dot(S, temp_current_path1)
        # Find strongest component of *this* scattering result
        idx1 = np.argmax(np.abs(out_path1))
        # Create next sparse incoming vector
        temp_current_path1 = np.zeros_like(out_path1)
        temp_current_path1[idx1] = out_path1[idx1]
        # Calculate pressure arriving from this step
        pressures_path1.append(mic_pressure(temp_current_path1))

    # --- Simulate Path 2 (originated from V_non_dom) ---
    temp_current_path2 = current_path2.copy()
    # Calculate pressure arriving *from* this first step
    pressures_path2.append(initial_node_pressure)
    pressures_path2.append(mic_pressure(temp_current_path2))
    for _ in range(1, num_iterations):
        # Scatter the incoming wave (which is sparse)
        out_path2 = np.dot(S, temp_current_path2)
        # Find strongest component of *this* scattering result
        idx2 = np.argmax(np.abs(out_path2))
         # Create next sparse incoming vector
        temp_current_path2 = np.zeros_like(out_path2)
        temp_current_path2[idx2] = out_path2[idx2]
         # Calculate pressure arriving from this step
        pressures_path2.append(mic_pressure(temp_current_path2))

    # Note: pressures_path1[0] is the pressure from V_dom arriving after 1 reflection (iter 1)
    # pressures_path1[1] is the pressure from the max component of S*V_dom arriving after 2 reflections (iter 2) etc.
    return pressures_path1, pressures_path2

def main(c_values = [-5,-4,-3,-2,-1,0,1, 2, 3, 4, 5,6,7]):

    num_iterations = 3
    results_p1 = {}
    results_p2 = {}

    for c in c_values:
        p1, p2 = simulate_mic_pressures_two_paths(c, num_iterations)
        results_p1[c] = p1
        results_p2[c] = p2
        print(f"c = {c}")
        print(f"  Path 1 (from 2-c): {np.round(p1, 3)}")
        print(f"  Path 2 (from 2-cn): {np.round(p2, 3)}")

    # Plotting
    plt.figure(figsize=(12, 14))  # Increased figure height for 2 rows
    iterations = np.arange(0, num_iterations + 1)

    # Top row: Regular values
    # Plot Path 1
    ax1 = plt.subplot(2, 2, 1)
    for c in c_values:
        ax1.plot(iterations, results_p1[c], marker='o', label=f'c = {c}')
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Mic Pressure (Dominant Path)")
    # ax1.set_title("Path branching from dominant directions = 2-c")
    ax1.set_xticks(iterations)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # Plot Path 2
    ax2 = plt.subplot(2, 2, 2)
    for c in c_values:
        ax2.plot(iterations, results_p2[c], marker='s', label=f'c = {c}')  # Different marker
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Mic Pressure (Non-dominant path)")
    # ax2.set_title("Path branching from the non-dom directions = (3+c)/4")
    ax2.set_xticks(iterations)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    # Bottom row: Absolute values
    # Plot Path 1 (absolute values)
    # ax3 = plt.subplot(2, 2, 3)
    # for c in c_values:
    #     ax3.plot(iterations, np.abs(results_p1[c]), marker='o', label=f'c = {c}')
    # ax3.set_xlabel("Iteration (Reflection Number)")
    # ax3.set_ylabel("Absolute Mic Pressure (Path 1)")
    # ax3.set_title("Absolute values: Path starting from V_dom = 2-c")
    # ax3.set_xticks(iterations)
    # ax3.grid(True, linestyle='--', alpha=0.7)
    # ax3.legend()
    #
    # # Plot Path 2 (absolute values)
    # ax4 = plt.subplot(2, 2, 4)
    # for c in c_values:
    #     ax4.plot(iterations, np.abs(results_p2[c]), marker='s', label=f'c = {c}')
    # ax4.set_xlabel("Iteration (Reflection Number)")
    # ax4.set_ylabel("Absolute Mic Pressure (Path 2)")
    # ax4.set_title("Absolute values: Path starting from V_non_dom = (3+c)/4")
    # ax4.set_xticks(iterations)
    # ax4.grid(True, linestyle='--', alpha=0.7)
    # ax4.legend()

    plt.suptitle("Simulated Mic Pressures Following Two Initial Paths")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjusted rect for better spacing with suptitle
    plt.show()

if __name__ == '__main__':

    # s = scat()
    # j = 1
    # k = 2
    # # swap the rows j and k
    # s_jk = s.copy()
    # s_jk[[j, k]] = s_jk[[k, j]]
    # print(s_jk)
    # print(np.linalg.det(s_jk))
    # print(np.linalg.det(s))
    c_values = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
    main(c_values)

    sa_out_list = []
    c_plus_cn_list = []
    norm_out = []
    norm_out_sq = []
    norm_inj_sq = []

    for c in c_values:
        cn = (5 - c) / 4.0
        inj = injection_vec(c, source_pressure_injection_coef=1/2)
        out = np.dot(scat(),inj)
        sa_out = np.sum(np.abs(out))
        sa_out_list.append(sa_out)
        norm_out.append(np.linalg.norm(out))
        norm_out_sq.append(np.linalg.norm(out)**2)
        norm_inj_sq.append(np.linalg.norm(inj)**2)
        c_plus_cn_list.append(np.abs(c+cn))
        print(f"c: {c} : {np.round(inj, 3)} : {np.round(out, 3)} sa: {np.round(sa_out, 3)}, c+cn: {np.round(np.abs(c+cn), 3)}, norm_out: {np.round(np.linalg.norm(out), 3)}, norm_out_sq: {np.round(np.linalg.norm(out)**2, 3)}")

    print("c_values:", c_values)
    # print("Sum of absolute outputs:", np.round(sa_out_list, 3))
    # print("Absolute c+cn values:", np.round(c_plus_cn_list, 3))
    print("L2 Norm of outputs:", np.round(norm_out, 3))
    print("L2 Norm square of outputs:", np.round(norm_out_sq, 3))
    print("L2 Norm square of input incoming wave vector", np.round(norm_inj_sq, 3))

    # w = np.array([0.4,0.4,0.4,0.4, 0.4])
    # v = np.ones(5)
    # S = scat()
    # w @ S @ v

    # def f_perm(i, N=6):
    #     f = (6*i -((i-1)%N-1) ) % (N*(N-1)) + 1
    #     return(f)
    #
    #
    # def f_perm(i, N=6):
    #     return (   (6 * i - ((i - 1) % N - 1)) % (N * (N - 1))   )


    # N = 6
    # B = N * (N - 1)  # total number of directed branches
    # P = np.zeros((B, B))

    # for i in range(B):
    #     j = f_perm(i, N)
    #     print(i,j)
    #     P[i, j] = 1