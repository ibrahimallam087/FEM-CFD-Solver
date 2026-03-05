import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Geometry and Physical Constants
# ===============================
L = 10
mid = L / 2
rho = 1.225


# ===============================
# Area Integrals
# ===============================
def compute_I1(x1, x2):
    return (2.5*x2 - 3*x2**2/10 + x2**3/50
            - 2.5*x1 + 3*x1**2/10 - x1**3/50)


def compute_I2(x1, x2):
    return (1.5*x2 - x2**2/10 + x2**3/150
            - 1.5*x1 + x1**2/10 - x1**3/150)


# ===============================
# Connectivity
# ===============================
def generate_connectivity(ne):
    conn = np.zeros((ne, 2), dtype=int)
    for e in range(ne):
        conn[e] = [e, e+1]
    return conn


# ===============================
# FEM Solver
# ===============================
def incomp_nozzle(ne):

    nn = ne + 1
    x = np.linspace(0, L, nn)
    connect = generate_connectivity(ne)

    KG = np.zeros((nn, nn))
    RHS = np.zeros(nn)

    # ---------- Assembly ----------
    for node1, node2 in connect:

        x1, x2 = x[node1], x[node2]
        Le = x2 - x1

        if x2 <= mid:
            I = compute_I1(x1, x2)
        elif x1 >= mid:
            I = compute_I2(x1, x2)
        else:
            I = compute_I1(x1, mid) + compute_I2(mid, x2)

        k_local = (rho * I / Le**2) * np.array([[1, -1], [-1, 1]])

        KG[np.ix_([node1, node2], [node1, node2])] += k_local

    # ---------- Dirichlet BC ----------
    bc_node = 0
    bc_value = 0.0

    k_diag = KG[bc_node, bc_node]

    KG[bc_node, :] = 0
    RHS -= KG[:, bc_node] * bc_value
    KG[:, bc_node] = 0

    KG[bc_node, bc_node] = k_diag
    RHS[bc_node] = bc_value * k_diag

    # ---------- Neumann BC ----------
    u_L = 69.9854
    A_L = L/10 + 0.5*(L/10)

    RHS[-1] = rho * A_L * u_L

    # ---------- Solve ----------
    phi = np.linalg.solve(KG, RHS)

    return phi


# ===============================
# Mesh Convergence Study
# ===============================
if __name__ == "__main__":

    E = np.arange(10, 110, 10)

    phi_store = []
    x_store = []
    err_history = []

    for i, ne in enumerate(E):

        phi = incomp_nozzle(ne)
        x = np.linspace(0, L, ne+1)

        phi_store.append(phi)
        x_store.append(x)

        if i > 0:

            phi_old = phi_store[i-1]
            phi_new = phi

            mid_old = len(phi_old)//2
            mid_new = len(phi_new)//2

            delta_mid = phi_new[mid_new] / phi_old[mid_old]
            delta_end = phi_new[-1] / phi_old[-1]

            err_mid = 100 * abs(delta_mid - 1)
            err_end = 100 * abs(delta_end - 1)

            err_history.append(max(err_mid, err_end))

    # ---------- Plot φ(x) ----------
    plt.figure(figsize=(7,5))

    for i in range(len(phi_store)):
        plt.plot(x_store[i], phi_store[i],
                 linewidth=1.5,
                 label=f"Elements = {E[i]}")

    plt.xlabel("x")
    plt.ylabel("Velocity Potential φ")
    plt.title("Variation of φ with Mesh Refinement")
    plt.grid(True)
    plt.legend()
    plt.show()

    # ---------- Plot Error ----------
    plt.figure(figsize=(7,5))

    plt.plot(E[1:], err_history,
             marker='o',
             linewidth=2)

    plt.xlabel("Number of Elements")
    plt.ylabel("Percentage Error (%)")
    plt.title("Mesh Convergence History")
    plt.grid(True)
    plt.show()