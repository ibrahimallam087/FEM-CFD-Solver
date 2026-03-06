import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

# =====================================================
# Geometry and Physical Constants
# =====================================================
L = 10
mid = L/2
rho = 1.225
u_L = 69.9854
A_L = 1.5
P0 = 3e5


# =====================================================
# Area Functions
# =====================================================
def area(x):

    A = np.zeros_like(x)

    left = x <= mid
    right = x > mid

    A[left] = 1 + 1.5*(1 - 0.2*x[left])**2
    A[right] = 1 + 0.5*(0.2*x[right] - 1)**2

    return A


def exact_velocity(x):
    return (A_L * u_L) / area(x)


# =====================================================
# Area Integrals
# =====================================================
def compute_I1(x1, x2):

    return (2.5*x2 - 3*x2**2/10 + x2**3/50
           -2.5*x1 + 3*x1**2/10 - x1**3/50)


def compute_I2(x1, x2):

    return (1.5*x2 - x2**2/10 + x2**3/150
           -1.5*x1 + x1**2/10 - x1**3/150)


# =====================================================
# FEM Solver
# =====================================================
def incomp_nozzle_phi(ne):

    nn = ne + 1
    x = np.linspace(0, L, nn)

    KG = np.zeros((nn, nn))
    RHS = np.zeros(nn)

    # Assembly
    for e in range(ne):

        node1 = e
        node2 = e+1

        x1 = x[node1]
        x2 = x[node2]

        Le = x2-x1

        if x2 <= mid:
            I = compute_I1(x1,x2)

        elif x1 >= mid:
            I = compute_I2(x1,x2)

        else:
            I = compute_I1(x1,mid) + compute_I2(mid,x2)

        k_local = (rho*I/Le**2) * np.array([[1,-1],[-1,1]])

        KG[np.ix_([node1,node2],[node1,node2])] += k_local


    # =====================================================
    # Dirichlet Boundary Condition
    # =====================================================
    bc_node = 0
    bc_value = 10

    k_diag = KG[bc_node, bc_node]

    KG[bc_node,:] = 0
    RHS -= KG[:,bc_node] * bc_value
    KG[:,bc_node] = 0

    KG[bc_node,bc_node] = k_diag
    RHS[bc_node] = bc_value * k_diag


    # =====================================================
    # Neumann Boundary Condition
    # =====================================================
    RHS[-1] = rho * A_L * u_L


    phi = np.linalg.solve(KG, RHS)

    return x, phi


# =====================================================
# Finite Difference Velocity
# =====================================================
def incomp_nozzle_u_FD(ne):

    x, phi = incomp_nozzle_phi(ne)

    u_FD = np.zeros_like(phi)

    for i in range(len(phi)):

        if i == 0:
            u_FD[i] = (phi[i+1]-phi[i])/(x[i+1]-x[i])

        elif i == len(phi)-1:
            u_FD[i] = (phi[i]-phi[i-1])/(x[i]-x[i-1])

        else:
            u_FD[i] = (phi[i+1]-phi[i-1])/(x[i+1]-x[i-1])

    return x, u_FD


# =====================================================
# FEM Velocity Reconstruction
# =====================================================
def incomp_nozzle_u_FEM(ne):

    x, phi = incomp_nozzle_phi(ne)

    nn = len(x)

    Kglob = np.zeros((nn,nn))
    RHS = np.zeros(nn)

    for e in range(nn-1):

        node1 = e
        node2 = e+1

        le = x[node2] - x[node1]

        RHS[node1] += (phi[node2]-phi[node1])/2
        RHS[node2] += (phi[node2]-phi[node1])/2

        KL = np.array([[le/3, le/6],
                       [le/6, le/3]])

        Kglob[np.ix_([node1,node2],[node1,node2])] += KL


    u_FEM = np.linalg.solve(Kglob,RHS)

    return x, u_FEM


# =====================================================
# Mesh Convergence Study
# =====================================================
if __name__ == "__main__":

    E = np.arange(10,110,10)

    phi_store = []
    x_store = []
    err_history = []

    for i, ne in enumerate(E):

        x, phi = incomp_nozzle_phi(ne)

        phi_store.append(phi)
        x_store.append(x)

        if i > 0:

            phi_old = phi_store[i-1]
            phi_new = phi

            mid_old = len(phi_old)//2
            mid_new = len(phi_new)//2

            delta_mid = phi_new[mid_new]/phi_old[mid_old]
            delta_end = phi_new[-1]/phi_old[-1]

            err_mid = 100*abs(delta_mid-1)
            err_end = 100*abs(delta_end-1)

            err_history.append(max(err_mid,err_end))


    # =====================================================
    # Plot φ(x)
    # =====================================================
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


    # =====================================================
    # Plot Error
    # =====================================================
    plt.figure(figsize=(7,5))

    plt.plot(E[1:], err_history,
             marker='o',
             linewidth=2)

    plt.xlabel("Number of Elements")
    plt.ylabel("Percentage Error (%)")
    plt.title("Mesh Convergence History")
    plt.grid(True)


    # =====================================================
    # Velocity Comparison
    # =====================================================
    x, velocity_FEM = incomp_nozzle_u_FEM(100)
    x, velocity_FD  = incomp_nozzle_u_FD(100)

    velocity_exact = exact_velocity(x)

    error_FEM = np.abs((velocity_FEM-velocity_exact)/velocity_exact)*100
    error_FD  = np.abs((velocity_FD-velocity_exact)/velocity_exact)*100


    # =====================================================
    # Mass Flow Rate
    # =====================================================
    A_vals = area(x)

    m_dot_FEM = rho*velocity_FEM*A_vals
    m_dot_FD  = rho*velocity_FD*A_vals
    m_dot_exact = rho*velocity_exact*A_vals


    # =====================================================
    # Pressure
    # =====================================================
    P_FEM = P0 - 0.5*rho*velocity_FEM**2
    P_FD  = P0 - 0.5*rho*velocity_FD**2


    # =====================================================
    # Velocity Plot
    # =====================================================
    plt.figure(figsize=(7,5))

    plt.plot(x, velocity_FEM, linewidth=2, label="FEM Galerkin")
    plt.plot(x, velocity_FD, linewidth=2, linestyle="--", label="Finite Difference")
    plt.plot(x, velocity_exact, linewidth=2, label="Exact")

    plt.xlabel("x")
    plt.ylabel("Velocity")
    plt.title("Velocity Comparison")
    plt.grid(True)
    plt.legend()


    # =====================================================
    # Error Plot
    # =====================================================
    plt.figure(figsize=(7,5))

    plt.plot(x, error_FEM, linewidth=2, label="FEM Error")
    plt.plot(x, error_FD, linewidth=2, linestyle="--", label="FD Error")

    plt.xlabel("x")
    plt.ylabel("Relative Error (%)")
    plt.title("Velocity Error Compared to Exact Solution")
    plt.grid(True)
    plt.legend()


    # =====================================================
    # Mass Flow Plot
    # =====================================================
    plt.figure(figsize=(7,5))

    plt.plot(x, m_dot_FEM, linewidth=2, label="FEM")
    plt.plot(x, m_dot_FD, linewidth=2, linestyle="--", label="Finite Difference")
    plt.plot(x, m_dot_exact, linewidth=2, color="black", label="Exact")

    plt.xlabel("x")
    plt.ylabel("Mass Flow Rate $\dot{m}$ (kg/s)")
    plt.title("Mass Flow Rate Variation Along the Nozzle")
    plt.grid(True)
    plt.legend()


    # =====================================================
    # Pressure Plot
    # =====================================================
    plt.figure(figsize=(7,5))

    plt.plot(x, P_FEM/1e5, linewidth=2, label="FEM")
    plt.plot(x, P_FD/1e5, linestyle="--", linewidth=2, label="Finite Difference")

    plt.xlabel("x")
    plt.ylabel("Pressure (bar)")
    plt.title("Pressure Distribution Along the Nozzle")
    plt.grid(True)
    plt.legend()


    plt.show()
    plt.close('all')