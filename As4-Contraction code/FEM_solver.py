import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


import os

output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

def plot_results(x_all, y_all, elements, psi, Vmag, u, v, D, case_name, output_dir):

    triangles = np.vstack([
        elements[:, [0, 1, 2]],
        elements[:, [0, 2, 3]]
    ])

    triang = mtri.Triangulation(x_all, y_all, triangles)

    # ===== Stream function =====
    plt.figure(figsize=(8,4.5))
    cf = plt.tricontourf(triang, psi, levels=200, cmap='turbo')
    plt.tricontour(triang, psi, levels=60, colors='k', linewidths=0.5)

    plt.colorbar(cf).set_label('Stream Function ψ')
    plt.title(f"Streamlines (ψ) - Case {case_name}")
    plt.axis('equal')
    plt.tight_layout()

    plt.savefig(f"{output_dir}/{case_name}_psi.png", dpi=300)
    plt.close()

    # ===== Velocity =====
    plt.figure(figsize=(8,4.5))
    cf = plt.tricontourf(triang, Vmag, levels=100, cmap='turbo')

    plt.colorbar(cf).set_label('Velocity Magnitude')
    plt.title(f"Velocity - Case {case_name}")
    plt.axis('equal')
    plt.tight_layout()

    plt.savefig(f"{output_dir}/{case_name}_velocity.png", dpi=300)
    plt.close()

    # ===== Divergence =====
    plt.figure(figsize=(8,4.5))
    cf = plt.tricontourf(triang, D, levels=100, cmap='viridis')

    plt.colorbar(cf).set_label('Divergence')
    plt.title(f"Divergence - Case {case_name}")
    plt.axis('equal')
    plt.tight_layout()

    plt.savefig(f"{output_dir}/{case_name}_divergence.png", dpi=300)
    plt.close()

def run_case(case_type):

    # =========================


    # Load mesh data
    # =========================
    nodes_data = np.loadtxt('slected_mesh/Nodes.txt', skiprows=1)
    elements_data = np.loadtxt('slected_mesh/Elements.txt', skiprows=1)

    # Extract nodes
    node_ids = nodes_data[:, 0].astype(int)
    x_all = nodes_data[:, 1]
    y_all = nodes_data[:, 2]

    # Extract elements (ignore element number column)
    elements = elements_data[:, 1:].astype(int) - 1  # zero-based indexing

    num_nodes = len(node_ids)
    num_elem = elements.shape[0]

    # Normalize coordinates
    d = np.max(y_all)
    x_all = x_all / d
    y_all = y_all / d

    # =========================
    # PDE parameter
    # =========================
    if case_type == 'a':
        C = 0
    elif case_type == 'b':
        C = -(np.pi**2) / 4

    # =========================
    # Initialize global system
    # =========================
    K_global = np.zeros((num_nodes, num_nodes))
    M_global = np.zeros((num_nodes, num_nodes))
    F_global = np.zeros(num_nodes)

    Gx_global = np.zeros((num_nodes, num_nodes))
    Gy_global = np.zeros((num_nodes, num_nodes))

    # =========================
    # Gauss points
    # =========================
    gp = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    w  = np.array([1, 1])

    # =========================
    # Assembly loop
    # =========================
    for e in range(num_elem):

        conn = elements[e, :]
        x = x_all[conn]
        y = y_all[conn]

        Ke = np.zeros((4, 4)) # stiffnes matrix for the elelemt 
        Me = np.zeros((4, 4)) # Mass matrix for the element
        Gxe = np.zeros((4, 4)) # Gradient matrix in x
        Gye = np.zeros((4, 4)) # Gradient matrix in y

        for i in range(2):
            for j in range(2):

                xi = gp[i]
                eta = gp[j]

                # Shape functions
                N = 0.25 * np.array([
                    (1-xi)*(1-eta),
                    (1+xi)*(1-eta),
                    (1+xi)*(1+eta),
                    (1-xi)*(1+eta)
                ])

                # Derivatives
                dN_dxi = 0.25 * np.array([
                    -(1-eta),
                    (1-eta),
                    (1+eta),
                    -(1+eta)
                ])

                dN_deta = 0.25 * np.array([
                    -(1-xi),
                    -(1+xi),
                    (1+xi),
                    (1-xi)
                ])

                # Jacobian
                J = np.vstack((dN_dxi, dN_deta)) @ np.column_stack((x, y)) # This maps reference elelent to real element
                detJ = np.linalg.det(J)

                # Derivatives in physical space
                dN = np.linalg.solve(J, np.vstack((dN_dxi, dN_deta)))
                dN_dx = dN[0, :]
                dN_dy = dN[1, :]

                B = np.vstack((dN_dx, dN_dy))

                # Element matrices
                Ke += (B.T @ B + C * np.outer(N, N)) * detJ * w[i] * w[j]
                Me += np.outer(N, N) * detJ * w[i] * w[j]
                Gxe += np.outer(N, dN_dx) * detJ * w[i] * w[j]
                Gye += np.outer(N, dN_dy) * detJ * w[i] * w[j]

        # Assembly
        for a in range(4):
            for b in range(4):
                K_global[conn[a], conn[b]] += Ke[a, b]
                M_global[conn[a], conn[b]] += Me[a, b]
                Gx_global[conn[a], conn[b]] += Gxe[a, b]
                Gy_global[conn[a], conn[b]] += Gye[a, b]

    # =========================
    # Boundary Conditions
    # =========================
    H1 = d
    H2 = d / 10
    x_step = 2 * d

    psi = np.zeros(num_nodes)
    tol = 1e-8

    for i in range(num_nodes):

        x = x_all[i] * d
        y = y_all[i] * d

        # Bottom symmetry
        if abs(y) < tol:
            K_global[i, :] = 0
            K_global[i, i] = 1
            F_global[i] = 0
            continue

        # Top wall before contraction
        if (x <= x_step) and (abs(y - H1) < tol):
            K_global[i, :] = 0
            K_global[i, i] = 1
            F_global[i] = 1
            continue

        # Vertical step wall
        if (abs(x - x_step) < tol) and (H2 <= y <= H1):
            K_global[i, :] = 0
            K_global[i, i] = 1
            F_global[i] = 1
            continue

        # Small channel top wall
        if (x > x_step) and (abs(y - H2) < tol):
            K_global[i, :] = 0
            K_global[i, i] = 1
            F_global[i] = 1
            continue

        # Inlet
        if abs(x) < tol:
            if case_type == 'a':
                psi_val = y / d
            elif case_type == 'b':
                psi_val = np.sin(np.pi/2 * (y/d))

            K_global[i, :] = 0
            K_global[i, i] = 1
            F_global[i] = psi_val

    # =========================
    # Solve system
    # =========================
    psi = np.linalg.solve(K_global, F_global)

    # =========================
    # Velocity (Galerkin FEM)
    # =========================
    u = np.linalg.solve(M_global, Gy_global @ psi)
    v = np.linalg.solve(M_global, -Gx_global @ psi)

    Vmag = np.sqrt(u**2 + v**2)
    D = np.linalg.solve(M_global, Gx_global @ u + Gy_global @ v)

    plot_results(x_all, y_all, elements, psi, Vmag, u, v, D, case_type, output_dir)

for case in ['a', 'b']:
    print(f"Running case {case}...")
    run_case(case)