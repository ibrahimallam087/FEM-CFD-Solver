import numpy as np
import os

def plot_mesh_comparison(mesh_data_list, titles):

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    # =========================
    # Create figure
    # =========================
    fig, axes = plt.subplots(1, len(mesh_data_list), figsize=(15, 4.5))

    # =========================
    # Global min/max (same color scale)
    # =========================
    all_fields = [data["field"] for data in mesh_data_list]
    vmin = min(np.min(f) for f in all_fields)
    vmax = max(np.max(f) for f in all_fields)

    # =========================
    # Loop over meshes
    # =========================
    for ax, data, title in zip(axes, mesh_data_list, titles):

        x_all = data["x"]
        y_all = data["y"]
        elements = data["elements"]
        field = data["field"]

        # Convert quad → triangles
        triangles = np.vstack([
            elements[:, [0, 1, 2]],
            elements[:, [0, 2, 3]]
        ])

        triang = mtri.Triangulation(x_all, y_all, triangles)

        # Filled contour
        cf = ax.tricontourf(
            triang,
            field,
            levels=100,
            cmap='turbo',
            vmin=vmin,
            vmax=vmax
        )

        # Mesh overlay
        ax.triplot(triang, color='k', linewidth=0.2, alpha=0.3)

        # Labels
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

    # =========================
    # 🔥 Proper horizontal colorbar (fixed)
    # =========================
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.05])  
    # [left, bottom, width, height]

    cbar = fig.colorbar(cf, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Velocity Magnitude')

    # =========================
    # Layout spacing
    # =========================
    plt.subplots_adjust(bottom=0.25, wspace=0.3)

    plt.show()

def run_case(x_all, y_all, elements, case_type):


    d = np.max(y_all)
    x_all = x_all / d
    y_all = y_all / d
    num_nodes = len(x_all)
    num_elem = elements.shape[0]
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

    return psi, u, v, Vmag, D



# =========================
# Mesh directories (EDIT IF NEEDED)
# =========================
mesh_dirs = [
    "mesh1",
    "mesh2",
    "mesh3"
]

case_type = 'b'

results = []
N_values = []
mesh_data_list = []
titles = []
# =========================
# Loop over meshes
# =========================
for mesh_dir in mesh_dirs:


    print(f"\nProcessing {mesh_dir}...")

    nodes_path = os.path.join(mesh_dir, "Nodes.txt")
    elements_path = os.path.join(mesh_dir, "Elements.txt")

    # -------------------------
    # Load mesh
    # -------------------------
    nodes_data = np.loadtxt(nodes_path, skiprows=1)
    elements_data = np.loadtxt(elements_path, skiprows=1)

    x_all = nodes_data[:, 1]
    y_all = nodes_data[:, 2]

    elements = elements_data[:, 1:].astype(int) - 1

    N = len(x_all)
    N_values.append(N)

    # -------------------------
    # Run solver
    # -------------------------
    psi, u, v, Vmag, D = run_case(x_all, y_all, elements, case_type)
    mesh_data_list.append({
    "x": x_all,
    "y": y_all,
    "elements": elements,
    "field": Vmag
    })

    titles.append(mesh_dir)

    # =========================
    # Vortex center detection (ROBUST)
    # =========================

    # Define domain bounds
    x_min, x_max = np.min(x_all), np.max(x_all)
    y_min, y_max = np.min(y_all), np.max(y_all)

    # Define tolerance (5% of domain size)
    tol_x = 0.05 * (x_max - x_min)
    tol_y = 0.05 * (y_max - y_min)

    # Mask: exclude boundaries
    mask = (
        (x_all > x_min + tol_x) &
        (x_all < x_max - tol_x) &
        (y_all > y_min + tol_y) &
        (y_all < y_max - tol_y)
    )

    # Copy psi and invalidate boundary nodes
    psi_masked = psi.copy()

    # Set boundary values to very small number (so they are ignored)
    psi_masked[~mask] = -1e20

    # Find vortex center (maximum ψ)
    idx = np.argmax(psi_masked)

    x_vortex = x_all[idx]
    y_vortex = y_all[idx]

    print(f"Vortex center: x = {x_vortex:.4f}, y = {y_vortex:.4f}")

    # Monitoring variable
    f = y_vortex

    results.append(f)
plot_mesh_comparison(mesh_data_list, titles)
# =========================
# Extract values
# =========================
f1, f2, f3 = results
N1, N2, N3 = N_values

D_dim = 2

# =========================
# Compute refinement ratio
# =========================
r = (N2 / N1)**(1 / D_dim)

# =========================
# Compute order of accuracy
# =========================
p = np.log((f3 - f2) / (f2 - f1)) / np.log(r)

# =========================
# Compute GCI
# =========================
Fs = 1.25
epsilon_21 = (f2 - f1) / f2

GCI_21 = Fs * abs(epsilon_21) / (r**p - 1)

# =========================
# Print results
# =========================
print("\n===== GRID CONVERGENCE =====")
print(f"f1 = {f1}")
print(f"f2 = {f2}")
print(f"f3 = {f3}")

print(f"\nr = {r:.4f}")
print(f"p = {p:.4f}")
print(f"GCI_21 = {GCI_21:.6f}")
f_exact = f2 + (f2 - f1) / (r**p - 1)

print(f"\nRichardson extrapolated value = {f_exact:.6f}")