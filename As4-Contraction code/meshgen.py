import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# =========================
# Geometry
# =========================
d = 1
L  = 3 * d
H1 = d
cont_ratio = 1 / 10
H2 = cont_ratio * H1       # H1 / 10
x_remaining = 1 / 3 * L   # L / 3
x_step = L - x_remaining

# =========================
# Grid
# =========================
E_x = 90
E_y = 60
nx = E_x + 1
ny = E_y + 1
x = np.linspace(0, L, nx)
y = np.linspace(0, H1, ny)

# =========================
# Nodes
# =========================
nodes = np.zeros((nx * ny, 2))
idx = 0
for j in range(ny):
    for i in range(nx):
        nodes[idx, :] = [x[i], y[j]]
        idx += 1

# =========================
# Connectivity
# =========================
elements = []
for j in range(E_y):
    for i in range(E_x):
        n1 = i + j * nx          # 0-based
        n2 = n1 + 1
        n3 = n2 + nx
        n4 = n3 - 1

        # center of element
        xc = (x[i] + x[i + 1]) / 2
        yc = (y[j] + y[j + 1]) / 2

        # remove top-right region
        if (xc > x_step) and (yc > H2):
            continue

        elements.append([n1, n2, n3, n4])

elements = np.array(elements, dtype=int)

# =========================
# Remove unused nodes
# =========================
# Get unique node IDs used in elements
used_nodes = np.unique(elements)

# Create mapping from old node IDs -> new node IDs (0-based)
new_id = np.zeros(nodes.shape[0], dtype=int)
new_id[used_nodes] = np.arange(len(used_nodes))

# Update nodes (keep only used ones)
nodes = nodes[used_nodes, :]

# Update element connectivity with new numbering
elements = new_id[elements]

# =========================
# Plot
# =========================
fig, ax = plt.subplots()
ax.set_aspect('equal')

for e in range(elements.shape[0]):
    pts = nodes[elements[e, :], :]
    poly = plt.Polygon(pts, closed=True, facecolor='white', edgecolor='black', linewidth=0.3)
    ax.add_patch(poly)

ax.autoscale_view()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('FINAL CORRECT STEP MESH')
ax.grid(True)
plt.tight_layout()
plt.show()

# =========================
# Export elements (1-based node IDs)
# =========================
with open('Elements.txt', 'w') as f:
    f.write('Element_Number\tNodes\t\t\t\n')
    for i in range(elements.shape[0]):
        n1, n2, n3, n4 = elements[i] + 1   # convert to 1-based
        f.write(f'{i+1}\t{n1}\t{n2}\t{n3}\t{n4}\n')

# =========================
# Export nodes (1-based, with Z=0)
# =========================
with open('Nodes.txt', 'w') as f:
    f.write('Node Number\tX Location (m)\tY Location (m)\tZ Location (m)\n')
    for i in range(nodes.shape[0]):
        f.write(f'{i+1}\t{nodes[i,0]}\t{nodes[i,1]}\t0\n')