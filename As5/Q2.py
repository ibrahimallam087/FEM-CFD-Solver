import numpy as np
import matplotlib.pyplot as plt

# Domain
x = np.linspace(0, 2, 600)

# Extend C range to ensure full coverage
C_values = np.linspace(-1.5, 4.5, 30)

# Create figure
plt.figure()

for C in C_values:
    # Two families
    y1 = (2/3) * x**(3/2) + C
    y2 = -(2/3) * x**(3/2) + C

    # Keep only inside domain
    y1[(y1 < 0) | (y1 > 2)] = np.nan
    y2[(y2 < 0) | (y2 > 2)] = np.nan

    # Plot
    plt.plot(x, y1, 'c', linewidth=1)
    plt.plot(x, y2, 'k', linewidth=1)

# Axes limits
plt.xlim(0, 2)
plt.ylim(0, 2)

# Labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Characteristic Net for Euler-Tricomi')

# Formatting
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')

# Optional box styling similar to MATLAB
for spine in plt.gca().spines.values():
    spine.set_visible(True)

plt.show()

