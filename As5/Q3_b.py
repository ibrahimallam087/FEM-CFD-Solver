import numpy as np
import matplotlib.pyplot as plt

# Initial positions and time array
x0 = np.linspace(-7, 7, 40)
t = np.linspace(0, 6, 200)   # smoother than MATLAB's 40 points

# Compute the characteristic speed for every initial point
u0 = 1 / (1 + x0**2)

# Use broadcasting to generate all characteristic lines at once
# Each row corresponds to one characteristic
X = x0[:, None] + u0[:, None] * t[None, :]

# Plot
fig, ax = plt.subplots(figsize=(6, 5))

ax.plot(X.T, t, color='tab:blue', linewidth=1.5)

ax.set(
    xlabel='x',
    ylabel='t',
    title='Characteristic Lines for Burgers Equation',
    xlim=(-7, 7)
)

ax.grid(True, alpha=0.3)

# Match MATLAB's clean appearance
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)

plt.tight_layout()
plt.show()

# Export to PDF if needed:
# fig.savefig("characteristics_Q3b.pdf", dpi=300, bbox_inches="tight")