import numpy as np
import matplotlib.pyplot as plt

# Parameter r
r = np.linspace(-5, 5, 1000)

# Shock time
ts = 8 / (3 * np.sqrt(3))

times = [0, 0.5 * ts, ts, 1.5 * ts]
labels = [r"$t=0$", r"$t=0.5\,t_s$", r"$t=t_s$", r"$t=1.5\,t_s$"]

# MATLAB-like color order but closer to your reference figure
colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E']

u0 = 1 / (1 + r**2)

fig, ax = plt.subplots(figsize=(7, 5))

for t, label, color in zip(times, labels, colors):
    x = r + u0 * t
    ax.plot(x, u0, color=color, linewidth=2.0, label=label)

ax.set_xlabel("x", fontsize=13)
ax.set_ylabel("u", fontsize=13)
ax.set_title("Evolution of the Burgers Solution", fontsize=15, weight='bold')

ax.set_xlim(-6, 6)
ax.set_ylim(0, 1.02)

ax.grid(True, alpha=0.3)
ax.legend(frameon=True, fontsize=11)

plt.tight_layout()
plt.show()