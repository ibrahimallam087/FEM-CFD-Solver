import numpy as np
import matplotlib.pyplot as plt
import os

# =============================
# Output folder
# =============================
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)


# =============================
# Exact solution
# =============================
def exact_solution(x, Pe):
    return 5 * (np.exp(Pe*x) - 1) / (np.exp(Pe) - 1)


# =============================
# Mesh
# =============================
def uniform_mesh(N):
    return np.linspace(0, 1, N+1)


def geometric_mesh(N, r):
    if abs(r - 1.0) < 1e-10:
        return uniform_mesh(N)

    x = np.zeros(N+1)
    L1 = (1 - r) / (1 - r**N)

    for i in range(1, N+1):
        x[i] = x[i-1] + L1 * r**(i-1)

    return 1 - x[::-1]


# =============================
# FEM Solver
# =============================
def fem_solver(x, Pe, method='SG'):

    N = len(x) - 1
    K = np.zeros((N+1, N+1))
    F = np.zeros(N+1)

    for e in range(N):
        Le = x[e+1] - x[e]

        K_diff = (1/Le) * np.array([[1,-1],[-1,1]])
        K_conv = (Pe/2) * np.array([[-1,1],[-1,1]])

        Ke = K_diff + K_conv

        if method == 'GLS':
            tau = Le / (2*np.sqrt(Pe**2 + (4/Le**2)))
            Ke += tau * Pe**2 * K_diff

        K[e:e+2, e:e+2] += Ke

    # BCs
    K[0,:]=0; K[0,0]=1; F[0]=0
    K[-1,:]=0; K[-1,-1]=1; F[-1]=5

    return x, np.linalg.solve(K, F)


# =============================
# Utilities
# =============================
def avg_temp(x, T):
    return np.trapz(T, x)


def compute_error(x, T, Pe):
    xd = np.linspace(0,1,1000)
    return np.sqrt(np.trapz((np.interp(xd,x,T)-exact_solution(xd,Pe))**2, xd))


# =============================
# GCI (FIXED)
# =============================
def gci_analysis(method, Pe):

    N_values = [50, 100, 200, 400, 800]

    phi = []
    h = []

    for N in N_values:
        x = uniform_mesh(N)
        _, T = fem_solver(x, Pe, method)

        phi.append(avg_temp(x, T))
        h.append(1/N)

    phi1, phi2, phi3 = phi[-1], phi[-2], phi[-3]
    r = h[-2] / h[-1]

    eps = 1e-14
    num = abs(phi3 - phi2)
    den = abs(phi2 - phi1)

    p = 1.0 if (num < eps or den < eps) else np.log(num/den)/np.log(r)

    Fs = 1.25
    GCI = Fs * abs((phi1 - phi2)/(phi1*(r**p - 1))) * 100

    # exact average
    xd = np.linspace(0,1,2000)
    phi_exact = np.trapz(exact_solution(xd, Pe), xd)
    err = abs((phi1 - phi_exact)/phi_exact)*100

    return {"GCI": GCI, "p": p, "true_error": err, "N_opt": N_values[-1]}


# =============================
# PART 1: SG vs GLS
# =============================
Pe_values = [1, 10, 150, 500]

for N in [10, 100]:
    for Pe in Pe_values:

        x = uniform_mesh(N)
        _, T_sg = fem_solver(x, Pe, 'SG')
        _, T_gls = fem_solver(x, Pe, 'GLS')

        xd = np.linspace(0,1,500)

        plt.figure()
        plt.plot(x, T_sg, 'o-', label='SG')
        plt.plot(x, T_gls, 's-', label='GLS')
        plt.plot(xd, exact_solution(xd,Pe), 'k--', label='Exact')
        plt.legend(); plt.grid()
        plt.title(f'N={N}, Pe={Pe}')

        plt.savefig(f"{output_dir}/solution_N{N}_Pe{Pe}.png")
        plt.close()


# =============================
# PART 2: Clustering
# =============================
Pe = 500
N = 10

r_vals = np.linspace(1.0, 2.0, 25)
errors = []

for r in r_vals:
    x = geometric_mesh(N, r)
    _, T = fem_solver(x, Pe, 'GLS')
    errors.append(compute_error(x,T,Pe))

best_r = r_vals[np.argmin(errors)]

plt.figure()
plt.plot(r_vals, errors, '-o')
plt.xlabel('r'); plt.ylabel('Error')
plt.grid()
plt.savefig(f"{output_dir}/clustering_error.png")
plt.close()


# =============================
# PART 3: Compare clustering
# =============================
x_uni = uniform_mesh(N)
x_cl = geometric_mesh(N, best_r)

_, T_uni = fem_solver(x_uni, Pe, 'GLS')
_, T_cl = fem_solver(x_cl, Pe, 'GLS')

xd = np.linspace(0,1,500)

plt.figure()
plt.plot(x_uni,T_uni,'o-',label='Uniform')
plt.plot(x_cl,T_cl,'s-',label='Clustered')
plt.plot(xd,exact_solution(xd,Pe),'k--',label='Exact')
plt.legend(); plt.grid()
plt.savefig(f"{output_dir}/clustering_compare.png")
plt.close()


# =============================
# PART 4: Local Peclet
# =============================
plt.figure()

for Pe in [1,10,100,500]:
    Pe_loc = [(x_cl[i+1]-x_cl[i])*Pe/2 for i in range(N)]
    plt.plot(range(1,N+1),Pe_loc,'o-',label=f'Pe={Pe}')

plt.legend(); plt.grid()
plt.savefig(f"{output_dir}/local_Pe.png")
plt.close()


# =============================
# PART 5: GCI
# =============================
Pe = 500

sg = gci_analysis('SG', Pe)
gls = gci_analysis('GLS', Pe)

print("\nGCI ANALYSIS (Pe=500)\n")
print("{:<10} {:<10} {:<10} {:<10} {:<15}".format(
    "Method","GCI","p","N","Error"))

print("{:<10} {:<10.4f} {:<10.2f} {:<10} {:<15.4f}".format(
    "SG", sg["GCI"], sg["p"], sg["N_opt"], sg["true_error"]))

print("{:<10} {:<10.4f} {:<10.2f} {:<10} {:<15.4f}".format(
    "GLS", gls["GCI"], gls["p"], gls["N_opt"], gls["true_error"]))