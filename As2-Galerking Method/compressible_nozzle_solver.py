import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# =====================================================
# Geometry
# =====================================================

L = 10
mid = L/2

# =====================================================
# Gas Properties
# =====================================================

gamma = 1.4
R = 287

rho0 = 1.225
P0 = 3e5
T0 = P0/(rho0*R)

# =====================================================
# Convergence Parameters
# =====================================================

tol = 1e-6
max_iter = 5000


# =====================================================
# Area Function
# =====================================================

def area(x):

    A = np.zeros_like(x)

    left = x <= mid
    right = x > mid

    A[left] = 1 + 1.5*(1 - 0.2*x[left])**2
    A[right] = 1 + 0.5*(0.2*x[right] - 1)**2

    return A


# =====================================================
# Area Integrals
# =====================================================

def compute_I1(x1,x2):

    return (2.5*x2 - 3*x2**2/10 + x2**3/50
           -2.5*x1 + 3*x1**2/10 - x1**3/50)


def compute_I2(x1,x2):

    return (1.5*x2 - x2**2/10 + x2**3/150
           -1.5*x1 + x1**2/10 - x1**3/150)


# =====================================================
# Compressible FEM Solver
# =====================================================

def compressible_nozzle(ne, bc_exit):

    nn = ne + 1
    x = np.linspace(0,L,nn)

    le = x[1] - x[0]

    rho = np.ones(nn)*rho0

    err_L2 = 1
    iteration = 0

    while err_L2 > tol and iteration < max_iter:

        iteration += 1

        KG = np.zeros((nn,nn))
        RHS = np.zeros(nn)

        # -----------------------------------------
        # Assembly
        # -----------------------------------------

        for e in range(ne):

            node1 = e
            node2 = e+1

            x1 = x[node1]
            x2 = x[node2]

            rho_mean = 0.5*(rho[node1] + rho[node2])

            if x2 <= mid:
                I = compute_I1(x1,x2)

            elif x1 >= mid:
                I = compute_I2(x1,x2)

            else:
                I = compute_I1(x1,mid) + compute_I2(mid,x2)

            K_val = rho_mean*I/(le**2)

            KL = np.array([[K_val,-K_val],
                           [-K_val,K_val]])

            KG[np.ix_([node1,node2],[node1,node2])] += KL


        # -----------------------------------------
        # Boundary Conditions
        # -----------------------------------------

        BN = [0, nn-1]
        BV = [0, bc_exit]

        for i in range(2):

            node = BN[i]
            value = BV[i]

            K_diag = KG[node,node]

            KG[node,:] = 0
            RHS -= KG[:,node]*value
            KG[:,node] = 0

            KG[node,node] = K_diag
            RHS[node] = value*K_diag


        # -----------------------------------------
        # Solve Potential
        # -----------------------------------------

        phi = np.linalg.solve(KG,RHS)


        # -----------------------------------------
        # Velocity Reconstruction (FEM)
        # -----------------------------------------

        Kglob = np.zeros((nn,nn))
        RHS = np.zeros(nn)

        for e in range(ne):

            node1 = e
            node2 = e+1

            RHS[node1] += (phi[node2]-phi[node1])/2
            RHS[node2] += (phi[node2]-phi[node1])/2

            KL = np.array([[le/3, le/6],
                           [le/6, le/3]])

            Kglob[np.ix_([node1,node2],[node1,node2])] += KL

        u = np.linalg.solve(Kglob,RHS)


        # -----------------------------------------
        # Mach Number Calculation
        # -----------------------------------------

        a0 = np.sqrt(gamma*R*T0)

        Mach0 = u/a0

        Mach = np.sqrt(Mach0**2/(1 - (gamma-1)/2*Mach0**2))


        # -----------------------------------------
        # Density Update (Isentropic)
        # -----------------------------------------

        rho_new = rho0*(1 - (gamma-1)/2*Mach0**2)**(1/(gamma-1))


        # -----------------------------------------
        # Error Calculation
        # -----------------------------------------

        Ri = 2*(rho_new-rho)/(rho_new+rho)

        err_L2 = np.sum(Ri**2)

        rho = rho_new


    print("Iterations =", iteration)

    return x,phi,u,rho,Mach


# =====================================================
# Grid Convergence Study (on φ)
# =====================================================

E = np.arange(10,110,10)

phi_store = []
x_store = []
err_history = []

for i, ne in enumerate(E):

    x, phi, u, rho, Mach = compressible_nozzle(ne,3000)

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

    plt.plot(x_store[i],
             phi_store[i],
             linewidth=1.5,
             label=f"E = {E[i]}")

plt.xlabel("x")
plt.ylabel("Velocity Potential φ")
plt.title("Mesh Convergence of φ")
plt.grid(True)
plt.legend()


# =====================================================
# Error Plot
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
# Mach Number for Different Exit Potentials
# =====================================================

phi_exit_values = [500,1000,2000,3000,3390.75]

plt.figure(figsize=(7,5))

for bc in phi_exit_values:

    x, phi, u, rho, Mach = compressible_nozzle(100, bc)

    plt.plot(x, Mach, linewidth=2, label=f"$\phi(L)$ = {bc}")

plt.axhline(1, linestyle="--", color="black")

plt.xlabel("x (m)")
plt.ylabel("Mach Number")
plt.title("Mach Number Distribution for Different Exit Potentials")
plt.grid(True)
plt.legend()


# =====================================================
# Thermodynamic Properties (E=100, φ(L)=3000)
# =====================================================

x, phi, u, rho, Mach = compressible_nozzle(100,3000)

T = T0*(1 - (gamma-1)/2*Mach**2)
P = rho*R*T


plt.figure(figsize=(7,5))
plt.plot(x,P/1e5,linewidth=2)
plt.xlabel("x")
plt.ylabel("Pressure (bar)")
plt.title("Pressure Distribution")
plt.grid(True)


plt.figure(figsize=(7,5))
plt.plot(x,T,linewidth=2)
plt.xlabel("x")
plt.ylabel("Temperature (K)")
plt.title("Temperature Distribution")
plt.grid(True)


plt.show()