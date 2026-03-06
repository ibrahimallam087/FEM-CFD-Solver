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
T0 = 850
rho0 = 1.225

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
# FEM Velocity Reconstruction
# =====================================================
def velocity_FEM(x,phi):

    nn = len(x)

    K = np.zeros((nn,nn))
    RHS = np.zeros(nn)

    for e in range(nn-1):

        n1 = e
        n2 = e+1

        le = x[n2]-x[n1]

        RHS[n1] += (phi[n2]-phi[n1])/2
        RHS[n2] += (phi[n2]-phi[n1])/2

        KL = np.array([[le/3,le/6],
                       [le/6,le/3]])

        K[np.ix_([n1,n2],[n1,n2])] += KL

    u = np.linalg.solve(K,RHS)

    return u


# =====================================================
# Compressible FEM Solver
# =====================================================
def compressible_solver(ne,phi_exit):

    nn = ne+1
    x = np.linspace(0,L,nn)

    rho = np.ones(nn)*rho0

    tol = 1e-6
    error = 1

    while error > tol:

        KG = np.zeros((nn,nn))
        RHS = np.zeros(nn)

        for e in range(ne):

            n1 = e
            n2 = e+1

            x1 = x[n1]
            x2 = x[n2]

            le = x2-x1

            rho_mean = 0.5*(rho[n1]+rho[n2])

            if x2 <= mid:
                I = compute_I1(x1,x2)
            elif x1 >= mid:
                I = compute_I2(x1,x2)
            else:
                I = compute_I1(x1,mid)+compute_I2(mid,x2)

            k_local = (rho_mean*I/le**2)*np.array([[1,-1],[-1,1]])

            KG[np.ix_([n1,n2],[n1,n2])] += k_local


        # Dirichlet BC
        BC_nodes = [0,nn-1]
        BC_vals = [0,phi_exit]

        K = KG.copy()

        for node,val in zip(BC_nodes,BC_vals):

            kdiag = K[node,node]

            K[node,:]=0
            RHS -= K[:,node]*val
            K[:,node]=0

            K[node,node]=kdiag
            RHS[node]=val*kdiag

        phi = np.linalg.solve(K,RHS)


        # velocity
        u = velocity_FEM(x,phi)


        # Mach calculations
        a0 = np.sqrt(gamma*R*T0)

        M0 = u/a0

        M = np.sqrt(M0**2/(1+(gamma-1)/2*M0**2))


        # density update
        rho_new = rho0*(1-((gamma-1)/2)*M0**2)**(1/(gamma-1))

        Ri = 2*(rho_new-rho)/(rho_new+rho)

        error = np.sum(Ri**2)

        rho = rho_new

    return x,phi,u,M,rho


# =====================================================
# 1) Solve φ(x) for E=100
# =====================================================
x,phi,u,M,rho = compressible_solver(100,3000)

plt.figure()
plt.plot(x,phi)
plt.xlabel("x (m)")
plt.ylabel("Velocity Potential φ")
plt.title("Velocity Potential Distribution")
plt.grid()


# =====================================================
# 2) Grid Convergence Study
# =====================================================
E = np.arange(10,110,10)

phi_mid = []

for ne in E:

    x,phi,u,M,rho = compressible_solver(ne,3000)

    phi_mid.append(phi[len(phi)//2])

error = []

for i in range(1,len(phi_mid)):

    err = abs((phi_mid[i]-phi_mid[i-1])/phi_mid[i-1])*100

    error.append(err)

plt.figure()

plt.plot(E[1:],error,'o-')

plt.xlabel("Number of Elements")
plt.ylabel("Error (%)")
plt.title("Grid Convergence Study")
plt.grid()


# =====================================================
# 3) Pressure and Temperature
# =====================================================
T = T0*(1 + (gamma-1)/2*M**2)**(-1)

P = rho*R*T

plt.figure()
plt.plot(x,P)
plt.xlabel("x (m)")
plt.ylabel("Pressure (Pa)")
plt.title("Pressure Distribution")
plt.grid()

plt.figure()
plt.plot(x,T)
plt.xlabel("x (m)")
plt.ylabel("Temperature (K)")
plt.title("Temperature Distribution")
plt.grid()


# =====================================================
# 4) Choking Study
# =====================================================
phi_vals = [500,1000,2000,3000,3390.75]

plt.figure()

for val in phi_vals:

    x,phi,u,M,rho = compressible_solver(100,val)

    plt.plot(x,M,label=f"φ(L) = {val}")

plt.axhline(1,linestyle="--")

plt.xlabel("x (m)")
plt.ylabel("Mach Number")
plt.title("Mach Number Variation Along Nozzle")
plt.legend()
plt.grid()

plt.show()