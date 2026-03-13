import sympy as sp
import numpy as np

# symbolic variables
xi, eta = sp.symbols('xi eta')

# ------------------------------------------------
# Node coordinates
# ------------------------------------------------

x_nodes = sp.Matrix([0.2,0.4,0.35,0.1])
y_nodes = sp.Matrix([0.2,0.4,0.5,0.6])

# ------------------------------------------------
# Shape function coefficient matrix
# ------------------------------------------------

A = sp.Matrix([
[1,-1,-1,1],
[1,1,-1,-1],
[1,1,1,1],
[1,-1,1,-1]
])

coeff = A.T/4

# ------------------------------------------------
# Shape functions
# ------------------------------------------------

normal = sp.Matrix([1,xi,eta,xi*eta])

N = []
N_xi = []
N_eta = []

for i in range(4):

    Ni = coeff[:,i].dot(normal)

    N.append(sp.simplify(Ni))
    N_xi.append(sp.diff(Ni,xi))
    N_eta.append(sp.diff(Ni,eta))

# ------------------------------------------------
# Evaluate x,y mapping
# ------------------------------------------------

xi_val = 0.5
eta_val = 0.5

Ni_val = [Ni.subs({xi:xi_val,eta:eta_val}) for Ni in N]

x = sum(x_nodes[i]*Ni_val[i] for i in range(4))
y = sum(y_nodes[i]*Ni_val[i] for i in range(4))

print("x =",float(x))
print("y =",float(y))

# ------------------------------------------------
# Jacobian terms
# ------------------------------------------------

x_xi = sum(x_nodes[i]*N_xi[i] for i in range(4))
y_xi = sum(y_nodes[i]*N_xi[i] for i in range(4))

x_eta = sum(x_nodes[i]*N_eta[i] for i in range(4))
y_eta = sum(y_nodes[i]*N_eta[i] for i in range(4))

J = sp.Matrix([[x_xi,y_xi],[x_eta,y_eta]])

J_inv = J.inv()
J_det = sp.simplify(J.det())

# ------------------------------------------------
# Derivatives with respect to x,y
# ------------------------------------------------

N_x = []
N_y = []

for i in range(4):

    temp = J_inv*sp.Matrix([N_xi[i],N_eta[i]])

    N_x.append(sp.simplify(temp[0]))
    N_y.append(sp.simplify(temp[1]))

# evaluate N1x N1y
N1x = N_x[0].subs({xi:0.5,eta:0.5})
N1y = N_y[0].subs({xi:0.5,eta:0.5})

print("N1x =",float(N1x))
print("N1y =",float(N1y))

# ------------------------------------------------
# 2x2 Gauss quadrature points
# ------------------------------------------------

gp = [-1/np.sqrt(3),1/np.sqrt(3)]
w = [1,1]

k = np.zeros((4,4))

for i in range(4):
    for j in range(4):

        integral = 0

        for a in range(2):
            for b in range(2):

                xi_g = gp[a]
                eta_g = gp[b]

                val = (N_x[i]*N_x[j] + N_y[i]*N_y[j])*J_det
                val = val.subs({xi:xi_g,eta:eta_g})

                integral += w[a]*w[b]*float(val)

        k[i,j] = integral

print("k matrix =")
print(np.round(k,4))