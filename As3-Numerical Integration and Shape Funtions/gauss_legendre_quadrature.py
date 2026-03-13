import numpy as np
import sympy as sp

# -------------------------------------------------
# Define symbolic variable
# -------------------------------------------------
x = sp.symbols('x')

# Define the function (symbolic form)
f_sym = x**3   

# Numerical version of the same function
f = sp.lambdify(x, f_sym, 'numpy')


# -------------------------------------------------
# Mapping from zeta → x
# -------------------------------------------------
def mapping(zeta, x1, x2):
    return ((x2 - x1)/2)*zeta + (x1 + x2)/2


# -------------------------------------------------
# 2-Point Gauss Quadrature
# -------------------------------------------------
def gauss_2_point(f, x1, x2):

    zeta = [-1/np.sqrt(3), 1/np.sqrt(3)]
    w = [1, 1]

    J = (x2 - x1)/2

    integral = 0

    for i in range(2):
        x_val = mapping(zeta[i], x1, x2)
        integral += w[i]*f(x_val)

    return J*integral


# -------------------------------------------------
# 3-Point Gauss Quadrature
# -------------------------------------------------
def gauss_3_point(f, x1, x2):

    zeta = [-np.sqrt(0.6), 0, np.sqrt(0.6)]
    w = [5/9, 8/9, 5/9]

    J = (x2 - x1)/2

    integral = 0

    for i in range(3):
        x_val = mapping(zeta[i], x1, x2)
        integral += w[i]*f(x_val)

    return J*integral


# -------------------------------------------------
# Integration limits
# -------------------------------------------------
x1 = -1
x2 = 4


# -------------------------------------------------
# Exact analytical solution
# -------------------------------------------------
exact = sp.integrate(f_sym, (x, x1, x2))
exact = float(exact)


# -------------------------------------------------
# Numerical solutions
# -------------------------------------------------
I2 = gauss_2_point(f, x1, x2)
I3 = gauss_3_point(f, x1, x2)


# -------------------------------------------------
# Errors
# -------------------------------------------------
error2 = abs(exact - I2)
error3 = abs(exact - I3)


# -------------------------------------------------
# Print results
# -------------------------------------------------
print("Exact Integral      =", exact)
print("2-Point Gauss       =", I2)
print("3-Point Gauss       =", I3)
print("Error (2-Point)     =", error2)
print("Error (3-Point)     =", error3)