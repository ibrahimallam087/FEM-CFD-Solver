import numpy as np
import sympy as sp

# -------------------------------------------------
# Symbolic variable
# -------------------------------------------------
x = sp.symbols('x')


# -------------------------------------------------
# Mapping from ζ → x
# -------------------------------------------------
def mapping(zeta, x1, x2):
    return ((x2 - x1)/2)*zeta + (x1 + x2)/2


# -------------------------------------------------
# 2-Point Gauss Quadrature (1D)
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
# 3-Point Gauss Quadrature (1D)
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
# Evaluate 1D integrals
# -------------------------------------------------
def evaluate_1D(f_sym, x1_sym, x2_sym):

    x1 = float(x1_sym)
    x2 = float(x2_sym)

    f = sp.lambdify(x, f_sym, 'numpy')

    I2 = gauss_2_point(f, x1, x2)
    I3 = gauss_3_point(f, x1, x2)

    print("------------------------------------------------")
    print("1D Integral of:", f_sym)
    print("Limits:", x1_sym, "to", x2_sym)
    print("2-Point Gauss Result =", I2)
    print("3-Point Gauss Result =", I3)
    print("------------------------------------------------\n")


# -------------------------------------------------
# 2D Gauss Quadrature (2-point tensor product)
# -------------------------------------------------
def gauss_2D_2point(f):

    zeta = [-1/np.sqrt(3), 1/np.sqrt(3)]
    eta  = [-1/np.sqrt(3), 1/np.sqrt(3)]

    w = [1,1]

    integral = 0

    for i in range(2):
        for j in range(2):
            integral += w[i]*w[j]*f(zeta[i], eta[j])

    return integral


# -------------------------------------------------
# 2D Gauss Quadrature (3-point tensor product)
# -------------------------------------------------
def gauss_2D_3point(f):

    zeta = [-np.sqrt(0.6), 0, np.sqrt(0.6)]
    eta  = [-np.sqrt(0.6), 0, np.sqrt(0.6)]

    w = [5/9, 8/9, 5/9]

    integral = 0

    for i in range(3):
        for j in range(3):
            integral += w[i]*w[j]*f(zeta[i], eta[j])

    return integral


# -------------------------------------------------
# Example 1
# -------------------------------------------------
evaluate_1D(x**2, -3, 9)


# -------------------------------------------------
# Example 2
# -------------------------------------------------
evaluate_1D(sp.cos(x), -sp.pi, 3*sp.pi)


# -------------------------------------------------
# Example 3 (2D integral)
# -------------------------------------------------
def f2(zeta, eta):
    return zeta + eta


I2_2D = gauss_2D_2point(f2)
I3_2D = gauss_2D_3point(f2)

print("------------------------------------------------")
print("2D Integral of (ζ + η) over [-1,1] × [-1,1]")
print("2-Point Gauss Result =", I2_2D)
print("3-Point Gauss Result =", I3_2D)
print("------------------------------------------------")