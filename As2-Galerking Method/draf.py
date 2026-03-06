import sympy as sp


# Integration
X, X1, X2 = sp.symbols('X X1 X2')
N1 =(X2 - X)/(X2 - X1)
N2 =(X - X1)/(X2 - X1)

expr = (N1) * (N1)+(N2) * (N2)
 
result = sp.integrate(expr, (X, X1, X2))

result = sp.simplify(result)

sp.pprint(result)