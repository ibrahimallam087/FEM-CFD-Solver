
import numpy as np

def compute_element_stiffness(x, y, C=0.0):
    """
    Compute the element stiffness matrix for a 4-node quadrilateral element
    using 2x2 Gauss quadrature.

    Parameters
    ----------
    x : array-like (4,)
        x-coordinates of the element nodes
    y : array-like (4,)
        y-coordinates of the element nodes
    C : float
        Reaction coefficient (default = 0)

    Returns
    -------
    K : ndarray (4x4)
        Element stiffness matrix
    """

    # Convert inputs to numpy arrays
    x = np.array(x)
    y = np.array(y)

    # 2x2 Gauss points and weights
    gp = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
    w = [1, 1]

    # Initialize stiffness matrix
    K = np.zeros((4, 4))

    # Loop over Gauss points
    for i in range(2):
        for j in range(2):

            xi = gp[i]
            eta = gp[j]

            # =========================
            # Shape functions (N)
            # =========================
            N = 0.25 * np.array([
                (1 - xi) * (1 - eta),
                (1 + xi) * (1 - eta),
                (1 + xi) * (1 + eta),
                (1 - xi) * (1 + eta)
            ])

            # =========================
            # Derivatives wrt xi
            # =========================
            dN_dxi = 0.25 * np.array([
                -(1 - eta),
                 (1 - eta),
                 (1 + eta),
                -(1 + eta)
            ])

            # =========================
            # Derivatives wrt eta
            # =========================
            dN_deta = 0.25 * np.array([
                -(1 - xi),
                -(1 + xi),
                 (1 + xi),
                 (1 - xi)
            ])

            # =========================
            # Jacobian matrix J
            # =========================
            # Build coordinate matrix
            coords = np.vstack((x, y)).T  # shape (4,2)

            # Derivatives matrix (2x4)
            dN_nat = np.vstack((dN_dxi, dN_deta))

            # Compute Jacobian (2x2)
            J = dN_nat @ coords

            detJ = np.linalg.det(J)

            # =========================
            # Derivatives wrt x,y
            # =========================
            # Solve J * dN = dN_nat
            dN = np.linalg.solve(J, dN_nat)

            dN_dx = dN[0, :]
            dN_dy = dN[1, :]

            # =========================
            # B matrix (gradient matrix)
            # =========================
            B = np.vstack((dN_dx, dN_dy))  # shape (2,4)

            # =========================
            # Stiffness contribution
            # =========================
            K += (B.T @ B + C * np.outer(N, N)) * detJ * w[i] * w[j]

    return K


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    x = [0.2, 0.4, 0.35, 0.1]
    y = [0.2, 0.4, 0.5, 0.6]

    K = compute_element_stiffness(x, y, C=0.0)

    print("Element stiffness matrix K =")
    print(K)