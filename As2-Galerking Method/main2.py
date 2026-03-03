import numpy as np

# ===============================
#Defining Geometry
# ===============================
nozzle_length = 10 # length of the nozzle is 10m
mid = nozzle_length / 2
# ===============================


#Defining Inputs
# ===============================

ne= 5
nn =ne+1 # Number of Elements

x = np.linspace(0, nozzle_length, nn)

rho = 1.225 # Defininig density in kg/m^3
# Integration functions
def compute_I1(x1, x2):
    return (2.5*x2 - 3*x2**2/10 + x2**3/50
            - 2.5*x1 + 3*x1**2/10 - x1**3/50)


def compute_I2(x1, x2):
    return (1.5*x2 - x2**2/10 + x2**3/150
            - 1.5*x1 + x1**2/10 - x1**3/150)



# ===============================
# Mesh Functions
# ===============================

def generate_connectivity(ne):
    # This generates connectivity matrix
    conn = np.zeros((ne,2),dtype=int)
    for e in range(ne):
        conn[e,0]=e
        conn[e,1]=e+1
    return conn

def main():
    # generate connectivity 
    connect = generate_connectivity(ne)
    #-------------Assembly--------------#
    #Initialize global stiffnes matrix
    KG = np.zeros((nn,nn))
    for i in range(ne):
        node1  = connect[i,0]
        node2 = connect[i,1]
        x1 = x[node1]
        x2 = x[node2]

        Le = x2 - x1
        # Evaluating the integration
      

        if x2 <= mid:
            # Entire element in left half
            I = compute_I1(x1, x2)

        elif x1 >= mid:
            # Entire element in right half
            I = compute_I2(x1, x2)

        else:
            # Element crosses midpoint → split integral
            I_left  = compute_I1(x1, mid)
            I_right = compute_I2(mid, x2)
            I = I_left + I_right
                
        #local stiffnes matrix 
        k_local = rho*I/Le**2* np.array([[1 ,-1],[-1 ,1]])

        nodes = [node1, node2]

        for a in range(2):
            for b in range(2):
                KG[nodes[a], nodes[b]] += k_local[a,b]

    print("Global Stiffness Matrix K:")
    print(KG)

    

    #----Aplying Drichilet boundary condition-----
    RHS = np.zeros(nn)
    KG_old = KG.copy()

    BN = [0];
    BV = [0.0];
    for i in range(len(BN)):
        n = BN[i]
        bc_value = BV[i]
        k_diag = KG[n,n]
        # zero row
        KG[n,:] = 0
        # Adjusting RHS BEFORE zeroing column
        RHS = RHS-KG[:,n]*bc_value
        #zero column
        KG[:,n] = 0
        #resotring diagonal
        KG[n,n] = k_diag
        #setting RHS
        RHS[n] = bc_value * k_diag
    print("Is symmetric after BC?", np.allclose(KG, KG.T))
    print("Condition number:", np.linalg.cond(KG))
    phi = np.linalg.solve(KG, RHS)
    print("phi =")
    print(phi)
   
    
    



if __name__ == "__main__":
    main()