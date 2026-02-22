import numpy as np
import pandas as pd
from scipy.sparse.linalg import gmres, lsqr, bicg, tfqmr
import matplotlib.pyplot as plt
import matplotlib as mpl
#Plot Style 
# ======================================================
# PLOT STYLE
# ======================================================
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "lines.linewidth": 2.2,
})
#------- Input data-------#
mu = 1e-5
d = 0.1
E = 16
NN = 14
e= 2

#-------------Reading Connectivity--------------#
data = pd.read_excel("Connectivity_matrix.xlsx")
connectivity = data[['Node1','Node2']].to_numpy().astype(int)

#-------------Defining Pipe lengths-------------#
L = np.ones(E)*10
L[1]=20
L[5]=20
L[3]=14.4
#-------------Defining Pipe conductance-------------#
K_elem = np.pi*d**4/(128*mu*L)

#----------Defining Global and local matrix---------#
KG = np.zeros((NN,NN))

for i in range(E):
    node1 = connectivity[i,0]-1 
    node2 = connectivity[i,1]-1

    k = K_elem[i]

    #local matrix assembly
    KG[node1,node1] +=k
    KG[node1,node2] -= k
    KG[node2,node1] -= k
    KG[node2,node2] += k

KG_old = KG.copy()

# --------- RHS VECTOR ----------
RHS = np.zeros(NN)

RHS[0] = 5
RHS[3] = -0.71428
RHS[4] = -0.71428
RHS[5] = -0.71428

#Boundary Conditions
P = np.zeros(NN)
BNode = [11,12,13,14]

for n in BNode:
    P[n-1] = 1

# --- MODIFYING RHS  ---
for bn in BNode:
    bn -= 1
    for i in range(NN):
        RHS[i] -= KG[i,bn]*P[bn]    

# --- Modify K matrix ---
KG_BNode = []

for bn in BNode:
    KG_BNode.append(KG[bn-1,bn-1])

KG[:,[bn-1 for bn in BNode]] = 0
KG[[bn-1 for bn in BNode],:] = 0

for i,bn in enumerate(BNode):
    KG[bn-1,bn-1] = KG_BNode[i]
    RHS[bn-1] = KG_BNode[i]*P[bn-1]        

# --------- DIRECT SOLVER ----------
P_exact = np.linalg.solve(KG,RHS)

q_exact = KG_old @ P_exact

print("\nExact Pressures [bar]:")
print(P_exact)

print("\nExact Node Flows [m3/s]:")
print(q_exact)

#GMRES
P_gmres,_ = gmres(KG,RHS)

#LSQR
P_lsqr = lsqr(KG,RHS)[0]

#BiCG
P_bicg,_ = bicg(KG,RHS)

#TFQMR
P_tfqmr,_ = tfqmr(KG,RHS)

#Relative Errors
Error1 = (P_gmres-P_exact)/P_exact*100
Error2 = (P_lsqr-P_exact)/P_exact*100
Error3 = (P_bicg-P_exact)/P_exact*100
Error4 = (P_tfqmr-P_exact)/P_exact*100

# ======================================================
# ERROR PLOTS
# ======================================================

plt.figure(figsize=(11,8))

titles = [
    "GMRES Method Error",
    "LSQR Method Error",
    "BiCG Method Error",
    "TFQMR Method Error"
]

errors = [Error1, Error2, Error3, Error4]

for i in range(4):
    ax = plt.subplot(2,2,i+1)
    ax.plot(range(1, NN+1), errors[i], marker='o',color ='r')
    ax.set_title(titles[i])
    ax.set_xlabel("Node Number")
    ax.set_ylabel("Percentage Error (%)")

    # symmetric y-axis around zero for clarity
    max_err = np.max(np.abs(errors[i]))
    ax.set_ylim(-1.1*max_err, 1.1*max_err)

plt.tight_layout()
plt.show()