from qutip import *
import numpy as np 
import matplotlib.pyplot as plt 
import itertools
import math

def y_rotate(psi, theta, num_qubits): 
    y_rot = np.cos(theta/2.0)*identity(2) + -1j*np.sin(theta/2.0)*sigmay()
    operator = y_rot
    for n in range(0,num_qubits-1):
        operator = tensor(operator,y_rot)
    output = operator*psi #this should rotate every qubit by theta about y
    return output   

def x_rotate(psi, theta, num_qubits): 
    x_rot = np.cos(theta/2.0)*identity(2) -1j*np.sin(theta/2.0)*sigmax()
    operator = x_rot
    for n in range(0,num_qubits-1):
        operator = tensor(operator,x_rot)
    output = operator*psi #this should rotate every qubit by theta about y
    return output

def y_rot_OP(theta, num_qubits): 
    y_rot = np.cos(theta/2.0)*identity(2) + -1j*np.sin(theta/2.0)*sigmay()
    operator = y_rot
    for n in range(0,num_qubits-1):
        operator = tensor([operator,y_rot])
    return operator  

def x_rot_OP(theta, num_qubits): 
    y_rot = np.cos(theta/2.0)*identity(2) + -1j*np.sin(theta/2.0)*sigmax()
    operator = y_rot
    for n in range(0,num_qubits-1):
        operator = tensor(operator,y_rot)
    return operator.unit()

def subspaceOp(op,totalN, index): #BEWARE --- I THINK THIS MIGHT BE BROKEN
    print("call to subspaceOp - BEWARE - might be broken")
    for i in range(0,totalN):
        if (i==0):
            if(index!=0):
                operator = identity(2)
            else:
                operator = op
        else:
            if(i==index):
                operator = tensor([operator,op]) #I think this is the wrong side
            else:
                operator = tensor([operator,identity(2)])

    return operator


def build_O(num, q1, q2,sign): #projector of the antisym subspace for q1, q2
    #assume q2 != q1+1 for now
    superpo = (tensor([basis(2,1),basis(2,0)]) + sign*tensor([basis(2,0),basis(2,1)])).unit()

    for i in range(0,num):
        if (i==0):
            if(i==q1):
                operator = superpo*superpo.dag()
            else:
                operator = identity(2)
        else: #for all i not equal to 0
            if(i==q1):
                operator = tensor([operator,superpo*superpo.dag()])
            else:
                if(i==q1+1):
                    continue #since we just tensored on two qubit gate
                else:
                    operator = tensor([operator, identity(2)])

    if(q1+1==q2):
        final_op = operator #this switches q1+1 with q2
    else:
        SW = swap(N=num,targets=[q1+1,q2])
        final_op = SW.dag()*operator*SW #this switches q1+1 with q2
    return final_op #this is the two operators tensored 


def TPI_Op(SS, forward): #takes in a pair of stabilizsers. or other operator

    exch = (1.0/2.0)*(tensor([identity(2),identity(2)])+tensor([sigmax(),sigmax()])+tensor([sigmay(),sigmay()])+tensor([sigmaz(),sigmaz()]))
    O = (1/np.sqrt(2))*(tensor([identity(2),identity(2)])+1j*exch)
    #SS = tensor([S1,S2])
    if(forward):
        flips = O.dag()*SS*O #rotate the stabilizer pair
    else:
        flips = O*SS*O.dag() #rotate the stabilizer pair

    print(str(SS) + " maps to " + str(flips))
    print("Hermiticity? " + str(SS.isherm) + " for SS and " + str(flips.isherm) + " for flips")

    return flips #this is the two operators tensored 

    #start with the single qubit. 

def singleQdiagonalizer(p, qubit, N):
    reduced_p = p.ptrace(qubit) #trace out all but qubit, can technically be an array too like [0,1]!
    #print(reduced_p.diag())
    #reduced p of dim 2 only for simplicty
    A, B = reduced_p.eigenstates()
    eigval1 = A[0]
    eigval2 = A[1]

    eigvec1 = B[0].unit()
    eigvec2 = B[1].unit()
    #print(eigvec1)
    #print(eigvec2)
    c1 = basis(2,0).dag()*eigvec1
    c2 = basis(2,1).dag()*eigvec1
    c3 = basis(2,0).dag()*eigvec2
    c4 = basis(2,1).dag()*eigvec2
    #not quite right
    rotation_mat = c1*basis(2,0)*basis(2,0).dag()+c3*basis(2,0)*basis(2,1).dag()+c2*basis(2,1)*basis(2,0).dag()+c4*basis(2,1)*basis(2,1).dag() #can I normalize the operator like this?
    
    if(not rotation_mat.check_herm()):
        print("Non hermitian matrix found?")
    diagform = rotation_mat.dag()*reduced_p*rotation_mat
    #print(diagform)
    return rotation_mat

def stabilizer(stabs): #tensors together a stabilizer operator based upon the list of inputs e.g. ['Z','X','Z','I','I','I']
    num = len(stabs) #grab the number of inputs

    for i in range(0,num):

        if (stabs[i]== 'X'):
            if (i==0):
                operator = sigmax()
            else:
                operator = tensor([operator, sigmax()])

        if (stabs[i]== 'Y'):
            if (i==0):
                operator = sigmay()
            else:
                operator = tensor([operator, sigmay()])

        if (stabs[i]== 'Z'):
            if (i==0):
                operator = sigmaz()
            else:
                operator = tensor([operator, sigmaz()])

        if (stabs[i]== 'I'):
            if (i==0):
                operator = identity(2)
            else:
                operator = tensor([operator, identity(2)])

    return operator


def megaStab(set_in, input_state): 
    #takes in a set and produces all permutations of it, e.g. Z X Z I I I
    newlist = list(itertools.permutations(set_in))
    newSet = set(newlist) #this way we get rid of duplicates :)
    #Probably dont want ot print these..
    #print("Trying all permutations for the set " + str(set_in) + " which is of size " + str(len(newSet)))
    for item in newSet:
        temp_op = stabilizer(item) #takes the jth element of the list of all lists possible 
        #eigenvalue = input_state.dag()*temp_op*input_state #this is probably the wrong type for the following equality.
        eigenvalue = temp_op.matrix_element(input_state.dag(), input_state)
        realpart = eigenvalue.real
        impart = eigenvalue.imag
        #print(type(impart))
        #print(impart)
        #print("Eigenvalue is "+str(realpart)+" for stabilizer " + str(newlist[j]))
        if (not (np.isclose(realpart, (0.0), rtol=1e-05, atol=1e-08, equal_nan=False)) or not (np.isclose(realpart, (0.0), rtol=1e-05, atol=1e-08, equal_nan=False))): #
            
            if (np.isclose(realpart, 1.0, rtol=1e-05, atol=1e-08, equal_nan=False) or np.isclose(realpart, -1.0, rtol=1e-05, atol=1e-08, equal_nan=False)): #
                print("Succesful stabilization with " + str(item) + " and eigenvalue " + str(eigenvalue))

            else:
                print("Notable stabilization with " + str(item) + " and eigenvalue " + str(eigenvalue))


def generateAllStabilizers(indicators, N, state):
    #indicators = set(['X','Y','Z','I']) #set containng all four character indicators
    allNlengthsets = itertools.combinations_with_replacement(indicators, N) #does this kill dups?
    for item in allNlengthsets:
        megaStab(item, state) #takes the 6 item combination of the fundamental indicators and then generates all permutations.=


def luEquivalence(A,B): #A and B are the input pure states of equal dims
    #first grab the dimensionality 

    #IN PROGRESS
    N = A.shape[0]
    N2 = B.shape[0] 

    if (not (N == N2)):
        print("Dimensional mismatch")
        return False

    num = int(math.log(N,2)) #this is the number of particles rather than matrix dimension

    #now, check if the states are generic
    Agen = (isGeneric(A,num,False))
    Bgen = (isGeneric(B,num,False))

    if((Agen and not Bgen) or (not Agen and Bgen)):
        print("Mismatch: One generic state, one non genreic.")
        return False

    if(Agen and Bgen): #if both states are generic, we can do the single qubit calculation from diagonalizer for all of them. 
        for z in range(0,num):
            Asub = isGenSubset(A,z)
            Bsub = isGenSubset(B,z)

    return True


def isGenSubset(state, sel): #checks if the reduced state density matrix is prop to I
    state = state.unit() #renormalize just in case
    p = state*state.dag() #the density matrix
    result = True
    length = len(sel)
    #print(length)
    reduced_p = p.ptrace(sel) #trace out all other qubits but i xlength 
    print(reduced_p)
    iden = identity(2)
    for m in range(1,length):
        iden = tensor([iden, identity(2)])
    op = (reduced_p) #I Think this works and I do not need reduced_p - I!
    #print(op)
    nrm = op.norm()
    print(nrm)
    if(np.isclose(nrm, (0), rtol=1e-05, atol=1e-08, equal_nan=False)):
        result = False
    return result

def grabPureEigenstate(eigenstates):
    num=len(eigenstates) #should always be 2
    numst = len(eigenstates[0])
    eigvals = eigenstates[0]
    eigsts = eigenstates[1]
    #print(eigvals)
    for i in range(0,numst):
        if(np.isclose(eigvals[i], (1.0), rtol=1e-05, atol=1e-08, equal_nan=False)):
            return eigsts[i] #found the unit eigenvalue, return the eigenvector

    print("No Unit Eigenstate found - not pure state?")


def isGeneric(state, N, printout): #determines if a state is a generic state, of N qubits
    state = state.unit() #renormalize just in case
    p = state*state.dag() #the density matrix
    result = True

    for i in range(0,N): 
        reduced_p = p.ptrace(i) #trace out all other qubits but i 
        op = (reduced_p - identity(2))
        nrm = op.norm()
        if(np.isclose(nrm, (1.0), rtol=1e-05, atol=1e-08, equal_nan=False)):
            if(printout):
                print("Failed for qubit " +str(i) + " with reduced p = " + str(reduced_p))#for now...
            result = False
    
    return result

def tangle(state, N):
    if (N>4): 
        print("Warning. N>4 may not return tangle properly... not verified.")
    for j in range(0,N):
        if (j==0):
            operator = -1*sigmay() #note, this basis is FLIPPED, but shouldnt matter...
        else: 
            operator=tensor(operator,-1*sigmay())
    tangle = state.dag()*operator*state.conj()
    return tangle


def linearclusterBuilder(N):
    #here we want to build up a cluster state of size N... the way outlined in the 2001 paper... .

    for i in range(0,N): #this could also be done recursively I bet, but this is fine for now.
        if (i==0):
            state = tensor([basis(2,0)+basis(2,1)]).unit()
            temp_op = sigmaz()
        else:
            state = (tensor([basis(2,0),temp_op*state])+tensor([basis(2,1),state])).unit()
            temp_op = tensor([temp_op, identity(2)]) #every iteration tensor on an identity at the end

    return state.unit()

def twoDclusterBuilder(N,adj_mat):
    #here we want to build up a 2D cluster state of size N... 
    #adj mat should be NxN
    for i in range(N-1,-1,-1): #step through each qubit starting at the top
        operator = identity(2)
        #first construct the operator from the adjacency matrix for this qubit
        temp_vec = adj_mat[i] #or is this column idk lol    
        for m in range(i,N): #step through the vector starting at the diagonal, so we only look at things with a higher 
            if(not (m==i)): #if we are not on the diagonal
                val = temp_vec[m]
                if (val == 1.0):
                    if(m == i+1): #if we are one off from the diagonal
                        operator = sigmaz()
                    else:
                        operator = tensor([operator,sigmaz()]) 
                else:
                    if(m == i+1): #if we are one off from the diagonal
                        operator = identity(2)
                    else:
                        operator = tensor([operator,identity(2)]) 
        #now we should have an operator that corresponds to the values after the diagonal 0 = I 1 = sigmaz
        if(i == N-1): #if this is the first run, operator does not have a value
            state = tensor([basis(2,0)+basis(2,1)]).unit() #so we prepare a superposition state
        else:
            state = (tensor([basis(2,0),operator*state])+tensor([basis(2,1),state])).unit()

    return state

def GHZBuilder(N):
    for i in range(0,N):
        if(i==0):
            v0 = basis(2,0)
            v1 = basis(2,1)
        else:
            v0 = tensor([v0,basis(2,0)])
            v1 = tensor([v1,basis(2,1)])

    v0=v0.unit()
    v1=v1.unit()
    GHZN = (v0+v1).unit()
    return GHZN

def generateEntanglementLandscape(rho, N):
    #state_p = state*state.dag()
    indicators = np.zeros(N)
    for j in range(0,N):
        indicators[j] = j
    #print(indicators)
    for i in range(1,(N+2)/2): #note, we do NOT want replacement here...
        all_i_length_sets = itertools.combinations(indicators, i) #does this kill dups?

        for item in all_i_length_sets:
            entropy = entropy_vn(rho.ptrace(item),2)
            print("Entropy for set " +str(item) +" is " +str(entropy))

 #FROM MICHAEL GOERZ BLOG - MODIFIED FOR QOBJ CLASS
def HS(M1, M2):
    """Hilbert-Schmidt-Product of two matrices M1, M2"""
    out = (M1.dag()*M2).tr() #is this the same as "dotting" them? Think its right based on what HS is 
    return out

def c2s(c):
    """Return a string representation of a complex number c"""
    if c == 0.0:
        return "0"
    if c.imag == 0:
        return "%g" % c.real
    elif c.real == 0:
        return "%gj" % c.imag
    else:
        return "%g+%gj" % (c.real, c.imag)

def decompose(H):
    """Decompose Hermitian 4x4 matrix H into Pauli matrices"""
    S = [identity(2), sigmax(), sigmay(), sigmaz()]
    labels = ['I', 'sigma_x', 'sigma_y', 'sigma_z']
    for i in xrange(4):
        for j in xrange(4):
            label = labels[i] + ' tensor ' + labels[j]
            a_ij = 0.25 * HS(tensor([S[i],S[j]]), H)
            if a_ij != 0.0:
                print "%s\t*\t( %s )" % (c2s(a_ij), label)  



######################### COMPUTATIONAL KERNEL #########################

input_set = set(['X','Z','I'])
######################################### test the cluster builder method #########################################
test_state = linearclusterBuilder(6)
#generateAllStabilizers(input_set, 6,test_state)

######################################### test the aj mat builder method #########################################
#write down an NxN matrix defining the adjacency of your graph for the twwDclusterBuilder method
adj_test = [[0,1,0,1],[1,0,1,0],[0,1,0,1],[1,0,1,0]] #should be the adj matrix for a square 2d cluster
box_state = twoDclusterBuilder(4,adj_test)
#print("Box State is")
#print(box_state)
#generateAllStabilizers(input_set, 4,box_state) #checks out, awesome!

######################################### Test some methods tending towards detremining LU equivalence #########################################
#try reduced ps of three qubit cluster
#print("checking if gensubset works 4 box is generic: "+str(isGenSubset(box_state,[0]))) # i see, this gives components

#print("checking if gensubset works 4 box is generic: "+str(box_state.extract_states(1)))

#singleQdiagonalizer(three_qb_cluster_p,0,3) #seems to be working here.


#luEquivalence(box_state,box_state)

CNOT = (1.0/2.0)*(tensor([identity(2),identity(2)])+tensor([sigmaz(),identity(2)])+tensor([identity(2),sigmax()])-tensor([sigmaz(),sigmax()]))
ZX = tensor([sigmax(),sigmaz()])
CX = controlled_gate(sigmax(),N=2)


#test = TPI_Op(ZX,False) #true -> forward operation
#print("eigenstates of ZX squig are ")
#print(test.eigenstates())
#print(hadamard_transform(N=2)*tensor([basis(2,0),basis(2,0)]))
#XZ00 = test*hadamard_transform(N=2)*tensor([basis(2,0),basis(2,0)])
#print(XZ00)

#decompose(test)
check = -0.5*tensor([identity(2),sigmay()])-0.5*tensor([sigmay(),identity(2)])+0.5*tensor([sigmax(),sigmaz()])+0.5*tensor([sigmaz(),sigmax()])
#test/check result

eigenstate1 = (1.0/np.sqrt(2))*tensor([basis(2,0),basis(2,0)])+(0.5*(1+1j)/np.sqrt(2))*tensor([basis(2,1),basis(2,0)])+(0.5*(1-1j)/np.sqrt(2))*tensor([basis(2,0),basis(2,1)])
eig_p = eigenstate1*eigenstate1.dag()
eigentropy =entropy_vn(eig_p.ptrace([0]),2)
#print(eigentropy)
 
########################### TEST THE SUBSPACE SUPERPOSITION PROJECTOR METHOD #########################
input_set2 = ['X','Z','I','Y']

subspace_pro6 = build_O(6,0,1,1) #inputs are num qubits, q1, q2, sign of superpositon A + sign*B
bell_antisym = (tensor([basis(2,0),basis(2,1)])+tensor([basis(2,1),basis(2,0)])).unit()
bell_sym = (tensor([basis(2,0),basis(2,0)])+tensor([basis(2,1),basis(2,1)])).unit()
twoBell = tensor([bell_sym,bell_sym]).unit()
bell_box = tensor([bell_antisym,box_state])
intermediate6q = subspace_pro6*bell_box
basisvec = tensor([basis(2,1),basis(2,1),basis(2,1),basis(2,0),basis(2,0),basis(2,1)])

roto_intermediate6q = y_rotate(intermediate6q,np.pi/2.0,6)
#print(roto_intermediate6q)
######################################################################################################

#print(box_state)
roto_box = y_rotate(box_state,0,4).unit()
boxTANGLE = tangle(roto_box,4)
#generateAllStabilizers(input_set, 4,roto_box)

#print("tangle of box is "+str(boxTANGLE))


CZ_box12 = controlled_gate(sigmaz(),N=4,control=0,target=1,control_value=1)
CZ_box23 = controlled_gate(sigmaz(),N=4,control=1,target=2,control_value=1)
CZ_box34 = controlled_gate(sigmaz(),N=4,control=2,target=3,control_value=1)
modified_box = CZ_box12*CZ_box23*CZ_box34*roto_box

#print(modified_box)

box_p = roto_box*roto_box.dag()
mod_box_p = modified_box*modified_box.dag()

#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*##**#*#*#*#*##*
SET4 = [0,1] #0,1,2,3
SET=[2]
print("#################################################")

#print("Set4 is " + str(SET4))
#compare the entropies of partitions of the state
box_entro = entropy_vn(box_p.ptrace(SET4),2)
mod_box_entro = entropy_vn(mod_box_p.ptrace(SET4),2)


######################### Manipulations on CL4 #############################

CL4 = linearclusterBuilder(4)
CL4_p = CL4*CL4.dag()
#print("LC4 entropy is "+str(entropy_vn(CL4_p.ptrace(SET4),2)))
#print(isGenSubset(CL4,SET4))
#print("Entanglement Landscape for CL4")
#generateEntanglementLandscape(CL4_p,4)
LC4TANGLE = tangle(CL4,4)
#print("tangle of LC4 is "+str(LC4TANGLE))

######################### Manipulations on CL3Cl3 #############################
subspace_pro6_03 = build_O(6,0,3,1)
subspace_pro6_14 = build_O(6,1,4,1)

CL3CL3 = tensor([linearclusterBuilder(3).unit(),linearclusterBuilder(3).unit()]).unit()

projected_CL3CL3 = (subspace_pro6_03*CL3CL3).unit() #RENORMALIZING HERE!!!
projected_CL3CL3_14 = (subspace_pro6_14*CL3CL3).unit() #RENORMALIZING HERE!!!

projected_CL3CL3_p = projected_CL3CL3*projected_CL3CL3.dag()
projected_CL3CL3_14_p = projected_CL3CL3_14*projected_CL3CL3_14.dag()

red_projected_CL3CL3_p=projected_CL3CL3_p.ptrace([1,2,4,5])

red_projected_CL3CL3_ent1 = entropy_vn(red_projected_CL3CL3_p.ptrace(SET4),2)
#print("mystery box (ghz?) CL3 03 entropy is "+str(red_projected_CL3CL3_ent1))

red_projected_CL3CL3_14_p=projected_CL3CL3_14_p.ptrace([0,2,3,5])
red_projected_CL3CL3_14_ent1 = entropy_vn(red_projected_CL3CL3_14_p.ptrace(SET4),2)
#print("mystery box (ghz?) CL3 14 entropy is "+str(red_projected_CL3CL3_14_ent1))


#grabbing the state post trace, manually
#print("Eigenstates of reduced CL3Cl3 rho is"+str(red_projected_CL3CL3_p.eigenstates()))
CL3CL3_03_eigenstate = grabPureEigenstate(red_projected_CL3CL3_p.eigenstates())

CL3CL3_14_eigenstate = grabPureEigenstate(red_projected_CL3CL3_14_p.eigenstates())

xro4= x_rot_OP(np.pi/2.0,4)
yro4= y_rot_OP(np.pi/2.0,4)
phaser1 = phasegate(np.pi/4.0,4,0) #-pi/2 phase shift on the second qubit
phaser2 = phasegate(-np.pi/4.0,4,1) #-pi/2 phase shift on the second qubit

X3 = subspaceOp(sigmax(),4,2)
X4 = subspaceOp(sigmax(),4,3)
#generateAllStabilizers(input_set, 4,yro4*CL3CL3_03_eigenstate) #I cannot quite seem to prove 
#that this is indeed a GHZ! Entanglement landscape checks out... 

modded_CL3CL3_14 = (xro4*X4*X3*phaser2*phaser1*yro4*CL3CL3_14_eigenstate).unit()

#print(modded_CL3CL3_14)
#generateAllStabilizers(input_set,4,modded_CL3CL3_14)

CL3CL3TANGLE = tangle(CL3CL3_03_eigenstate,4)
#print("tangle of mystery CL3CL3 state is "+str(CL3CL3TANGLE))
######################### Manipulations on CL4Cl4 #############################
subspace_pro8_04 = build_O(8,1,5,1)
yro4 = y_rot_OP(np.pi/2.0,4)
#Wow!
CL4CL4 = tensor([linearclusterBuilder(4).unit(),(yro4*linearclusterBuilder(4)).unit()]).unit()
input_set3=['X','Z','I']
yro8 = y_rot_OP(np.pi/2.0,8)

yro6 = y_rot_OP(np.pi/2.0,6)

SET2 = [0,3] #0,1,2,3
projected_CL4CL4 = (CL4CL4).unit()
projected_CL4CL4 = (subspace_pro8_04*projected_CL4CL4).unit() 
projected_CL4CL4_p = projected_CL4CL4*projected_CL4CL4.dag()

red_projected_CL4CL4_p=projected_CL4CL4_p.ptrace([0,2,3,4,6,7])

red_projected_CL4CL4_ent1 = entropy_vn(red_projected_CL4CL4_p.ptrace(SET2),2)

measured6Q = grabPureEigenstate(red_projected_CL4CL4_p.eigenstates())
print("Six Q state from a single rotated CL4!")
print(yro6*measured6Q)
#generateEntanglementLandscape(red_projected_CL4CL4_p,6)
#generateAllStabilizers(input_set,6,measured6Q)



######################### bell pair pair #############################

twoBell_p = twoBell*twoBell.dag()
twoBell_ent = entropy_vn(twoBell_p.ptrace(SET4),2)
#print("TwoBell entropy is "+str(twoBell_ent))
twobell = tangle(twoBell,4)
#print("tangle of two bell pairs is "+str(twobell))

######################### Manipulations on CL5Cl5 #############################
subspace_pro10_05 = build_O(10,0,5,1)
subspace_pro10_16 = build_O(10,1,6,1)
subspace_pro10_38 = build_O(10,3,8,1)
subspace_pro10_49 = build_O(10,4,9,1)

yro5= y_rot_OP(np.pi/2.0,5)

yro1= y_rot_OP(np.pi/2.0,1)

RY_5 = subspaceOp(yro1,10,5) #
RY_9 = subspaceOp(yro1,10,9) #



CL5CL5 = tensor([linearclusterBuilder(5).unit(),(yro5*linearclusterBuilder(5)).unit()]).unit()

yro10= y_rot_OP(np.pi/2.0,10)
H_5 = subspaceOp(hadamard_transform(N=1),10,5) #
Z_5 = subspaceOp(sigmaz(),10,5) #Z on qubit 7

H_6 = subspaceOp(hadamard_transform(N=1),10,6) #
Z_6 = subspaceOp(sigmaz(),10,6) #Z on qubit 7

H_8 = subspaceOp(hadamard_transform(N=1),10,8) #
Z_8 = subspaceOp(sigmaz(),10,8) #Z on qubit 7

H_9 = subspaceOp(hadamard_transform(N=1),10,9) #


projected_CL5CL5 = (subspace_pro10_38*subspace_pro10_16*CL5CL5).unit() #RENORMALIZING HERE!!!
#projected_CL5CL5 = (subspace_pro10_49*projected_CL5CL5).unit()

projected_CL5CL5_p = projected_CL5CL5*(projected_CL5CL5).dag()

red_projected_CL5CL5_p=projected_CL5CL5_p.ptrace([0,2,4,5,7,9]) #SOMETHING IS WRONG

yro1_m= y_rot_OP(-np.pi/2.0, 1)
half_ro = tensor([identity(2),identity(2),identity(2),yro1_m,yro1_m,yro1_m]) #should be a six qubit

#red_projected_CL5CL5_p_ent1 = entropy_vn(red_projected_CL5CL5_p.ptrace(SET3),2)
CL5CL5_6q_measured = half_ro*grabPureEigenstate(red_projected_CL5CL5_p.eigenstates())
print("Entanglement Landscape on twice measured CL5CL5")
generateEntanglementLandscape(CL5CL5_6q_measured,6)


print(CL5CL5_6q_measured)

generateAllStabilizers(input_set,6,CL5CL5_6q_measured)

######################### Manipulations on CL6Cl6 #############################
subspace_pro12_06 = build_O(12,0,6,1)
subspace_pro12_17 = build_O(12,1,7,1)

subspace_pro12_28 = build_O(12,2,8,1)
subspace_pro12_410 = build_O(12,4,10,1)

yro6= y_rot_OP(np.pi/2.0,6)

H_7 = subspaceOp(hadamard_transform(N=1),12,7) #hadamrads qubit 7 of 12
Z_7 = subspaceOp(sigmaz(),12,7) #Z on qubit 7

CL6CL6 = tensor([linearclusterBuilder(6).unit(),(linearclusterBuilder(6)).unit()]).unit()

projected_CL6CL6 = (Z_7*H_7*subspace_pro12_17*Z_7*H_7*CL6CL6).unit() #RENORMALIZING HERE!!!

projected_CL6CL6_p = projected_CL6CL6*projected_CL6CL6.dag()

red_projected_CL6CL6_p=projected_CL6CL6_p.ptrace([0,2,3,4,5,6,8,9,10,11])

CL6CL6_6q_measured = grabPureEigenstate(red_projected_CL6CL6_p.eigenstates())
print("Entanglement Landscape on thrice measured CL6CL6")
#generateEntanglementLandscape(CL6CL6_6q_measured,6)

yro1 = y_rot_OP(-np.pi/2.0, 1)
half_ro = tensor([identity(2),identity(2),identity(2),yro1,yro1,yro1]) #should be a six qubit

#lets try rotating it back.
#ro_CL6CL6_6q_measured = half_ro*CL6CL6_6q_measured

#print(CL6CL6_6q_measured)
#generateAllStabilizers(input_set,10,CL6CL6_6q_measured)

######################### Manipulations on CL7CL7 #############################
subspace_pro14_07 = build_O(14,0,7,1)
subspace_pro14_29 = build_O(14,2,9,1)

subspace_pro14_310 = build_O(14,3,10,1)

subspace_pro14_411 = build_O(14,4,11,1)
subspace_pro14_613 = build_O(14,6,13,1)

projection1 = (basis(2,0)).unit()
fail_op = projection1*projection1.dag()
fail_07 = subspaceOp(fail_op,14,0)*subspaceOp(fail_op,14,7)
fail_310 = subspaceOp(fail_op,14,3)*subspaceOp(fail_op,14,10)
fail_613 = subspaceOp(fail_op,14,6)*subspaceOp(fail_op,14,13)

yro7= y_rot_OP(np.pi/2.0,7)

CL7CL7 = tensor([linearclusterBuilder(7).unit(),(yro7*linearclusterBuilder(7)).unit()]).unit()

#fail case - first succeeds, second fails.
projected_CL7CL7 = (fail_310*subspace_pro14_07*CL7CL7).unit() #RENORMALIZING HERE!!!

#Success case
projected_CL7CL7 = (subspace_pro14_310*
subspace_pro14_07*CL7CL7).unit() #RENORMALIZING HERE!!! subspace_pro14_613*

projected_CL7CL7_p = projected_CL7CL7*projected_CL7CL7.dag()

red_projected_CL7CL7_p=projected_CL7CL7_p.ptrace([1,2,4,5,6,8,9,11,12,13])

CL7CL7_6q_measured = grabPureEigenstate(red_projected_CL7CL7_p.eigenstates())
print("Entanglement Landscape on thrice measured CL7CL7")

yro1 = y_rot_OP(-np.pi/2.0, 1)
half_ro = tensor([identity(2),identity(2),identity(2),identity(2),identity(2),yro1,yro1,yro1,yro1,yro1]) #should be a six qubit

#lets try rotating it back.
ro_CL7CL7_6q_measured = half_ro*CL7CL7_6q_measured
#print(ro_CL7CL7_6q_measured)
#generateEntanglementLandscape(CL7CL7_6q_measured,10)

#generateAllStabilizers(input_set,10,ro_CL7CL7_6q_measured)



################# Compare various 4 qubit targets ###################################

################# 4 qubits, five links --- pretty sure equivalent to linear or box###########################
adj_test2 = [[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]] #should be the adj matrix for a square 2d cluster
twoD_mid_4Q = twoDclusterBuilder(4,adj_test2).unit()

#print("2D 4 qubit State is")
#print(twoD_4Q)
#generateAllStabilizers(input_set, 4,twoD_mid_4Q) #checks out, awesome!

yro4= x_rot_OP(np.pi/2.0,4)
twoD_mid_4Q_p = twoD_mid_4Q*twoD_mid_4Q.dag()
twoD_mid_4Q_ent = entropy_vn(twoD_mid_4Q_p.ptrace(SET4),2)
#print("4 qubit mid con 2d cluster entropy is " + str(twoD_mid_4Q_ent))
twoDtangle = tangle(twoD_mid_4Q,4)
#print("4 qubit 2d mid con tangle is " + str(twoDtangle))

################# 4 qubits, six links - equivalent to GHZ4 and CNOT ###################################

adj_test2 = [[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]] #should be the adj matrix for a square 2d cluster
twoD_4Q = twoDclusterBuilder(4,adj_test2).unit()
#print("2D 4 qubit State is")
#print(twoD_4Q)
#generateAllStabilizers(input_set, 4,twoD_4Q) #checks out, awesome!
yro4= x_rot_OP(np.pi/2.0,4)
twoD_4Q_p = twoD_4Q*twoD_4Q.dag()
twoD_4Q_ent = entropy_vn(twoD_4Q_p.ptrace(SET4),2)
#print("4 qubit max con 2d cluster entropy is " + str(twoD_4Q_ent))
twoDtangle = tangle(twoD_4Q,4)

#print(yro4*twoD_4Q)
#print("4 qubit 2d maxcon tangle is " + str(twoDtangle))

################# 4 qubits, GHZ ###################################

GHZ4 = GHZBuilder(4)
GHZ4_p = GHZ4*GHZ4.dag()
GHZ4_ent = entropy_vn(GHZ4_p.ptrace(SET4),2)
#print("4 qubit GHZ entropy is " + str(GHZ4_ent))
#generateAllStabilizers(input_set,4,GHZ4)
#print(isGenSubset(GHZ4,SET4))
GHZ4tangle = tangle(GHZ4,4)
#print("4 qubit GHZ tangle is " + str(GHZ4tangle))

################################## 4 qubits, 4 links - CNOT = GHZ4 and tetra? ###################################
yro4= y_rot_OP(-np.pi/2.0,4)
xro4= x_rot_OP(-np.pi/2.0,4)

adj_test2 = [[0,1,0,0],[1,0,1,1],[0,1,0,0],[0,1,0,0]] #should be the adj matrix for a square 2d cluster
CNOTclust_4Q = twoDclusterBuilder(4,adj_test2).unit()
#print("2D 4 qubit CNOT Cluster State is")
#print(yro4*xro4*CNOTclust_4Q)

#generateAllStabilizers(input_set, 4,CNOTclust_4Q) #checks out, awesome!
yro4= x_rot_OP(np.pi/2.0,4)
CNOTclust_4Q_p = CNOTclust_4Q*CNOTclust_4Q.dag()
CNOTclust_4Q_ent = entropy_vn(CNOTclust_4Q_p.ptrace(SET4),2)


#generateEntanglementLandscape(CNOTclust_4Q_p,4)
#print("4 qubit CNOT cluster entropy is " + str(CNOTclust_4Q_ent))
CNOTclust_4Qtangle = tangle(CNOTclust_4Q,4)
#print(yro4*twoD_4Q)
#print("4 qubit CNOT cluster tangle is " + str(CNOTclust_4Qtangle))

################# Manipulations on Tetra Tetra ###################################
tetratetra = tensor([twoD_4Q,twoD_4Q]).unit()
subspace_pro8_04 = build_O(8,0,5,1)
subspace_pro8_37 = build_O(8,3,7,1)

projected_tet2 = (subspace_pro8_04*tetratetra).unit()
projected_tet2_p = projected_tet2*projected_tet2.dag()

yro6= y_rot_OP(-np.pi/2.0,6)
xro6= x_rot_OP(-np.pi/2.0,6)

SET6=[4,5] #note to self, need to write method to generate all of these at once... should be trivial...
red_projected_tet2_p = projected_tet2_p.ptrace([1,2,3,4,6,7]) #trace out the measured qubits


red_projected_tet2_p_ent1 = entropy_vn(red_projected_tet2_p.ptrace(SET6),2)
#print("Entropy of tettet 6 qubit state is "+str(red_projected_tet2_p_ent1))

tet_tet_measured = grabPureEigenstate(red_projected_tet2_p.eigenstates())
#print(xro6*tet_tet_measured) #confirmed, this is GHZ6

#generateEntanglementLandscape(tet_tet_measured,6)


############ Compare to GHZ6 ##########

GHZ6 = GHZBuilder(6)
GHZ6_p = GHZ6*GHZ6.dag()
GHZ6_ent = entropy_vn(GHZ6_p.ptrace(SET6),2)
print("6 qubit GHZ entropy is " + str(GHZ6_ent))
#generateAllStabilizers(input_set,4,GHZ4)
#print(isGenSubset(GHZ4,SET4))
#GHZ4tangle = tangle(GHZ4,4)
#print("4 qubit GHZ tangle is " + str(GHZ4tangle))


################# Square Lattice 6 ##################
adj_test6 = [[0,1,0,1,0,0],[1,0,1,0,1,0],[0,1,0,0,0,1],[1,0,0,0,1,0],[0,1,0,1,0,1],[0,0,1,0,1,0]] #should be the adj matrix for a square 2d cluster
lattice_6 = twoDclusterBuilder(6,adj_test6).unit()
print("Six Q Lattice!")

lattice_6_p = lattice_6*lattice_6.dag()
#appears to check out.
#generateAllStabilizers(input_set, 6,lattice_6) #checks out, awesome!
#generateEntanglementLandscape(lattice_6_p,6)
yro4= x_rot_OP(np.pi/2.0,4)

twoDtangle = tangle(twoD_4Q,4)

################# Manipulations on CNOT CNOT#################################
CNOTCNOT = tensor([CNOTclust_4Q,CNOTclust_4Q]).unit()
subspace_pro8_14 = build_O(8,1,5,1)
#print("CNOT CNOT projections")
projected_CNOTCNOT = (subspace_pro8_14*CNOTCNOT).unit()

projected_CNOTCNOT_p = projected_CNOTCNOT*projected_CNOTCNOT.dag()

yro6= y_rot_OP(-np.pi/2.0,6)
xro6= x_rot_OP(-np.pi/2.0,6)

red_projected_CNOTCNOT_p = projected_CNOTCNOT_p.ptrace([0,2,3,4,6,7]) #trace out the measured qubits

CNOTCNOT_measured = grabPureEigenstate(red_projected_CNOTCNOT_p.eigenstates())
#print(yro6*CNOTCNOT_measured)
#generateAllStabilizers(input_set,6,CNOTCNOT_measured)
#generateEntanglementLandscape(red_projected_CNOTCNOT_p,6)

################# Twist Lattice 1 #################################
adj_testtwist = [[0,1,0,1,0,0],[1,0,1,0,0,0],[0,1,0,0,0,0],[1,0,0,0,1,0],[0,0,0,1,0,1],[0,0,0,0,1,0]] #should be the adj matrix for a square 2d cluster
lattice_twist = twoDclusterBuilder(6,adj_testtwist).unit()
print("Six Q table Lattice! EUREKA!")

lattice_twist_p = lattice_twist*lattice_twist.dag()

#generateAllStabilizers(input_set, 6,lattice_6) #checks out, awesome!
#generateEntanglementLandscape(lattice_twist_p,6)

################# BabYag State  ##################
adj_test8 = [[0,1,0,0,1,0,0,0],[1,0,1,0,0,1,0,0],[0,1,0,1,0,0,0,0],
[0,0,1,0,0,0,0,0],[1,0,0,0,0,1,0,0],[0,1,0,0,1,0,1,0],[0,0,0,0,0,1,0,1],[0,0,0,0,0,0,1,0]] #should be the adj matrix for a square 2d cluster
babyag8 = twoDclusterBuilder(8,adj_test8).unit()
print("BabYag!")

#generateAllStabilizers(input_set, 8,babyag8) #checks out, awesome!

yro1_forward = y_rot_OP(np.pi/2.0,1)

subspace_pro8_37 = build_O(8,3,7,1)
yro_single7 = subspaceOp(yro1_forward,8,7) #should just rotate the final qubit

yro_single7_rev = subspaceOp(yro1.dag(),6,7) #should just rotate the final qubit


babyag8 = (yro_single7*babyag8).unit() #make the single qubit rotation
projected_babyag8 = (subspace_pro8_37*babyag8).unit()

projected_babyag8_p = projected_babyag8*projected_babyag8.dag()

red_projected_babyag8_p = projected_babyag8_p.ptrace([0,1,2,4,5,6])

measured_babyag = grabPureEigenstate(red_projected_babyag8_p.eigenstates())

#dont reverse it! that one is measured out!
#measured_babyag = yro_single7_rev*measured_babyag
#appears to check out.
#generateAllStabilizers(input_set, 6,measured_babyag) #checks out, awesome!
#generateEntanglementLandscape(measured_babyag,6)


####################### Tests on the linear cluster 7 for stability #################
projection_single = (basis(2,0)).unit()
fail_op = projection_single*projection_single.dag()
fail_0 = subspaceOp(fail_op,7,0)

fail_0_new = tensor([identity(2),identity(2),identity(2),fail_op,identity(2),identity(2),identity(2)])

L7_OG = linearclusterBuilder(7).unit()
L7_RO = (yro7*linearclusterBuilder(7)).unit()

m_L7_OG = (fail_0_new*L7_OG).unit()
m_L7_RO = (fail_0_new*L7_RO).unit()

m_L7_OG_p = m_L7_OG*m_L7_OG.dag()
m_L7_RO_p = m_L7_RO*m_L7_RO.dag()

red_m_L7_OG_p = m_L7_OG_p.ptrace([0,1,2,4,5,6])
red_m_L7_RO_p = m_L7_RO_p.ptrace([0,1,2,4,5,6])

eig_m_L7_OG = grabPureEigenstate(red_m_L7_OG_p.eigenstates())
eig_m_L7_RO = grabPureEigenstate(red_m_L7_RO_p.eigenstates())


print("OG L7 measured")
#generateEntanglementLandscape(eig_m_L7_OG,6)
print("RO L7 measured")
#generateEntanglementLandscape(eig_m_L7_RO,6)



######################### Manipulations on CL10CLs10 jeez#############################
subspace_pro20_016 = build_O(20,0,16,1)
subspace_pro20_313= build_O(20,3,13,1)
subspace_pro20_616 = build_O(20,6,16,1)


projection1 = (basis(2,0)).unit()

fail_op = projection1*projection1.dag()
fail_07 = subspaceOp(fail_op,14,0)*subspaceOp(fail_op,14,7)
fail_310 = subspaceOp(fail_op,14,3)*subspaceOp(fail_op,14,10)
fail_613 = subspaceOp(fail_op,14,6)*subspaceOp(fail_op,14,13)

yro10= y_rot_OP(np.pi/2.0,10)

CL10CL10 = tensor([linearclusterBuilder(10).unit(),(yro10*linearclusterBuilder(10)).unit()]).unit()


#Success case
projected_CL10CL10 = (subspace_pro20_616*subspace_pro20_313*
subspace_pro20_016*CL10CL10).unit() #RENORMALIZING HERE!!! subspace_pro14_613*

projected_CL10CL10_p = projected_CL10CL10*projected_CL10CL10.dag()

red_projected_CL10CL10_p=projected_CL10CL10_p.ptrace([1,2,4,5,7,8,9,11,12,14,15,17,18,19])

CL10CL10_14q_measured = grabPureEigenstate(red_projected_CL10CL10_p.eigenstates())
print("Entanglement Landscape on thrice measured CL10CL10")

yro1 = y_rot_OP(-np.pi/2.0, 1)
half_ro = tensor([identity(2),identity(2),identity(2),identity(2),identity(2),identity(2),identity(2),
yro1,yro1,yro1,yro1,yro1,yro1,yro1]) #should be a six qubit

#lets try rotating it back.
ro_CL10CL10_14q_measured = half_ro*CL10CL10_14q_measured
#print(ro_CL7CL7_6q_measured)
#generateEntanglementLandscape(CL7CL7_6q_measured,10)

generateAllStabilizers(input_set,14,ro_CL10CL10_14q_measured)
