#here I will attempt to construct the TPI state from two cluster states 
from qutip import *
import numpy as np 
import matplotlib.pyplot as plt 

def x_rotate(psi, theta, num_qubits): 
	x_rot = np.cos(theta/2.0)*identity(2) -1j*np.sin(theta/2.0)*sigmax()
	operator = x_rot
	for n in range(0,num_qubits-1):
		operator = tensor(operator,x_rot)
	output = operator*psi #this should rotate every qubit by theta about y
	return output

def y_rotate(psi, theta, num_qubits): 
	y_rot = np.cos(theta/2.0)*identity(2) -1j*np.sin(theta/2.0)*sigmay()
	operator = y_rot
	for n in range(0,num_qubits-1):
		operator = tensor(operator,y_rot)
	output = operator*psi #this should rotate every qubit by theta about y
	return output


#constuct two cluster states in the textbook basis
three_qb_clusterA = (tensor([basis(2,0), basis(2,0), basis(2,0)])+tensor([basis(2,0), basis(2,0), basis(2,1)])+tensor([basis(2,0), basis(2,1), basis(2,0)])+
	tensor([basis(2,1), basis(2,0), basis(2,0)])-tensor([basis(2,0), basis(2,1), basis(2,1)])+tensor([basis(2,1), basis(2,0), basis(2,1)])+
	-tensor([basis(2,1), basis(2,1), basis(2,0)])+tensor([basis(2,1), basis(2,1), basis(2,1)])).unit()


roto_cluster_x= x_rotate(three_qb_clusterA,-np.pi/2.0,3)



RCX_p = roto_cluster_x*roto_cluster_x.dag()
#RCY_p = roto_cluster_y*roto_cluster_y.dag()

#entrop_X = entropy_vn(RCX_p.ptrace([2]),2)

#entrop_Y = entropy_vn(RCY_p.ptrace([2]),2)

#print(entrop_Y)
#print(entrop_X) #checks out! but this isnt a good metric... 

#Let's try measuring on the rotated cluster state density matrix

#THREE QUBIT MEASUREMENT MATRICES in elementary basis
th_SQM_qubit1_0 = tensor((basis(2,0)*basis(2,0).dag()),identity(2),identity(2))
th_SQM_qubit1_1 = tensor((basis(2,1)*basis(2,1).dag()),identity(2),identity(2))
th_SQM_qubit2_0 = tensor(identity(2),(basis(2,0)*basis(2,0).dag()),identity(2))
th_SQM_qubit2_1 = tensor(identity(2),(basis(2,1)*basis(2,1).dag()),identity(2))
th_SQM_qubit3_0 = tensor(identity(2),identity(2),(basis(2,0)*basis(2,0).dag()))
th_SQM_qubit3_1 = tensor(identity(2),identity(2),(basis(2,1)*basis(2,1).dag()))

#this should be the PURE sub ensemble 
temp = th_SQM_qubit2_1*th_SQM_qubit2_1*RCX_p
cl_mq2_is1_p = (th_SQM_qubit2_1*RCX_p*th_SQM_qubit2_1)/(temp.tr())
cl_mq2_is1_p = cl_mq2_is1_p.ptrace([0,2]) #trace out qubit 2
#print("check rho 1")
#print(cl_mq2_is1_p)

#this should now describe the pure 2 qubit state
rho_entrop = entropy_vn(cl_mq2_is1_p.ptrace(1),2)
#print("rho entropy is")
#print(rho_entrop)




######################## PRETTY SURE THIS IS WRONG, DENSITY MATRIX FORM IS RIGHT?
#now to check out the post measurement states - this is what I could get measuring Q2 in the X basis
apl = (1.0/4)*(1+1j)
amin = (1.0/4)*(1-1j)


#I think I am making a mistake here... when I do the projective measurement on the rotates staet it works again ...

X_basis_q2_0 = (amin*tensor([basis(2,0),basis(2,0)])-apl*tensor([basis(2,0),basis(2,1)])+
	-apl*tensor([basis(2,1),basis(2,0)])+amin*tensor([basis(2,1),basis(2,1)])).unit()

X_basis_q2_1 = (-amin*tensor([basis(2,0),basis(2,0)])-apl*tensor([basis(2,0),basis(2,1)])+
	-apl*tensor([basis(2,1),basis(2,0)])-amin*tensor([basis(2,1),basis(2,1)])).unit()


#print(X_basis_q2_0)
#print(X_basis_q2_1)

X_basis_q2_0_p = X_basis_q2_0*X_basis_q2_0.dag()
en_X_basis_q2_0 = entropy_vn(X_basis_q2_0_p.ptrace([1]),2)
print(en_X_basis_q2_0)



X_basis_q2_1_p = X_basis_q2_1*X_basis_q2_1.dag()
en_X_basis_q2_1 = entropy_vn(X_basis_q2_1_p.ptrace([1]),2)
print(en_X_basis_q2_1)


#print("check rho 2")
#print(X_basis_q2_1_p) #okay theyre super similar but a few signs off...


#check if the density matrix of the measured state is an eigen operator of the one I wrote?

##################Construct POST TPI state#################
four_qubit_TPI = (tensor(X_basis_q2_0,X_basis_q2_1) + tensor(X_basis_q2_1,X_basis_q2_0)).unit()
print(four_qubit_TPI)

