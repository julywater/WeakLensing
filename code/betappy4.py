import random
import cmath
import math
import numpy as np
import emcee
import scipy
from scipy import special 
from scipy import stats
N=8
#number per patch
NP=64
#number patches
NIN=1
sigma=0.000
sig=0.05
#Nbin=20
#number of bins
ndim=2*NP+2
nwalks=300
index=np.array([i for i in range(NP) for j in range(N)])
#t = (np.arange(Nbin)+0.5) / float(Nbin)
inf=np.array([-np.inf for i in range(N*NP)])
def shear(e0,g):
	return (e0+g)/(1+g.conjugate()*e0)
def reshear(e,g):
	return (e-g)/(1-g.conjugate()*e)
def jacobian(e,e0,g):
	delt=0.000001
	x=e0.real
	y=e0.imag
	J1=(reshear(e+delt,g).real-x)/delt
	J2=(reshear(e+delt*1J,g).imag-y)/delt
	J3=(reshear(e+delt*1J,g).real-x)/delt
	J4=(reshear(e+delt,g).imag-y)/delt
	return np.abs(J1*J2-J3*J4)

def fep(x,alpha,beta):
	if x<0 or x>1:
		return 0
	return pow(x,alpha-1)*pow(1-x,beta-1)
def lnfep(x,P):
	if len(x[x<0])>=1 or len(x[x>=1])>=1:
		return inf
	return (P[0]-2)*np.log(x)+(P[1]-1)*np.log(1-x)-math.log(special.beta(P[0],P[1]))
def initial(average):
	a=[0 for i in range(2*NP+2)]
	for i in range(0,2*NP):
		a[i]=average[i]+random.gauss(0.0,0.1)
	for i in range(2):
		a[2*NP+i]=2.8+random.gauss(0.0,0.2)
	return a
def lnlikelihood(e,gamma,P):
	e0=reshear(e,gamma)
	value=lnfep(np.abs(e0),P)+np.log(jacobian(e,e0,gamma))
	return value
	#return prior(gamma)*math.exp(-p-q)
def postfunc(X,E):
	gamma = X[0:2*NP:2] + 1j * X[1:2*NP:2]
	if len(gamma[np.abs(gamma)>=1])>0:
		return -np.inf
	P=X[2*NP:2*NP+2]
	if P[0]<=2 or P[1]<=2:
		return -np.inf
	value=lnlikelihood(E,gamma[index],P)
	return np.sum(value)
phi=[]
epsilon=[]
#random.seed(57)

for i in range(0,N*NP):
	phi.append(random.random())
random.shuffle(phi)
i=0
p=2.8
q=2.8
while i<N*NP :
	temp=random.random()
	if 4*fep(temp,p,q)>random.random():
		epsilon.append(temp)
		i=i+1
random.shuffle(epsilon)
elipse=[]
for i in range(0,N*NP):
	elipse.append(epsilon[i]*cmath.exp(2J*phi[i]*math.pi))
gamma=[]
for i in range(0,NP):
	gamma.append(complex(0.1*(1-2*random.random()),0.1*(1-2*random.gauss(0,sig))))
f=file("gamma.txt","w")
for i in range(NP):
	f.write("%f   %f\n" %(gamma[i].real,gamma[i].imag))
f.close()
E=np.array([1J+1 for i in range(NP*N)])
f=file("sheared.txt","w")
for i in range(0,NP):
	for j in range(0,N):
		E[i*N+j]=shear(elipse[i*N+j],gamma[i])
		f.write("%f   %f\n" %(E[i*N+j].real,E[i*N+j].imag))
f.close()
for i in range(0,NP):
	for j in range(0,N):
		epsilon[i*N+j]=np.abs(E[i*N+j])
average=np.array([0.0 for i in range(2*NP)])
for i in range(NP):
	for j in range(N):
		average[2*i]+=E[i*N+j].real
		average[2*i+1]+=E[i*N+j].imag			
average=average/N
		
X0=[initial(average) for i in range(nwalks)]

sampler = emcee.EnsembleSampler(nwalks, ndim, postfunc, args=[E],threads=10)
pos, prob, state = sampler.run_mcmc(X0, 100)
sampler.reset()

for i in range(9):
	pos, prob, state = sampler.run_mcmc(pos,2000)
	sampler.reset()

f = file("chain.dat", "w")
f.close()
i=0
for result in sampler.sample(pos, iterations=100,storechain=False):	
	i+=1
        position = result[0]
	if 1:	
        	f = open("chain.dat", "a")
	        for k in range(position.shape[0]):
        	        f.write("{0:4d} {1:s}\n".format(k, " ".join([str(p) for p in position[k]])))	
		f.close()




dim=2*NP
A=[[] for i in range(dim)]
Datafile=open("chain.dat",'r')
lines = Datafile.readlines()
for line in lines:
	X = line.split()
	for w in range(dim):
		A[w].append(float(X[w+1]))
Datafile=open("gamma.txt",'r')
gamma=[]
lines=Datafile.readlines()
for line in lines:
	real,imag=line.split()
	gamma.append((float(real),float(imag)))
t1=np.zeros(NP)
t2=np.zeros(NP)
v1=np.zeros(NP)
v2=np.zeros(NP)
for i in range(NP):
	t1[i]=np.mean(A[2*i])
	t2[i]=np.mean(A[2*i+1])
	v1[i]=np.std(A[2*i])
	v2[i]=np.std(A[2*i+1])

g1=np.array([gamma[i][0] for i in range(NP)])
g2=np.array([gamma[i][1] for i in range(NP)])
dg1=t1-g1
dg2=t2-g2

f=open("deltag1.txt",'a')
for i in range(len(g1)):
	f.write("%f   %f   %f\n" %(g1[i],dg1[i],v1[i]))
f.close()
f=open("deltag2.txt",'a')
for i in range(len(g2n)):
	f.write("%f   %f   %f\n" %(g2[i],dg2[i],v2[i]))
f.close()
