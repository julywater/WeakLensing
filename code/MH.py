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
NP=1
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
	a=np.zeros(2*NP)
	for i in range(0,2*NP):
		a[i]=average[i]
	return a
def lnlikelihood(e,gamma,P):
	e0=reshear(e,gamma)
	value=lnfep(np.abs(e0),P)+np.log(jacobian(e,e0,gamma))
	return value
	#return prior(gamma)*math.exp(-p-q)
def postfunc(X,E):
	gamma = X[0:2:2] + 1J* X[1:2:2]
	if len(gamma[np.abs(gamma)>=1])>0:
		return -np.inf
	P=np.array((2.8,2.8))
	value=lnlikelihood(E,gamma[index],P)
	return np.sum(value)
phi=[]
epsilon=[]
random.seed(67)

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

#print(average,abs(complex(average[0],average[1])-gamma[0]))
X0=initial(average) 

pros=0.16
Nstep=6000
Nburnin=1000
g1_chain=np.zeros(Nstep)
g2_chain=np.zeros(Nstep)
fold=postfunc(X0,E)
j=0
for i in xrange(Nstep):
	X1=X0+(random.gauss(0,pros),random.gauss(0,pros))
	fnew =postfunc(X1,E)
        lnprob=min([0,fnew-fold]) 
	u=math.log(random.random())
	if u<lnprob:
		j+=1
		X0=X1
		fold=fnew
	g1_chain[i]=(X0[0])
	g2_chain[i]=(X0[1])
g1=g1_chain[Nburnin::]
g2=g2_chain[Nburnin::]
print('accept rate =')
print(float(j)/i)
x=np.mean(g1)
y=np.mean(g2)


import matplotlib.pyplot as pl
pl.figure()
pl.xlim((-0.4,0.4))
pl.ylim((-0.4,0.4))
pl.axvline(x=gamma[0].real)
pl.axhline(y=gamma[0].imag)
pl.scatter((average[0]),(average[1]),marker='+',s=100,edgecolor='r')	
pl.scatter((x),(y),marker='+',s=100,edgecolor='b')	
pl.scatter(g1,g2,s=5)
pl.show()



