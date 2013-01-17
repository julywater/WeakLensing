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
NP=5
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
inf1=np.array([-np.inf for i in range(NP+1)])
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
	x1=x[np.all([x>=0,x<1],axis=0)]
	result=np.zeros(N*NP)
	result[np.all([x>=0,x<1],axis=0)]=(P[0]-2)*np.log(x1)+(P[1]-1)*np.log(1-x1)-math.log(special.beta(P[0],P[1]))
	result[np.any([x<0,x>=1],axis=0)]=-np.inf
	return result
def initial(P,average):
	a=np.zeros(2*NP+2)
	a[2:]=average
	a[:2]=P
	return a
def lnlikelihood(e,gamma,P):
	e0=reshear(e,gamma)
	value=lnfep(np.abs(e0),P)+np.log(jacobian(e,e0,gamma))
	return value
	#return prior(gamma)*math.exp(-p-q)
def postfunc(X,E):
	result=np.zeros(NP+1)
	gamma = X[2::2] + 1J* X[3::2]
#	print(gamma[index])
	P=X[:2]
#	print(P)
	if P[0]<2 or P[1]<2:
		return inf1
	value=lnlikelihood(E,gamma[index],P)
#	print(value)
	value.resize((NP,N))
	result[1:]=np.sum(value,axis=1)
	if len(gamma[np.abs(gamma)>=1])>0:
		result[1:][np.abs(gamma)>=1]=-np.inf	
	result[0]=np.sum(result[1:])
	
	return result
random.seed(67)
i=0
p=2.8
q=2.8
phi=np.random.rand(N*NP)
epsilon=np.random.beta(p,q,size=N*NP)
elipse=epsilon*np.exp(2J*phi*math.pi)
gamma=0.1*(1-2*np.random.rand(N*NP))+0.1*(1J*(1-2*np.random.rand(N*NP)))
E=shear(elipse,gamma[index])
f=file("gamma.txt","w")
for i in range(NP):
	f.write("%f   %f\n" %(gamma[i].real,gamma[i].imag))
f.close()
f=file("sheared.txt","w")
for i in range(0,N*NP):
	f.write("%f   %f\n" %(E[i].real,E[i].imag))
f.close()
E.shape=(NP,N)
ave=np.mean(E,1)
average=np.zeros(2*NP)
average[0::2]=np.real(ave)
average[1::2]=np.imag(ave)
E.shape=(NP*N)
#print(average,abs(complex(average[0],average[1])-gamma[0]))
P0=np.array((2.8,2.8))
X0=initial(P0,average) 
#print(X0)
prosig=np.zeros(2*NP+2)
prosig[2:]=0.16
prosig[:2]=0.01


def Gibbssampler(X0,function,E,prosig,Nsteps):
	fary0=function(X0,E)
	Chain=np.zeros((Nsteps,ndim))
	accnum1=0.0
	X1=np.zeros(2*NP+2)
	X1[:]=X0[:]
	lnprob=np.zeros(NP+1)
	for i in xrange(Nsteps):
		X1[:2]=X0[:2]+np.random.normal(loc=0.0,scale=prosig[:2])
		fary1=function(X1,E)
		lnprob[0]=(fary1[0]-fary0[0])
		u=math.log(random.random())
	
		if u<lnprob[0]:
			accnum1+=1
			X0[:2]=X1[:2]
			fary0[:]=fary1[:]
		X1[2:]=X0[2:]+np.random.normal(loc=0.0,scale=prosig[2:])
		fary1=function(X1,E)
#		print(0,fary0[1:])
#		print(1,fary1[1:])
		lnprob[1:]=fary1[1:]-fary0[1:]
		u=np.log(np.random.rand(NP))
		print(0,X0[2:])
		X0[2::2][u<lnprob[1:]]=X1[2::2][u<lnprob[1:]]
		X0[3::2][u<lnprob[1:]]=X1[3::2][u<lnprob[1:]]
		fary0[1:][u<lnprob[1:]]=fary1[1:][u<lnprob[1:]]
		fary0[0]=np.sum(fary0[1:])
#		print(lnprob)
#		print(1,X0[2:])
		Chain[i]=X0
	print(accnum1/Nsteps)
	return Chain.transpose()
Chain=Gibbssampler(X0,postfunc,E,prosig,100)
for i in xrange(2*NP+2):
	print(np.mean(Chain[i]))	
#import matplotlib.pyplot as pl
#pl.figure()
#pl.xlim((-0.4,0.4))
#pl.ylim((-0.4,0.4))
#pl.axvline(x=gamma[0].real)
#pl.axhline(y=gamma[0].imag)
#pl.scatter((average[0]),(average[1]),marker='+',s=100,edgecolor='r')	
#pl.scatter((x),(y),marker='+',s=100,edgecolor='b')	
#pl.scatter(g1,g2,s=5)
#pl.show()



