import random
import cmath
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import special 
from scipy import stats
N=8
#number per patch
NP=64
#number patches
NIN=1
sigma=0.000
sig=0.05
Nbin=10
#number of bins
ndim=2*NP+Nbin
#nwalks=300
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
def prior(x):
	return -0.0*np.sum((x[1:]-x[:-1])**2)
def fep(x,alpha,beta):
	if x<0 or x>1:
		return 0
	return pow(x,alpha-1)*pow(1-x,beta-1)
def lnfep(x,P):
	x1=x[np.all([x>=0,x<1],axis=0)]
	result=np.zeros(N*NP)
#	result[np.all([x>=0,x<1],axis=0)]=(P[0]-2)*np.log(x1)+(P[1]-1)*np.log(1-x1)-math.log(special.beta(P[0],P[1]))
	result[np.all([x>=0,x<1],axis=0)]=P[np.floor(x1/(1.0/Nbin)).astype(int)]
	result[np.any([x<0,x>=1],axis=0)]=-np.inf
	return result
def initial(P,average):
	a=np.zeros(2*NP+Nbin)
	a[Nbin:]=average
	a[:Nbin]=P
	return a
def lnlikelihood(e,gamma,P):
	e0=reshear(e,gamma)
	value=lnfep(np.abs(e0),P)+np.log(jacobian(e,e0,gamma))
	return value
	#return prior(gamma)*math.exp(-p-q)
def postfunc(X,E):
	result=np.zeros(NP+1)
	gamma = X[Nbin::2] + 1J* X[Nbin+1::2]
	P=X[:Nbin]
	P-=math.log(np.sum(np.exp(P)*(1.0/Nbin*np.linspace(0,Nbin-1,Nbin)+0.5/Nbin)))
	value=lnlikelihood(E,gamma[index],P)
	value.resize((NP,N))
	result[1:]=np.sum(value,axis=1)
	if len(gamma[np.abs(gamma)>=1])>0:
		result[1:][np.abs(gamma)>=1]=-np.inf	
	result[0]=np.sum(result[1:])+prior(P)
#	result[0]=np.sum(result[1:])
	return result
random.seed(93)
np.random.seed(seed=89)
i=0
p=2.8
q=2.8
phi=np.random.rand(N*NP)
epsilon=np.random.beta(p,q,size=N*NP)
elipse=epsilon*np.exp(2J*phi*math.pi)
gamma=0.1*(1-2*np.random.rand(NP))+0.1*(1J*(1-2*np.random.rand(NP)))
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
LIN=1.0/Nbin*np.linspace(0,Nbin-1,Nbin)+0.5/Nbin
P0=np.log(np.histogram(np.abs(E),bins=Nbin, range=((0,1)), normed=True)[0]/LIN)

#print(average,abs(complex(average[0],average[1])-gamma[0]))
#P0=np.array((2.8,2.8))
X0=initial(P0,average) 
#print(X0)
prosig=np.zeros(2*NP+Nbin)
prosig[Nbin:]=0.13
prosig[:Nbin]=0.08

np.random.seed(seed=None)
def Gibbssampler(X,function,E,prosig,Nsteps):
	X0=1.0*X
	fary0=function(X0,E)
	Chain=np.zeros((Nsteps,ndim))
	accnum1=0.0
	accnum2=0.0
#	logpost=np.zeros(Nsteps)
	X1=np.zeros(len(X0))
	X1[:]=X0[:]
	lnprob=np.zeros(NP+1)
	for i in xrange(Nsteps):
		X1[Nbin:]=X0[Nbin:]
		X1[:Nbin]=X0[:Nbin]+np.random.normal(loc=0.0,scale=prosig[:Nbin])
		fary1=function(X1,E)
		lnprob[0]=fary1[0]-fary0[0]
#		print(fary1[0],fary0[0])
#		print(function(X0,E)[0])
		u=math.log(random.random())
		if u<lnprob[0]:
			accnum1+=1
			X0[:Nbin]=X1[:Nbin]
			fary0[:]=fary1[:]
		X1[:Nbin]=X0[:Nbin]
		X1[Nbin:]=X0[Nbin:]+np.random.normal(loc=0.0,scale=prosig[Nbin:])
		fary1=function(X1,E)
		lnprob[1:]=fary1[1:]-fary0[1:]
		u=np.log(np.random.rand(NP))
		X0[Nbin::2][u<lnprob[1:]]=X1[Nbin::2][u<lnprob[1:]]
		X0[Nbin+1::2][u<lnprob[1:]]=X1[Nbin+1::2][u<lnprob[1:]]
		temp=np.sum(fary0[1:])
		fary0[1:][u<lnprob[1:]]=fary1[1:][u<lnprob[1:]]
		fary0[0]+=np.sum(fary0[1:])-temp
		accnum2+=1.0*len(u[u<lnprob[1:]])/NP
#		logpost[i]=fary0[0]
		Chain[i]=X0
	print(1.0*accnum1/Nsteps)
	print(1.0*accnum2/Nsteps)
#	f=file("logpost.txt","w")
#	for i in range(len(logpost[::10])):
#		f.write("%f\n" %logpost[::10][i])
#	return Chain.transpose()
	return Chain
Chain=Gibbssampler(X0,postfunc,E,prosig,2000)
Chain1=Gibbssampler(Chain[1999],postfunc,E,prosig,22000)
result=np.array([1+1J for i in xrange(NP)])
for i in xrange(NP):
	result[i]=complex(np.mean(Chain1.transpose()[2*i+Nbin]),np.mean(Chain1.transpose()[2*i+Nbin+1]))
print(np.mean(np.abs(result-gamma)**2)**0.5)
print(np.mean(np.abs(ave-gamma)**2)**0.5)
#print(ave-gamma)
#print(np.mean(Chain1.transpose()[2*NP+1]),np.std(Chain1.transpose()[2*NP+1]),gamma[NP-1].imag)
P=np.zeros(Nbin)
for i in xrange(Nbin):
	P[i]=np.mean(Chain1.transpose()[i])
P=P-math.log(np.sum(np.exp(P)*(1.0/Nbin*np.linspace(0,Nbin-1,Nbin)+0.5/Nbin)))
P=np.exp(P)*(1.0/Nbin*np.linspace(0,Nbin-1,Nbin)+0.5/Nbin)
X=np.log(np.histogram(epsilon,bins=Nbin, range=((0,1)), normed=True)[0]/LIN)
Q=X[:Nbin]-math.log(np.sum(np.exp(X[:Nbin])*(1.0/Nbin*np.linspace(0,Nbin-1,Nbin)+0.5/Nbin)))
Q=np.exp(Q)*(1.0/Nbin*np.linspace(0,Nbin-1,Nbin)+0.5/Nbin)

a=1.0/Nbin*np.linspace(0,Nbin-1,Nbin)+0.5/Nbin
plt.scatter(a,P)
plt.scatter(a,Q,color='r')
plt.ylim((0,0.4))
plt.show()

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
