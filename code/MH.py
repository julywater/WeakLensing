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
def initial(average,P):
	a=np.zeros(2*NP+2)
	a[:2*NP]=average
	a[2*NP:]=P
	return a
def lnlikelihood(e,gamma,P):
	e0=reshear(e,gamma)
	value=lnfep(np.abs(e0),P)+np.log(jacobian(e,e0,gamma))
	return value
	#return prior(gamma)*math.exp(-p-q)
def postfunc(X,E):
	gamma = X[0:2*NP:2] + 1J* X[1:2*NP:2]
	if len(gamma[np.abs(gamma)>=1])>0:
		return -np.inf
	P=X[2*NP:]
	if P[0]<2 or P[1]<2:
		return -np.inf
#beta function model
#       pri=prior(P)
	value=lnlikelihood(E,gamma[index],P)
	return np.sum(value)
random.seed(67)
i=0
p=2.8
q=2.8
phi=np.random.rand(N*NP)
epsilon=np.random.beta(p,q,size=N*NP)
elipse=epsilon*np.exp(2J*phi*math.pi)
gamma=1-2*np.random.rand(N*NP)+1J*(1-2*np.random.rand(N*NP))
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
average=np.mean(E,1)
E.shape=(NP*N)
#print(average,abs(complex(average[0],average[1])-gamma[0]))
P0=np.array((2.8,2.8))
X0=initial(average,P0) 

prosig=np.zeros(2*NP+2)
prosig[:2*NP]=0.16
prosig[2*NP:]=0.1
Nstep=6000
Nburnin=1000
g1=np.zeros(Nstep)
g2=np.zeros(Nstep)
Pchain=[]


#only works for 1 patch
#for many patch I want make postfunc returns arrays of ln_prob for each patch while sampling g and retuns total ln_prob while doing shape parameter sampling
def mhsampler(X0,E,fold,function,prosig,index)
	X1[index]=X0[index]+np.random.normal(loc=0.0,prosig[index])
	fnew =function(X1,E)
	lnprob=(few-fold)
	u=random.random()
	if u<lnprob:
		X0=X1
		fold=fnew
	return X0
def mhsampler_g(X0,E,fold,function,prosig):
	X1[:2*NP]=X0[:2*NP]+np.random.normal(loc=0.0,prosig[2*NP])
	fnew =function(X1,E)
 #      lnprob=np.min([0,fnew-fold]) 
 #now fnew is an array (NP) size
 	lnprob=(few-fold)
	u=np.log(np.random.rand(NP)
	X0[u<lnprob]=X1[u<lnprob]:
	retrun X0

def gibbssampler(X0,E,prosig,nsample):
	chain=np.zeros((nsample,ndim))
	for i in xrange(nsample):
		ind=np.array([2*NP,2*NP+1])
		fold=postfunc(X0,E)
		#for this case postfunc need to return arrays for (nP) patches
		X0[ind],fold=mhsampler_a(X0,E,fold,postfunc,prosig,ind)
		#not finished mhsampler_s need to return this array like fold to mhsampler_g to avoid over call of postfunc
		X0[:2*NP],fold=mhsampler_g(X0,E,fold,postfunc,prosig)
		chain[i]=X0
		
#I can't get rid of the loop using this one below		
#def gibbs(X0,E,prosig,nsamples):
#	chain=np.zeros((nsamples,ndim))
#	for i in xrange(nsamples):
#		ind=np.array((2*NP,2*NP+1))
#		fold=postfunc(X0,E)
#		X0[ind]=mhsampler(X0,E,fold,prosig,ind)[ind]
#loops version		
#		for j in xrange(NP):
#			e=E[j*N:(j+1)*N]
#			ind=np.array((2*j,2*j+1))
#			ind1=np.array((2*j,2*j+1,2*NP,2*NP+1))
#			ind2=np.array([0,1])
#			X0[ind]=mhsampler(X0[ind1],e,fold,prosig,ind2)[ind2]
#np.array version....not known yet
#		e=np.zeros((N,NP))
#		ind=np.zeros((2,NP))
#		ind1=np.zeros((4,NP))
		
		chain[i]=X0	
	return chain
	
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



