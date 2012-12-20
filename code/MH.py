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
def postfunc(X,P,E):
	gamma = X[0:2*NP:2] + 1J* X[1:2*NP:2]
	if len(gamma[np.abs(gamma)>=1])>0:
		return -np.inf
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
X0=initial(average) 
P0=(2.8,2.8)
prosig=0.16
Nstep=6000
Nburnin=1000
g1=np.zeros(Nstep)
g2=np.zeros(Nstep)
Pchain=[]
fold=postfunc(X0,P0,E)


#only works for 1 patch
#for many patch I want make postfunc returns arrays of ln_prob for each patch while sampling g and retuns total ln_prob while doing shape parameter sampling
def mhsample(X0,P0,prosig,fold,E,samp):
	if samp=='g':
		X1=X0+np.random.gauss(0,prosig,2)
		fnew =postfunc(X1,P0,E)
        	lnprob=min([0,fnew-fold]) 
		u=np.log(random.random())
		if u<lnprob:
			return(X1,fnew,1)
		else :
			return(X0,fold,0)
	if samp=='shape':
		P1=P0+np.random.gauss(0,prosig,2)
		fnew=postfunc(X0,P1,E)
		lnprob=min([0.,fnew-fold])
		u=math.log(rangom.random())
		if u<lnprob:
			return(P1,fnew,1)
		else :
			return(P0,fold,0)
	else:
		print('wrong input')
		return(0,0,-1)
j=0
for i in xrange(Nstep)
	X0,fold,j+=mhsample(X0,P0,0.16,fold,E,'g')
	g1[i]=X0[0]
	g2[i]=X0[1]
print('accept rate of g sampling =')
print(float(j)/Nstep)
fold=postfunc(X0,P0,E)
j=0
for i in xrange(Nstep)
	P0,fold,j+=mhsample(X0,P0,0.4,fold,E,'shape')
	Pchain.append(P0)


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



