import random
import cmath
import math
import numpy as np
import emcee
N=8
NP=64
NIN=1
sigma=0.000
sig=0.1
def shear(e0,g):
	return (e0+g)/(1+g.conjugate()*e0)
def reshear(e,g):
	return (e-g)/(1-g.conjugate()*e)
def jacobian(e,e0,g):
	abs(1+4*(e.real*g.real+e.imag*g.imag))
#	delt=0.000001
#	x=e0.real
#	y=e0.imag
#	J1=(reshear(e+delt,g).real-x)/delt
#	J2=(reshear(e+delt*1J,g).imag-y)/delt
#	J3=(reshear(e+delt*1J,g).real-x)/delt
#	J4=(reshear(e+delt,g).imag-y)/delt
#	return abs(J1*J2-J3*J4)
def alogam(x):
	if x<=0:
		return 0
	y=x
	if x<7:
		f=1.0
		z=y
		while z<7 :
			f=f*z
			z=z+1.0
		y=z
		f=-math.log(f)
	else:
		f=0.0
	z=1.0/y/y
	value=f+(y-0.5)*math.log(y)-y+0.918938533204673+(((-0.000595238095238*z+0.000793650793651)*z-0.002777777777778)*z+0.083333333333333)/y
	return value
def lnBeta(p,q):
	if p<=0 or q<=0:
		return 0
	value=alogam(p)+alogam(q)-alogam(p+q)
	return value
def fep(x,alpha,beta):
	if x<0 or x>1:
		return 0
	return pow(x,alpha-1)*pow(1-x,beta-1)
def initial():
	a=[0 for i in range(2*NP+2)]
	for i in range(0,2*NP):
		a[i]=random.gauss(0.0,sig)
	a[2*NP]=2.8+2.0*(1-2*random.random())/10
	a[2*NP+1]=2.8+2.0*(1-2*random.random())/10
	return a
def lnprior(gamma,p,q):
	return -(gamma.real*gamma.real+gamma.imag*gamma.imag)/2/(sig*sig)-p-q
def lnlikelihood(e,p,q,lnB,gamma):
	e0=reshear(e,gamma)
	value=fep(abs(e0),p,q)/abs(e0)*jacobian(e,e0,gamma)
	if value<=0:
		return -np.inf
	return math.log(value)-lnB
def lnprob(X,E):
	p=X[2*NP]
	q=X[2*NP+1]
	if p<2 or q <2:
		return -np.inf
	lnB=lnBeta(p,q)
	gamma=[]
	value=0.0
	for i in range(0,NP):
		g=complex(X[2*i],X[2*i+1])
		if abs(g)>=1:
			return -np.inf
		gamma.append(g)
	for i in range(0,NP):
		for j in range(0,N):
			value+=lnlikelihood(E[i][j],p,q,lnB,gamma[i])
		value=value+lnprior(gamma[i],p,q)
	return loglike
phi=[]
epsilon=[]
random.seed(98)

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
	gamma.append(complex(random.gauss(0,sig),random.gauss(0,sig)))
f=file("gamma.txt","w")
for i in range(NP):
	f.write("%f   %f\n" %(gamma[i].real,gamma[i].imag))
f.close()
E=[[0.0 for i in range(N)] for j in range(NP)]
f=file("sheared.txt","w")
for i in range(0,NP):
	for j in range(0,N):
		E[i][j]=shear(elipse[i*N+j],gamma[i])
		f.write("%f   %f\n" %(E[i][j].real,E[i][j].imag))
f.close()
ndim,nwalks=130,500
X0=[initial() for i in range(nwalks)]

sampler = emcee.EnsembleSampler(nwalks, ndim, lnprob, args=[E],threads=12)
pos, prob, state = sampler.run_mcmc(X0, 2000)
sampler.reset()
for i in range(9):
	pos, prob, state = sampler.run_mcmc(pos, 2000)
	sampler.reset()
f = file("chain.dat", "w")
f.close()
for result in sampler.sample(pos, iterations=100, storechain=False):	
    position = result[0]	
    f = open("chain.dat", "a")
    for k in range(position.shape[0]):
        f.write("{0:4d} {1:s}\n".format(k, " ".join([str(p) for p in position[k]])))	
    f.close()


