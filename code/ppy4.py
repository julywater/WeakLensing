import random
import cmath
import math
import numpy as np
import emcee
import scipy
N=8
#number per patch
NP=64
#number patches
NIN=1
sigma=0.000
sig=0.1
Nbin=20
#number of bins
ndim=2*NP+Nbin
nwalks=400
def shear(e0,g):
#the function that produce sheared ellipticity
	return (e0+g)/(1+g.conjugate()*e0)
def reshear(e,g):
#the function that recovers unlensed ellipticity
	return (e-g)/(1-g.conjugate()*e)
def jacobian(e,e0,g):
#calculate the jacobean det which is part of the likelihood function
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
def lnfepfit(x,P):
#the step function of P(|epsilon|)
	for i in range(N*NP):
		if x[i]>=1 or x[i]<0:
			return np.array([-np.inf for i in range(N*NP)])
		i=np.floor(x/(1.0/Nbin)).astype(int)
	return P[i]
def initial(average,hist):
#initialize the emcee walkers around the output of simple average and histogram of true P(e)
	a=[0 for i in range(2*NP+Nbin)]
	for i in range(0,2*NP):
		a[i]=average[i]+random.gauss(0.0,sig)
	for i in range(Nbin):
		a[2*NP+i]=hist[i]+random.gauss(0.0,0.3)
	return a
#def lnprior(gamma):
#returns the ln prior probability density of gamma
#	return -(pow(gamma.real,2)+pow(gamma.imag,2))/2/pow(sig,2)
def lnprior(P):
#prior of step function parameters,makes the step function smooth
	summ=0
	for i in range(1,Nbin):
		summ+=(P[i]-P[i-1])**2
	return -2*summ	
def lnlikelihood(E,P,gamma):
#returns the numpy array value ,which contains the log_likelihood calculated with each galaxies.
	e0=reshear(e,gamma)
	value=lnfepfit(np.abs(e0),P)+np.log(jacobian(e,e0,gamma))
	return value
	#return prior(gamma)*math.exp(-p-q)
def postfunc(X,E):
	gamma=np.array([1J+1 for i in range(NP*N)])
	for i in range(0,NP):
		g=complex(X[2*i],X[2*i+1])
		if abs(g)>1:
			return -np.inf
		gamma[i*N:(i+1)*N]=g
	P=np.zeros(Nbin)
	P=X[2*NP:2*NP+Nbin]
        t=np.array([1.0/Nbin*i+0.5/Nbin for i in range(Nbin)])
	temp=np.sum(np.exp(P)*t)
        P=P-math.log(temp)
#load gamma and stepfunction parameters P to np arrays
#	value=lnprior(P)	
	value=lnlikelihood(E,P,gamma)
	return sum(value)+lnprior(P)
#add likelihood and  prior together
phi=[]
epsilon=[]
random.seed(87)

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
E=np.array([1J+1 for i in range(NP*N)])
f=file("sheared.txt","w")
for i in range(0,NP):
	for j in range(0,N):
		E[i*N+j]=shear(elipse[i*N+j],gamma[i])
		
for i in range(0,NP):
	for j in range(0,N):
		epsilon[i*N+j]=abs(elipse[i*N+j])
f.close()


hist,num=np.histogram(epsilon,bins=Nbin,range=(0,1),normed=True)
for i in range(Nbin):
	hist[i]=hist[i]/Nbin
average=[0.0for i in range(2*NP)]
for i in range(NP):
	for j in range(N):
		average[2*i]+=E[i*N+j].real
		average[2*i+1]=E[i*N+j].imag			

X0=[initial(average,hist) for i in range(nwalks)]

sampler = emcee.EnsembleSampler(nwalks, ndim, postfunc, args=[E],threads=1)
pos, prob, state = sampler.run_mcmc(X0, 2)
sampler.reset()
print('done')
for i in range(1):
	pos, prob, state = sampler.run_mcmc(pos,500)
	sampler.reset()
print(1)
#f = file("chain20.dat", "w")
#f.close()
#for result in sampler.sample(pos, iterations=100, storechain=False):	
#    position = result[0]	
#    f = open("chain20.dat", "a")
#    for k in range(position.shape[0]):
#        f.write("{0:4d} {1:s}\n".format(k, " ".join([str(p) for p in position[k]])))	
#    f.close()


