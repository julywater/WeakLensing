import random
import cmath
import math
import numpy as np
import emcee
import scipy
N=8
NP=64
NIN=1
sigma=0.000
sig=0.1
ndim=2*NP+20
nwalks=400
Nbin=20
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
	return abs(J1*J2-J3*J4)
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
def Beta(p,q):
	if p<=0 or q<=0:
		return 0
	value=alogam(p)+alogam(q)-alogam(p+q)
	return math.exp(value)
def fep(x,alpha,beta):
	if x<0 or x>1:
		return 0
	return pow(x,alpha-1)*pow(1-x,beta-1)
def lnfepfit(x,P):
	if x>=1 or x<0:
		return -np.inf
	i=(int)(x/(1.0/Nbin))
	return(P[i])
def initial(average,hist):
	a=[0 for i in range(2*NP+Nbin)]
	for i in range(0,2*NP):
		a[i]=random.gauss(0.0,0.05)
	for i in range(Nbin):
		a[2*NP+i]=hist[i]+random.gauss(0.0,0.3)
	return a
def prior(gamma):
	return math.exp(-(pow(gamma.real,2)+pow(gamma.imag,2))/2/pow(sig,2))
def lnprior(P):
	summ=0
	for i in range(1,Nbin):
		summ+=(P[i]-P[i-1])**2
	return -summ	
def lnlikelihood(e,P,gamma):
	e0=reshear(e,gamma)
	value=lnfepfit(abs(e0),P)+math.log(jacobian(e,e0,gamma))

	return value
	#return prior(gamma)*math.exp(-p-q)
def loglikefunc(X,E):
	gamma=[]
	
	for i in range(0,NP):
		g=complex(X[2*i],X[2*i+1])
		if abs(g)>1:
			return -np.inf
		gamma.append(g)
	P=[0 for i in range(Nbin)]
	temp=0.0
	for i in range(0,Nbin):
		temp+=math.exp(X[2*NP+i])*(0.05*i+0.025)
	for i in range(0,Nbin):
		P[i]=X[2*NP+i]-math.log(temp)
	value=lnprior(P)	
	for i in range(0,NP):
		for j in range(0,N):
			value+=lnlikelihood(E[i][j],P,gamma[i])
	return value
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
E=[[0.0 for i in range(N)] for j in range(NP)]
f=file("sheared.txt","w")
for i in range(0,NP):
	for j in range(0,N):
		elipse[i*N+j]=shear(elipse[i*N+j],gamma[i])
for i in range(0,NP):
	for j in range(0,N):
		epsilon[i*N+j]=abs(elipse[i*N+j])
for i in range(0,NP):
	for j in range(0,N):
		E[i][j]=elipse[i*N+j]
		f.write("%f   %f\n" %(E[i][j].real,E[i][j].imag))
f.close()


hist,num=np.histogram(epsilon,bins=Nbin,range=(0,1),normed=True)
for i in range(Nbin):
	hist[i]=hist[i]/Nbin
average=np.array([0.0for i in range(2*NP)])
for i in range(NP):
	for j in range(N):
		average[2*i]+=E[i][j].real
		average[2*i+1]+=E[i][j].imag			
average=average/N
		

X0=[initial(average,hist) for i in range(nwalks)]


sampler = emcee.EnsembleSampler(nwalks, ndim, loglikefunc, args=[E],threads=1)
pos, prob, state = sampler.run_mcmc(X0, 2)
sampler.reset()
print('done')
for i in range(1):
	pos, prob, state = sampler.run_mcmc(pos,5)
	sampler.reset()
print(1)
f = file("1.dat", "w")
f.close()
for result in sampler.sample(pos, iterations=1, storechain=False):	
    position = result[0]	
    f = open("1.dat", "a")
    for k in range(position.shape[0]):
        f.write("{0:4d} {1:s}\n".format(k, " ".join([str(p) for p in position[k]])))	
    f.close()


