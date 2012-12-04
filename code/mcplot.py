import matplotlib
matplotlib.use('Agg')
from matplotlib.pylab import *
from scipy import stats
NP=64
N=8
dim=2*NP+2
#title('alpha')
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
	f.write("%f   %f\n" %(g1[i],dg1[i])
f.close()
f=open("deltag2.txt",'a')
for i in range(len(g2n)):
	f.write("%f   %f\n" %(g2[i],dg2[i])






