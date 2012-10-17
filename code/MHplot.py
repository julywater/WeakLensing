import matplotlib
matplotlib.use('Agg')
from matplotlib.pylab import *

#title('alpha')
A=[[] for i in range(130)]
Datafile=open("chain.dat",'r')
lines = Datafile.readlines()
i=0
for line in lines:
	i=i+1
	if i%100==0:
		X = line.split()
		for i in range(130):
			A[i].append(float(X[i+1]))
Datafile=open("gamma.txt",'r')
gamma=[]
lines=Datafile.readlines()
for line in lines:
	real,imag=line.split()
	gamma.append((float(real),float(imag)))
subplot(111)
xlabel("alpha")
ylabel("beta")
scatter(A[128],A[129],s=1)
show()
fig = figure(figsize=(30,30))
for i in range(64):
	fig.add_subplot(8,8,i+1)
	
	xlim((-0.08,0.08))
	ylim((-0.08,0.08))
	axvline(x=gamma[i][0])
	axhline(y=gamma[i][1])	
	scatter(A[2*i],A[2*i+1],s=1)
savefig('test.png')	
Datafile=open("sheared.txt",'r')
Ereal=[]
Eimag=[]
lines=Datafile.readlines()
for line in lines:
	real,imag=line.split()
	Ereal.append(float(real))
	Eimag.append(float(imag))
r=[[] for i in range(64)]
k=[[] for i in range(64)]
for i in range(64):
	for j in range(8):
		r[i].append(Ereal[8*i+j])
		k[i].append(Eimag[8*i+j])
fig2=figure(figsize=(20,20))
for i in range(64):
	fig2.add_subplot(8,8,i+1)
	xlim((-1,1))
	ylim((-1,1))
	axvline(x=gamma[i][0])
	axhline(y=gamma[i][1])		
	scatter(r[i],k[i],s=8)
savefig('data.png')
	
	

