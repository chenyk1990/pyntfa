import pyntfa as ntfa
import numpy as np
import matplotlib.pyplot as plt

din=np.random.randn(1001)
dout=ntfa.tf(din)

err=np.linalg.norm(din-dout)
print(err)

# Download the data from https://github.com/aaspip/data/blob/main/cchirps.bin
fid=open("cchirps.bin","rb");
din = np.fromfile(fid, dtype = np.float32, count = 512).reshape([512,1],order='F')
plt.plot(din)
plt.show();

n1=din.shape[0]
dout,w0,dw,nw = ntfa.ntfa1d(din,dt=1,niter=10,rect=7,ifb=0,inv=0)
dout1,basis,w02,dw2,nw2 = ntfa.ntfa1d(din,dt=1,niter=10,rect=7,ifb=1,inv=0)

dout=dout.reshape([n1,nw,2],order='F');
basis=basis.reshape([n1,nw,2],order='F')

print(dout[:,:,0].max(),dout[:,:,0].min()) ##should be [0.0864,-0.0876]
## NOTE: dout[:,:,0] is imaginary and dout[:,:,1] is real

## plot TF spectrum
plt.figure();
plt.imshow(dout[:,:,0]*dout[:,:,0]+dout[:,:,1]*dout[:,:,1],cmap=plt.cm.jet, interpolation='none', extent=[0,nw*dw-dw,n1*1-1,0],aspect='auto');plt.xlabel('Frequency (Hz)');plt.ylabel('Time (s)');plt.title('Time-frequency Spectrum')
plt.show();

## plot basis functions
plt.figure();
plt.subplot(1,2,1);
plt.imshow(basis[:,:,1],cmap=plt.cm.jet, interpolation='none', extent=[0,nw*dw-dw,n1*1-1,0],aspect='auto');plt.xlabel('Frequency (Hz)');plt.ylabel('Time (s)');plt.title('Basis function (Real)')
plt.subplot(1,2,2);
plt.imshow(basis[:,:,0],cmap=plt.cm.jet, interpolation='none', extent=[0,nw*dw-dw,n1*1-1,0],aspect='auto');plt.xlabel('Frequency (Hz)');plt.title('Basis function (Imaginary)')
plt.show();

## Inverse transform
trace=ntfa.ntfa1d(dout,dt=1,niter=10,rect=7,ifb=0,inv=1);


## Better visualization
dt=1;
t=np.linspace(0,(n1-1)*dt,n1)
fig = plt.figure(figsize=(16, 8))
plt.subplot(3,3,1)
plt.plot(t,din,'k',linewidth=1);plt.ylim(-3,3);plt.gca().set_xticks([]);plt.ylabel('Amplitude');plt.title('Input');
plt.subplot(3,3,4)
plt.plot(t,trace,'k',linewidth=1);plt.ylim(-3,3);plt.gca().set_xticks([]);plt.ylabel('Amplitude');plt.title('Reconstruction');
plt.subplot(3,3,7)
plt.plot(t,din.flatten()-trace,'k',linewidth=1);plt.ylim(-3,3);plt.ylabel('Amplitude');plt.xlabel('Time (s)');plt.title('Reconstruction Error');

plt.subplot(1,3,2)
plt.imshow(dout[:,:,0]*dout[:,:,0]+dout[:,:,1]*dout[:,:,1],cmap=plt.cm.jet, interpolation='none', extent=[0,nw*dw-dw,n1*1-1,0],aspect='auto');plt.xlabel('Frequency (Hz)');plt.ylabel('Time (s)');plt.title('Time-frequency Spectrum')

plt.subplot(1,3,3)
plt.imshow(basis[:,:,1],cmap=plt.cm.jet, interpolation='none', extent=[0,nw*dw-dw,n1*1-1,0],aspect='auto');plt.xlabel('Frequency (Hz)');plt.ylabel('Time (s)');plt.title('Basis function (Real)')

plt.savefig('test_pyntfa_syn1d.png',format='png',dpi=300)
plt.show();

