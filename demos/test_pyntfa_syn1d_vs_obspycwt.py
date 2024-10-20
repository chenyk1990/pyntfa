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
t=np.linspace(0,dt,n1*dt)
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

## comparison with CWT

import obspy
from obspy.imaging.cm import obspy_sequential
from obspy.signal.tf_misfit import cwt

f_min = 0.001
f_max = 0.5
w0=8 #parameter for the wavelet, tradeoff between time and frequency resolution
dt=1
# scalogram = cwt(din, dt, w0, f_min, f_max, nf=257)

# st = obspy.read()
# tr = st[0]
# npts = tr.stats.npts
# dt = tr.stats.delta
# t = np.linspace(0, dt * npts, npts)
# f_min = 1
# f_max = 50
din2=np.zeros(3000);din2[:512]=din[:,0];
# din2=tr.data
scalogram = cwt(din2, dt, 8, f_min, f_max, nf=257) #size 257x3000
scalogram=scalogram[:,0:512];


## from NTFA
dout=np.transpose(dout, (1, 0, 2)); #frequency|time|real&imag
scalogram2 = np.empty(dout.shape[0:2], dtype=np.complex64)
scalogram2.real = dout[:,:,0]
scalogram2.imag = dout[:,:,1]
fig = plt.figure(figsize=(16, 8))
plt.subplot(1,2,1)
plt.imshow(np.abs(scalogram),aspect='auto')
# plt.show()

plt.subplot(1,2,2)
plt.imshow(np.abs(scalogram2),aspect='auto')
plt.show()

t=np.arange(512)
grid_x, grid_y = np.meshgrid(
    t,
    np.logspace(np.log10(f_min), np.log10(f_max), scalogram.shape[0]))

points1,points2 = np.meshgrid(
    t,
    np.linspace(0, 0.5, scalogram.shape[0]))



# Below figure will help you understand the difference between OBSPY CWT and other real-world time-frequency transforms

from scipy.interpolate import griddata
# grid_z2 = griddata(np.transpose(np.array([points1.flatten(),points2.flatten()])),scalogram2.flatten(), (grid_x, grid_y),method='linear')

grid_z2 = griddata(np.transpose(np.array([points1.flatten(),points2.flatten()])),scalogram2.flatten(), (grid_x, grid_y),method='cubic')

fig = plt.figure(figsize=(16, 8))
plt.subplot(1,3,1)
plt.imshow(np.abs(scalogram),aspect='auto')
# plt.show()

plt.subplot(1,3,2)
plt.imshow(np.abs(scalogram2),aspect='auto')

plt.subplot(1,3,3)
plt.imshow(np.abs(grid_z2),aspect='auto')
# plt.savefig('comp_linear.png')
plt.savefig('comp_cubic.png')
plt.show()




# grid_z2 = griddata(scalogram2.real.flatten(), np.transpose(np.array([points1.flatten(),points2.flatten()])), np.transpose(np.array([points1.flatten(),points2.flatten()])))


# import scipy
# from scipy import io
# datas = {"ntfa_r":dout[:,:,1],"ntfa_i":dout[:,:,0],'tracere':trace}
# scipy.io.savemat("ntfa.mat", datas)






