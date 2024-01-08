import numpy as np
import matplotlib.pyplot as plt

import obspy
from obspy.imaging.cm import obspy_sequential
from obspy.signal.tf_misfit import cwt


st = obspy.read()
tr = st[0]
npts = tr.stats.npts
dt = tr.stats.delta
t = np.linspace(0, dt * npts, npts)
f_min = 1
f_max = 50

scalogram = cwt(tr.data, dt, 8, f_min, f_max)

df=(f_max-f_min)/99.0

import pyntfa as ntfa
din=tr.data;
n1=din.shape[0]
dout,w0,dw,nw = ntfa.ntfa1d(din,dt,niter=10,rect=5,ifb=0,inv=0)
dout=dout.reshape([n1,nw,2],order='F');
dout=np.transpose(dout, (1, 0, 2));

# scalogram2=complex(dout[:,:,0],dout[:,:,1])

scalogram2 = np.empty(dout.shape[0:2], dtype=np.complex64)
scalogram2.real = dout[:,:,0]
scalogram2.imag = dout[:,:,1]

scalogram2=scalogram2[1::15,:]

fig = plt.figure()
ax = fig.add_subplot(211)

x, y = np.meshgrid(
    t,
    np.logspace(np.log10(f_min), np.log10(f_max), scalogram.shape[0]))

ax.pcolormesh(x, y, np.abs(scalogram), cmap=obspy_sequential)
ax.set_xlabel("Time after %s [s]" % tr.stats.starttime)
ax.set_ylabel("Frequency [Hz]")
ax.set_yscale('log')
ax.set_ylim(f_min, f_max)
# plt.show()

ax = fig.add_subplot(212)

x, y = np.meshgrid(
    t,
    np.logspace(np.log10(f_min), np.log10(f_max), scalogram2.shape[0]))

ax.pcolormesh(x, y, np.abs(scalogram2), cmap=obspy_sequential)
ax.set_xlabel("Time after %s [s]" % tr.stats.starttime)
ax.set_ylabel("Frequency [Hz]")
ax.set_yscale('log')
ax.set_ylim(f_min, f_max)
plt.show()


fig = plt.figure()
ax = fig.add_subplot(211)
# ax.pcolormesh(np.abs(scalogram), cmap=obspy_sequential)
plt.imshow(np.abs(scalogram),aspect='auto')
ax.set_xlabel("Time after %s [s]" % tr.stats.starttime)
ax.set_ylabel("Frequency [Hz]")
# ax.set_yscale('log')
ax.set_ylim(f_min, f_max)
# plt.show()

ax = fig.add_subplot(212)
plt.imshow(np.abs(scalogram2),aspect='auto')
# ax.pcolormesh(np.abs(scalogram2), cmap=obspy_sequential)
ax.set_xlabel("Time after %s [s]" % tr.stats.starttime)
ax.set_ylabel("Frequency [Hz]")
# ax.set_yscale('log')
ax.set_ylim(f_min, f_max)
plt.show()









