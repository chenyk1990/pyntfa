import pyntfa as ntfa
import numpy as np
import matplotlib.pyplot as plt

din=np.random.randn(1001)
dout=ntfa.tf(din)

err=np.linalg.norm(din-dout)
print(err)



fid=open("cchirps.bin","rb");
din = np.fromfile(fid, dtype = np.float32, count = 512).reshape([512,1],order='F')
plt.plot(din)
plt.show();



dout=ntfa.ntfa1d(din)

n1=512;
nw=np.int32(dout.size/n1/2);
dout=dout.reshape([n1,nw,2],order='F');

# plt.imshow(dout[:,:,0],cmap=plt.cm.jet, interpolation='none', extent=[0,5,5,0]);

plt.imshow(dout[:,:,0]*dout[:,:,0]+dout[:,:,1]*dout[:,:,1],cmap=plt.cm.jet, interpolation='none', extent=[0,5,5,0]);

plt.show();

