from ntfacfun import *

def tf(din):
	"""
	Time-frequency transform
	
	By Yangkang Chen
	Nov 6, 2022
	"""
	import numpy as np
	
	din=np.float32(din)
	ndata=din.size;
	dout=ntft(din,ndata,3.0)
	
	return dout
	
def ntfa1d(din,dt=0.004,niter=100,rect=10,ifb=0,inv=0,verb=1,alpha=0):
	"""
	Non-stationary time-frequency transform on 1D trace
	
	
	ifb: if output basis functions
	inv: flag of inverse transform (0: forward; 1: inverse; default: 0)
	
	By Yangkang Chen
	Nov 6, 2022
	"""
	import numpy as np
	
	din=np.float32(din)
	ndata=din.size;
	
# 	if din.ndim==1:	#for 2D problems
# 		n1=din.size;
# 	elif din.ndim==2:
# 		[n1,n2]=din.shape;
	n1=din.shape[0];
	print(n1)
	
	dtmp=tf1d(din.flatten(order='F'),niter,n1,verb,rect,ifb,inv,dt,alpha)
	
	#dout: real,imag,basis,w0,dw,nw
	
	if inv==0: #forward transform
		if ifb==1: #output basis functions
			nw=np.int32((dtmp.size-3)/n1/4);
			dout=dtmp[0:n1*nw*2]
			basis=dtmp[n1*nw*2:n1*nw*4]
			w0=dtmp[n1*nw*4]/np.pi*1/dt/2
			dw=dtmp[n1*nw*4+1]/np.pi*1/dt/2
			nw2=np.int32(dtmp[n1*nw*4+2])
			if nw2 != nw:
				print('nw=',nw,'nw2=',nw2,'dimension discrepancy')
			else:
				print('dimension consistent')
			return dout,basis,w0,dw,nw
		else:
			nw=np.int32((dtmp.size-3)/n1/2);
			dout=dtmp[0:n1*nw*2]
			w0=dtmp[n1*nw*2]/np.pi*1/dt/2
			dw=dtmp[n1*nw*2+1]/np.pi*1/dt/2
			nw2=np.int32(dtmp[n1*nw*2+2])
			print(dtmp.size,w0,dw,nw2)
			if nw2 != nw:
				print('nw=',nw,'nw2=',nw2,'dimension discrepancy')
			else:
				print('dimension consistent')
			return dout,w0,dw,nw
	else:
		return dtmp


	

