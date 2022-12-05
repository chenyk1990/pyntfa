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
	
def ntfa1d(din,dt=0.004,niter=100,rect=10,verb=1,alpha=0):
	"""
	Non-stationary time-frequency transform on 1D trace
	
	By Yangkang Chen
	Nov 6, 2022
	"""
	import numpy as np
	
	din=np.float32(din)
	ndata=din.size;
	
	
	if din.ndim==1:	#for 2D problems
		n1=din.size;
	elif din.ndim==2:
		[n1,n2]=din.shape;
	
	
	dout=tf1d(din,niter,n1,verb,rect,dt,alpha)
	
	#dout: real,imag,basis

	
	return dout
	

