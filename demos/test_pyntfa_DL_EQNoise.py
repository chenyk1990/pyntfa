#!/usr/bin/env python
# coding: utf-8

# In[1]:


# First download data (Github allows file size < 25Mb)
# !git clone https://github.com/chenyk1990/mldata 
# !cp -r mldata/eq/signalnoise ./
# 
# Following is based on the EQCCT environment
# conda create -n eqcct python=3.7.16
# conda activate eqcct
# conda install ipython notebook
# pip install obspy tqdm matplotlib-scalebar tensorflow==2.8.0 protobuf==3.20.1 pandas==1.3.5 scikit-learn==1.0.2
# conda activate eqcct
#
# Install PyNTFA package
# pip install git+https://github.com/chenyk1990/pyntfa


# In[2]:

# get_ipython().system('ls signalnoise/')

# In[3]:


import glob
import numpy as np


# In[4]:

## Here we load the signal and noise samples
# Noise is manually picked by YKC, this is only a small portion of the whole dataset of TXED 
# Signal is segmented according to analysts' picks in the TexNet catalog

noise=glob.glob("signalnoise/noise*.npy") #this is a file list
noise=[np.load(ii) for ii in noise]       #this is list of numpy array
noise=np.concatenate(noise,axis=2)        #this is a numpy array

signal=glob.glob("signalnoise/signal*.npy") #this is a file list
signal=[np.load(ii) for ii in signal]       #this is list of numpy array
signal=np.concatenate(signal,axis=2)        #this is a numpy array

print(noise.shape)
print(signal.shape)


# In[5]:


#Let's plot some sample waveforms
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (6,6)
def plot(datan,no,mode,figdir='./'):
	'''
	datan: numpy array of signal/noise
	no: number of figures
	mode: in order or random
	'''
	import os
	if os.path.isdir(figdir) == False:  
		os.makedirs(figdir,exist_ok=True)
	data=[]
	if mode==1:
		n_total=datan.shape[2]
		np.random.seed(20212223);
		order=np.arange(n_total);
		np.random.shuffle(order)
		inds=order[0:no]
	
	for jj in range(no):
		if mode==0:
			ii=jj;
		elif mode==1:
			ii=inds[jj];
		else:
			ii=jj;
		data=datan[:,:,ii]
		print("noise %d/%d"%(jj+1,no))
		
		fig, ax = plt.subplots()
		ax.set_xticklabels([]);
		ax.set_yticklabels([]);
		ax1 = fig.add_subplot(311)		
		plt.plot(data[:,0], 'k',label='Z')
		legend_properties = {'weight':'bold'}		
		ymin, ymax = ax1.get_ylim()
		plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)				
		plt.ylabel('Amplitude', fontsize=12) 
		ax1.set_xticklabels([])
		ax = fig.add_subplot(312)				 
		plt.plot(data[:,1], 'k',label='N')
		legend_properties = {'weight':'bold'}		
		ymin, ymax = ax.get_ylim()
		plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)				
		plt.ylabel('Amplitude', fontsize=12) 
		ax.set_xticklabels([])

		ax = fig.add_subplot(313)				 
		plt.plot(data[:,2], 'k',label='E')
		legend_properties = {'weight':'bold'}		
		ymin, ymax = ax.get_ylim()
		plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)				
		plt.ylabel('Amplitude', fontsize=12) 

		ax1.set_title('Waveform #%d'%ii, fontsize=14)

# 		plt.savefig(fname='/noise-'+str(jj)+'.png', format="png")
# 		plt.close() 


# In[6]:
# plot(signal,10,1,'./signal/')
# 
# 
# # In[7]:
# plot(noise,10,1,'./noise/')


# In[8]:
from keras.layers import Input
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers import Reshape
from keras.layers import GlobalAveragePooling1D


from keras.layers.convolutional import MaxPooling1D
# from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import concatenate

signal1=np.swapaxes(np.swapaxes(signal,0,2),1,2)
noise1=np.swapaxes(np.swapaxes(noise,0,2),1,2)
labels=np.ones([signal1.shape[0],1])
labeln=np.zeros([noise1.shape[0],1])

data=np.concatenate([signal1,noise1],axis=0)
label=np.concatenate([labels,labeln],axis=0)
label = to_categorical(label)


nsample=200;
data=np.concatenate([signal1[0:nsample,:,:],noise1[0:nsample,:,:]],axis=0)
label=np.concatenate([labels[0:nsample],labeln[0:nsample]],axis=0)
label = to_categorical(label)


# In[9]:


import pyntfa as ntfa
datas=data[0:nsample*2]
# dds=[]
n1=6000
nf=10

## uncomment the following to re-generate the spectra (and comment "datanew=np.load('./datanew%d.npy'%(nsample*2))")
# for ii in range(np.shape(datas)[0]):
#     print(ii,np.shape(datas)[0])
#     dtmp,basis,w0,dw,nw = ntfa.ntfa1d(datas[ii,:,0],dt=0.01,niter=10,rect=20,ifb=1,inv=0)
#     dtmp=dtmp.reshape([n1,nw,2],order='F');dtf0=dtmp[:,:,0]*dtmp[:,:,0]+dtmp[:,:,1]*dtmp[:,:,1];
    
#     dtmp,basis,w0,dw,nw = ntfa.ntfa1d(datas[ii,:,1],dt=0.01,niter=10,rect=20,ifb=1,inv=0)
#     dtmp=dtmp.reshape([n1,nw,2],order='F');dtf1=dtmp[:,:,0]*dtmp[:,:,0]+dtmp[:,:,1]*dtmp[:,:,1];
    
#     dtmp,basis,w0,dw,nw = ntfa.ntfa1d(datas[ii,:,2],dt=0.01,niter=10,rect=20,ifb=1,inv=0)
#     dtmp=dtmp.reshape([n1,nw,2],order='F');dtf2=dtmp[:,:,0]*dtmp[:,:,0]+dtmp[:,:,1]*dtmp[:,:,1];
    
#     nf=10
#     dtf0=dtf0[:,np.linspace(100,300,nf,dtype='int')].reshape([1,n1,nf,1])
#     dtf1=dtf1[:,np.linspace(100,300,nf,dtype='int')].reshape([1,n1,nf,1])
#     dtf2=dtf2[:,np.linspace(100,300,nf,dtype='int')].reshape([1,n1,nf,1])
#     dout=np.concatenate([dtf0,dtf1,dtf2],axis=3)
    
#     dds.append(dout)
    
# datanew=np.concatenate(dds,axis=0)


dt=0.01;freqs=np.linspace(w0,dw*(nw-1),nw);
print('Selected freqs for each comp:',freqs[np.linspace(100,300,nf,dtype='int')])

# In[10]:


# First run, please uncomment below

# np.save('datanew%d'%(nsample*2),datanew)
datanew=np.load('./datanew%d.npy'%(nsample*2))


# In[11]:


# datanew.shape
# labelnew=label[0:nsample*2]
labelnew=label;


# In[12]:


ind1=1;
plt.subplot(1,2,1)
plt.imshow(datanew[ind1,:,:,:].reshape([n1,nf*3]),aspect='auto')
plt.title('TF of waveform sample #%d (Signal)'%ind1)
plt.ylabel('Time (sample)');plt.xlabel('Frequency sample (%d freqs for each comp)'%nf);
ax=plt.subplot(1,6,4)
plt.plot(data[ind1,:,0],np.linspace(0,n1-1,n1),'k')
plt.gca().invert_yaxis();plt.setp(ax.get_yticklabels(), visible=False);plt.setp(ax.get_xticklabels(), visible=False);
plt.xlabel('Z');

ax=plt.subplot(1,6,5)
plt.plot(data[ind1,:,1],np.linspace(0,n1-1,n1),'k')
plt.gca().invert_yaxis();plt.setp(ax.get_yticklabels(), visible=False);plt.setp(ax.get_xticklabels(), visible=False);
plt.xlabel('N');

ax=plt.subplot(1,6,6)
plt.plot(data[ind1,:,2],np.linspace(0,n1-1,n1),'k')
plt.gca().invert_yaxis();plt.setp(ax.get_yticklabels(), visible=False);plt.setp(ax.get_xticklabels(), visible=False);
plt.xlabel('E');
plt.savefig('waveform1.png',format='png',dpi=300)
plt.show()



# In[13]:


ind1=201;
plt.subplot(1,2,1)
plt.imshow(datanew[ind1,:,:,:].reshape([n1,nf*3]),aspect='auto')
plt.title('TF of waveform sample #%d (Noise)'%ind1)
plt.ylabel('Time (sample)');plt.xlabel('Frequency sample (%d freqs for each comp)'%nf);
ax=plt.subplot(1,6,4)
plt.plot(data[ind1,:,0],np.linspace(0,n1-1,n1),'k')
plt.gca().invert_yaxis();plt.setp(ax.get_yticklabels(), visible=False);plt.setp(ax.get_xticklabels(), visible=False);
plt.xlabel('Z');

ax=plt.subplot(1,6,5)
plt.plot(data[ind1,:,1],np.linspace(0,n1-1,n1),'k')
plt.gca().invert_yaxis();plt.setp(ax.get_yticklabels(), visible=False);plt.setp(ax.get_xticklabels(), visible=False);
plt.xlabel('N');

ax=plt.subplot(1,6,6)
plt.plot(data[ind1,:,2],np.linspace(0,n1-1,n1),'k')
plt.gca().invert_yaxis();plt.setp(ax.get_yticklabels(), visible=False);plt.setp(ax.get_xticklabels(), visible=False);
plt.xlabel('E');
plt.savefig('waveform2.png',format='png',dpi=300)
plt.show()


# In[14]:
## TF-backed learning
niter=50
X_train, X_test, y_train, y_test = train_test_split(datanew, labelnew, test_size=0.2, random_state=42)
model = Sequential()
model.add(Flatten(input_shape=(6000, nf, 3)))
model.add(Dense(1000, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.summary()
model.compile('adam',loss='categorical_crossentropy',metrics=['accuracy'])
history1 = model.fit(X_train,y_train,epochs=niter,validation_data=([X_test],y_test), shuffle=True, batch_size=32)



# In[15]:
history=history1
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[16]:


## Non-TF-backed learning (smaller size)
X_train, X_test, y_train, y_test = train_test_split(data, labelnew, test_size=0.2, random_state=42)
model = Sequential()
model.add(Flatten(input_shape=(6000, 3)))
model.add(Dense(1000, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.summary()
model.compile('adam',loss='categorical_crossentropy',metrics=['accuracy'])
history2 = model.fit(X_train,y_train,epochs=niter,validation_data=([X_test],y_test), shuffle=True, batch_size=32)


# In[17]:


## Non-TF-backed learning (same size)

datanew2=datanew.copy();
for ii in range(nf):
    datanew2[:,:,ii,:] = data[:,:,:]


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(datanew2, labelnew, test_size=0.2, random_state=42)
model = Sequential()
model.add(Flatten(input_shape=(6000, nf, 3)))
model.add(Dense(1000, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.summary()
model.compile('adam',loss='categorical_crossentropy',metrics=['accuracy'])
history3 = model.fit(X_train,y_train,epochs=niter,validation_data=([X_test],y_test), shuffle=True, batch_size=32)


# In[19]:


## Non-TF-backed learning (smaller size, different network)
X_train, X_test, y_train, y_test = train_test_split(data, labelnew, test_size=0.2, random_state=42)
model = Sequential()
model.add(Conv1D(filters=16, kernel_size=3, input_shape=(6000, 3)))
model.add(MaxPooling1D(pool_size=3 ))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.summary()
model.compile('adam',loss='categorical_crossentropy',metrics=['accuracy'])
history4 = model.fit(X_train,y_train,epochs=niter,validation_data=([X_test],y_test), shuffle=True, batch_size=32)


# In[20]:
# history=history4
# list all data in history
## Plot training
plt.figure(figsize=(16, 12))
print(history1.history.keys())
# summarize history for accuracy
plt.plot(history1.history['accuracy'][0:niter],'r-',linewidth=2,label='Training (NTFA)')
plt.plot(history1.history['val_accuracy'][0:niter],'k-',linewidth=2,label='Validation (NTFA)')

plt.plot(history3.history['accuracy'][0:niter],'g--',linewidth=2,label='Training')
plt.plot(history3.history['val_accuracy'][0:niter],'c--',linewidth=2,label='Validation')

plt.title('Training accuracy comparison',size=16)
plt.ylabel('Accuracy',size=16)
plt.xlabel('Epoch',size=16)
plt.gca().xaxis.set_tick_params(labelsize=16)
plt.gca().yaxis.set_tick_params(labelsize=16)
plt.legend(loc='upper left',fontsize=16)
plt.savefig('training.png',format='png',dpi=300)
plt.show()


## For reproduce only
# np.save('my_history1.npy',history1.history)
# np.save('my_history3.npy',history3.history)
import numpy as np
import matplotlib.pyplot as plt
niter=30
history1=np.load('my_history1.npy',allow_pickle='TRUE').item()
history3=np.load('my_history3.npy',allow_pickle='TRUE').item()

plt.figure(figsize=(16, 12))
print(history1.keys())
# summarize history for accuracy
plt.plot(history1['accuracy'][0:niter],'r-',linewidth=2,label='Training (NTFA)')
plt.plot(history1['val_accuracy'][0:niter],'k-',linewidth=2,label='Validation (NTFA)')

plt.plot(history3['accuracy'][0:niter],'g--',linewidth=2,label='Training')
plt.plot(history3['val_accuracy'][0:niter],'c--',linewidth=2,label='Validation')

plt.title('Training accuracy comparison',size=16)
plt.ylabel('Accuracy',size=16)
plt.xlabel('Epoch',size=16)
plt.gca().xaxis.set_tick_params(labelsize=16)
plt.gca().yaxis.set_tick_params(labelsize=16)
plt.legend(loc='upper left',fontsize=16)
plt.savefig('training2.png',format='png',dpi=300)
plt.show()

# In[ ]:




