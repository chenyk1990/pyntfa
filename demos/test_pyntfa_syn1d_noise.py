import pywt
def spectrum_cwt(t, data, wavelet='mexh', widths=200, cmap='RdBu',
                 colorscale=1, contour_dvision=41, freqmax=100,
                 figsize=(12,3), plot=True,ymin=5, ymax=30, vmax=None):
    '''
    cwt: contineous wavelet transforms for seismic
    '''
    '''
    t: in ms
    data: 1d array
    wavelet:
        ['haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus', 'mexh',
        'morl', 'cgau', 'shan', 'fbsp', 'cmor'] short for:
        ['Haar', 'Daubechies', 'Symlets', 'Coiflets', 'Biorthogonal',
        'Reverse biorthogonal','Discrete Meyer(FIR Approximation)','Gaussian',
        'Mexican hat wavelet', 'Morlet wavelet', 'Complex Gaussian wavelets',
        'Shannon wavelets', 'Frequency B-Spline wavelets',
        'Complex Morlet wavelets']
    widths: the number of frequencies
    colorscale: change color scale, the smaller the darker the color
    contour_division: the number of color divided
    freqmax: the maximum of frequency
    plot: True or False
    '''
    t = np.array(t).reshape(len(t),)
    data = np.array(data).reshape(len(data),)
    
    widths = np.arange(1, widths)
    sampling_rate = 1000 * len(t)/(max(t) - min(t))
    cwtmatr, freqs = pywt.cwt(data, widths, wavelet, 1/sampling_rate)
    
    maximum = colorscale * max(abs(cwtmatr.max()), abs(cwtmatr.min()))
    if vmax != None:  maximum = vmax
    
    cwtmatr_c = np.clip(cwtmatr, -maximum, maximum, cwtmatr)
    levels = np.linspace(-maximum, maximum, contour_dvision)
    
    if plot==True:
        plt.figure(figsize=figsize)
        plt.contourf(t, freqs, cwtmatr_c, levels=levels, cmap=cmap)
        #plt.ylim(min(freqs), freqmax) #
        #plt.yscale('log')
        plt.ylim(ymin, ymax)
        #plt.xlim(t.min(), t.max()+1)
        plt.xlabel(u"Time(ms)")
        plt.ylabel(u"Frequency(Hz)")
        plt.title(u"Time-frequency Spectrum")
        plt.colorbar()
        
        plt.tight_layout()
        plt.show()
    
    
    return cwtmatr, freqs


from pyseistr import genflat
data=genflat();
n1=data.shape[0]
dt = 0.004
t = np.arange(n1)*dt*1000
cwtmatr, freqs =spectrum_cwt(t, data[:,1], wavelet='morl',
                 widths=100, cmap='RdBu', colorscale=1, contour_dvision=41,
                 freqmax=100, figsize=(12,3), plot=True, ymin=6, ymax=90)

import pyntfa as ntfa
#n1=din.shape[0]
dout,w0,dw,nw = ntfa.ntfa1d(data[:,1],dt=0.004,niter=10,rect=3,ifb=0,inv=0)
dout=dout.reshape([n1,nw,2],order='F');

#print(dout[:,:,0].max(),dout[:,:,0].min()) ##should be [0.0864,-0.0876]
## NOTE: dout[:,:,0] is imaginary and dout[:,:,1] is real

## plot TF spectrum
plt.figure();
plt.imshow(dout[:,:,0]*dout[:,:,0]+dout[:,:,1]*dout[:,:,1],cmap=plt.cm.jet, interpolation='none', extent=[0,nw*dw-dw,n1*1-1,0],aspect='auto');plt.xlabel('Frequency (Hz)');plt.ylabel('Time (s)');plt.title('Time-frequency Spectrum')
plt.show();



#        plt.figure(figsize=figsize)
#        plt.contourf(t, freqs, cwtmatr_c, levels=levels, cmap=cmap)
#        #plt.ylim(min(freqs), freqmax) #
#        #plt.yscale('log')
#        plt.ylim(ymin, ymax)
#        #plt.xlim(t.min(), t.max()+1)
#        plt.xlabel(u"Time(ms)")
#        plt.ylabel(u"Frequency(Hz)")
#        plt.title(u"Time-frequency Spectrum")
#        plt.colorbar()
#
#        plt.tight_layout()
#        plt.show()
