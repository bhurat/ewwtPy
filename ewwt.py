import numpy as np
from scipy.special import iv
from scipy.signal import convolve2d
from scipy import ndimage
from ewt.boundaries import *
from ewt.ewt1d import ewt_beta
from skimage.segmentation import watershed
from math import ceil,floor

"""
ewt2d_watershed(f,params)
Performs the empirical watershed wavelet transform on image f as proposed by 
Hurat et al (2021). 

Input:
    f       - Image to perform empirical watershed wavelet on
    params  - parameters for EWT (see utilities)
Output:
    ewtc    - empirical watershed wavelet coefficients
    mfb     - ewwt filter bank
    centers - detected maxima
Author: Basile Hurat"""

def ewt2dWatershed(f,params=ewt_params()):
    #check real or complex
    if(np.any(np.iscomplex(f))):
        params.complex = 1
    else:
        params.complex = 0
    
    #take Fourier transform
    ff = np.fft.fftshift(np.fft.fft2(f))
    [h,w] = ff.shape;
    extH = 0; extW = 0;
    
    if h%2 == 0 or w%2 == 0:
        ff_orig = np.copy(ff) #store original
        ff = np.zeros([h + (h%2 == 0), w + (w%2 == 0)],dtype='complex128')
        ff[0:h,0:w] = ff_orig
    if h%2 == 0:
        ff[h,:] = ff[0,:]
        extH = 1
    if w%2 == 0:
        ff[:,w] = ff[:,0]
        extW = 1
    absff = np.abs(ff)
    
    [L,centers] = ewt2d_watershedBoundariesDetect(absff,params,extH,extW)
    
    mfb = ewt2d_arbitraryFilterbank(L,np.unique(L),extH,extW,params.tau)
    
    ewtc = []
    for i in range(0,len(mfb)):
        if extH or extW:
            ewtc.append(np.real(np.fft.ifft2(np.fft.ifftshift(ff_orig*np.conj(mfb[i])))))
        else:
            ewtc.append(np.real(np.fft.ifft2(np.fft.ifftshift(ff*np.conj(mfb[i])))))
    return [ewtc,mfb,centers]

"""
iewt2d_watershed(ewtc,mfb,params)
Performs the inverse empirical watershed wavelet transform to reconstruct image
f from its empirical watershed wavelet and its corresponding ceofficients. 

Input:
    ewtc    - empirical watershed wavelet coefficients
    mfb     - corresponding empirical watershed wavelets
    params  - parameters for EWT (see utilities) 
                Needed for complex vs real image
Output:
    recon   - Reconstruction
Author: Basile Hurat"""
def iewt2dWatershed(ewtc,mfb,params=ewt_params()):
    img = np.zeros(ewtc[0].shape,dtype = 'complex128')
    dual_sum = np.zeros(ewtc[0].shape)
    for i in range(0,len(ewtc)):
        dual_sum += np.fft.ifftshift(mfb[i])**2
        img += np.fft.fft2(ewtc[i])*np.fft.ifftshift(mfb[i])
    img = img/dual_sum
    if params.complex == 0:
        recon = np.real(np.fft.ifft2(img))
    else:
        recon = np.fft.ifft(img)
    return recon

"""
ewt2d_watershedBoundariesDetect(f,params)
Given the magnitude spectrum of an image, this function partitions the Fourier
domain to capture modes based on the paper by Hurat et al (2021). This is done 
by first detecting persistent maxima using scale-space representations and then
using the watershed transform to separate those maxima through paths of lowest 
separation.

Input:
    absff   - magnitude spectrum of image (abs(fft(image)))
    params  - parameters for EWT (see utilities)
    extH    - Flag tied to horizontal extension of even dimensioned images
    extW    - Flag tied to vertical extension of even dimensioned images
Output:
    L       - Label image representing detected partition
    centers - detected maxima 
Author: Basile Hurat"""
def  ewt2d_watershedBoundariesDetect(absff,params,extH,extW):
    if params.log == 1:
        absff = np.log(absff)
    if  params.removeTrends.lower() != 'none':
        absff = EWT_RemoveTrends(absff,params)
    if params.spectrumRegularize != 'none':
        absff = EWT_spectrumRegularize(absff,params)
    
    centers = ewwt_getMaxima(absff,params,extH,extW)
    
    L = ewwt_getBoundaries(absff,centers,params)
    return [L,centers]

"""
ewwt_getMaxima(absff,params,extH,extW)
From the magnitude spectrum of an image, this function extracts persistent 
maxima using scale-space representations. It post-processes those maxima if 
necessary.

Input:
    absff   - Magnitude spectrum of image (abs(fft(image)))
    params  - parameters for EWT (see utilities)
    extH    - Flag tied to horizontal extension of even dimensioned images
    extW    - Flag tied to vertical extension of even dimensioned images
Output:
    centers - detected maxima 
Author: Basile Hurat"""
def ewwt_getMaxima(absff,params,extH,extW):
    [h,w] = absff.shape
    centers = gss2d(absff,params)
    
    #check to see if we must post-process found maxima
    if params.edges > 0 or params.includeCenter == 1:
        center_found = 0
        i = 0;
        while i < len(centers):
            if centers[i][0] == h//2 and centers[i][1] == w//2:
                center_found = 1
            if centers[i][0] <= params.edges or centers[i][1] <= params.edges:
                centers.remove(centers[i])
            elif centers[i][0] >= h-params.edges or centers[i][1] >= w-params.edges:
                centers.remove(centers[i])
            else: 
                i+= 1
        if center_found == 0:
            centers.append([h//2,w//2])
    
    #If height or width is extended, include none of the maxima on the edge in that dimension
    if extH == 1:
        i = 0
        while i < len(centers):
            if centers[i][0] == 0 or centers[i][0] == h-1:
                centers = np.delete(centers,i,axis = 0)
            else:
                i += 1
    if extW == 1:
        i = 0
        while i < len(centers):
            if centers[i][1] == 0 or centers[i][1] == w-1:
                centers = np.delete(centers,i,axis = 0)
            else:
                i += 1
    return centers

"""
ewt2d_watershedBoundariesDetect(absff,centers,params)
Given the magnitude spectrum of an image and its corresponding set of detected
persistent maxima, this function returns a watershed partitioning of the 
Fourier spectrum in the form of a Label image. 

Input:
    absff   - Magnitude spectrum of image (abs(fft(image)))
    centers - detected maxima
    params  - parameters for EWT (see utilities)
                Needed for complex vs real image
Output:
    L       - Label image representing detected partition
Author: Basile Hurat"""
def ewwt_getBoundaries(absff,centers,params):
    mask = np.zeros(absff.shape)
    for i in centers:
        mask[i[0],i[1]] = 1
    mask = ndimage.label(mask)[0]
    L = watershed(-absff,mask)
    #If real, mirror boundaries!
    if params.complex == 0:
        for i in range(0,L.shape[0]):
            for j in range(0,L.shape[1]):
                L[i,j] = min(L[i,j],L[L.shape[0]-i-1,L.shape[1]-j-1])
    return L
        
"""
ewt2d_arbitraryFilterbank(L,region_nums,extH,extW,tau)
Given a partitioning of the Fourier spectrum, this function constructs an 
array of empirical wavelets as per the construction in Hurat et al (2021)

Input:
    L           - Label image representing detected partition
    region_nums - detected maxima
    extH        - Flag tied to horizontal extension of even dimensioned images
    extW        - Flag tied to vertical extension of even dimensioned images
    tau         - transition width of empirical watershed wavelets
Output:
    mfb         - filterbank of empirical watershed wavelets
Author: Basile Hurat"""
def ewt2d_arbitraryFilterbank(L,region_nums,extH,extW,tau):
    mfb = []
    for i in range(0,len(region_nums)):
        mfb.append(ewt2d_arbitraryFilter(L,region_nums[i],tau))
    #Resize if extended
    if extH:
        for i in range(0,len(mfb)):
            mfb[i] = mfb[i][0:-1,:]
    if extW:
        for i in range(0,len(mfb)):
            mfb[i] = mfb[i][:,0:-1]
    
    [h,w] = mfb[0].shape
    #If extended, Resymmetrize Fourier transform
    if extH:
        s = np.zeros(w)
        if w%2 == 0:
            for i in range(0,len(mfb)):
                mfb[i][0,floor(w/2)+1:] += mfb[i][0,floor(w/2)-1:0:-1]
                mfb[i][0,1:floor(w/2)] = mfb[i][0,-1:floor(w/2):-1]
                s += mfb[i][0,:]**2
            #normalize for frame condition
            for i in range(0,len(mfb)):
                for j in range(1,w):
                    if j == floor(w/2):
                        continue
                    if s[j] > 0:
                        mfb[i][0,j] /= np.sqrt(s[j])
        else:
            for i in range(0,len(mfb)):
                mfb[i][0,floor(w/2)+1:] += mfb[i][0,floor(w/2)-1::-1]
                mfb[i][0,0:floor(w/2)] = mfb[i][0,-1:floor(w/2):-1]
                s += mfb[i][0,:]**2
            #normalize for frame condition
            for i in range(0,len(mfb)):
                for j in range(1,w):
                    if j == floor(w/2):
                        continue
                    if s[j] > 0:
                        mfb[i][0,j] /= np.sqrt(s[j])
    #repeat for widths
    if extW:
        s = np.zeros(h)
        if h%2 == 0:
            for i in range(0,len(mfb)):
                mfb[i][floor(h/2)+1:,0] += mfb[i][floor(h/2)-1:0:-1,0]
                mfb[i][1:floor(h/2),0] = mfb[i][-1:floor(h/2):-1,0]
                s += mfb[i][:,0]**2
                #normalize for frame condition
                for i in range(0,len(mfb)):
                    for j in range(1,h):
                        if j == floor(h/2):
                            continue
                        if s[j] > 0:
                            mfb[i][j,0] /= np.sqrt(s[j])
        else:
            for i in range(0,len(mfb)):
                mfb[i][floor(h/2)+1:,0] += mfb[i][floor(h/2)-1::-1,0]
                mfb[i][0:floor(h/2)-1,0] = mfb[i][-1:floor(h/2)+1:-1,0]
                s += mfb[i][:,0]**2
            #normalize for frame condition
            for i in range(0,len(mfb)):
                for j in range(0,h):
                    if j == floor(h/2):
                        continue
                    if s[j] > 0:
                        mfb[i][j,0] /= np.sqrt(s[j])
    return mfb

"""
ewt2d_arbitraryFilter(L,region_nums,extH,extW,tau)
Given a partitioning of the Fourier spectrum (in the form of a label image)
and a value corresponding to a region within that partitioning, this function
constructs a corresponding empirical watershed wavelet. 

Input:
    L       - Label image representing detected partition
    val     - integer related to desired region in partition
    tau     - transition width of empirical watershed wavelets
Output:
    filt    - filter of corresponding empirical watershed wavelet
Author: Basile Hurat"""
def ewt2d_arbitraryFilter(L,val,tau):
    filt = np.zeros(L.shape)
    dist = 2*np.pi*(-chamferDist(L==val,'quasieuclidean') + chamferDist(1-(L==val),'quasieuclidean'))/L.shape[0]
    for i in range(0,L.shape[0]):
        for j in range(0,L.shape[1]):
            if dist[i,j] > tau:
                filt[i,j] = 1
            elif abs(dist[i,j]) <= tau:
                filt[i,j] = np.cos(ewt_beta((tau - dist[i,j])/(2*tau))/2)
    return filt

"""
chamferDist(mask,distname):
The chamfering distance algorithm quickly performs a distance transform on a 
mask image. Given a mask (where pixels of interest equal 1), this function 
outputs a corresponding matrix where the element value is equal to the shortest
distance from that pixel to a region of interest (with regions of interest 
equaling 0).

Input:
    mask     - mask with regions of interest equaling 1
    distname - name of distance metric
                Options: quasieuclidean, cityblock
Output:
    dist     - matrix with element values equaling distance to closest region 
                of interest
Author: Basile Hurat"""      
def chamferDist(mask,distname = 'quasieuclidean'):
    dist = np.inf*np.ones([mask.shape[0]+2,mask.shape[1]+2])
    for i in range(0,mask.shape[0]):
            for j in range(0,mask.shape[1]):
                if mask[i,j] == 1:
                    dist[i+1,j+1] = 0
    if distname.lower() == "quasieuclidean":
        for i in range(0,mask.shape[0]):
            for j in range(0,mask.shape[1]):
                dist[i+1,j+1] = min(dist[i+1,j+1], dist[i,j]+np.sqrt(2)) #top left
                dist[i+1,j+1] = min(dist[i+1,j+1], dist[i,j+1] + 1) #top middle
                dist[i+1,j+1] = min(dist[i+1,j+1], dist[i+1,j] + 1) # middle left
                dist[i+1,j+1] = min(dist[i+1,j+1], dist[i+2,j]+np.sqrt(2)) # bottom left
        for i in range(1,mask.shape[0]+1):
            for j in range(1,mask.shape[1]+1):
                dist[-i-1,-j-1] = min(dist[-i-1,-j-1],dist[-i-2,-j]+np.sqrt(2)) #top right
                dist[-i-1,-j-1] = min(dist[-i-1,-j-1],dist[-i-1,-j] + 1) #middle right
                dist[-i-1,-j-1] = min(dist[-i-1,-j-1],dist[-i,-j-1] + 1) #bottom middle
                dist[-i-1,-j-1] = min(dist[-i-1,-j-1],dist[-i,-j]+np.sqrt(2)) #bottom right
        
    elif dist.lower() == "cityblock":
        for i in range(0,mask.shape[0]):
            for j in range(0,mask.shape[1]):
                dist[i+1,j+1] = min(dist[i+1,j+1], dist[i,j] + 1) #top left
                dist[i+1,j+1] = min(dist[i+1,j+1], dist[i,j+1] + 1) #top middle
                dist[i+1,j+1] = min(dist[i+1,j+1], dist[i+1,j] + 1) # middle left
                dist[i+1,j+1] = min(dist[i+1,j+1], dist[i+2,j] + 1) # bottom left
        for i in range(1,mask.shape[0]+1):
            for j in range(1,mask.shape[1]+1):
                dist[-i-1,-j-1] = min(dist[-i-1,-j-1],dist[-i-2,-j] + 1) #top right
                dist[-i-1,-j-1] = min(dist[-i-1,-j-1],dist[-i-1,-j] + 1) #middle right
                dist[-i-1,-j-1] = min(dist[-i-1,-j-1],dist[-i,-j-1] + 1) #bottom middle
                dist[-i-1,-j-1] = min(dist[-i-1,-j-1],dist[-i,-j] + 1) #bottom right
    return dist[1:-1,1:-1]
"""
gss2d(f,params)
This constructs a gaussian scale-space representation of input 2D function f.
It uses the discrete Gaussian kernel.

Input:
    f       - input function to perform gaussian 2D scale-space on
    params  - parameters for EWT (see utilities)
                Needed for scale-space stepsize t, kernel size n,
                number of iterations niter, detection method, and whether 
                image is complex or real
Output:
    ewtc    - empirical watershed wavelet coefficients
    mfb     - ewwt filter bank
    centers - detected maxima
Author: Basile Hurat"""
def gss2d(f,params):
    t = params.t
    n = params.n
    [h,w] = f.shape
    f=(f-np.min(f))/(np.max(f)-np.min(f))
    #build kernel
    ker = np.exp(-t)*iv(np.arange(-n,n+1),t)
    ker = ker/np.sum(ker)
    num_iter = int(params.niter*max(h,w)/n)
    #allocate plane
    plane = np.zeros([h, w, num_iter+1],'bool')
    
    #If f is complex, work only half the spectrum
    if params.complex == 0:
        f = f[0:int(np.ceil(h/2))+n,:]
        
    if params.complex == 0:
        Localmax = f == ndimage.maximum_filter(f,size = (3,3))
        for i in range(0,int(np.ceil(h/2))):
            for j in range(0,w):
                if Localmax[i,j] == 1:
                    plane[i,j,0] = 1
    else:
        plane[:,:,0] = f == ndimage.maximum_filter(f,size = (3,3))
    
    for i in range(0,num_iter):
        f =  np.pad(f,n,'reflect')
        f = convolve2d(f,ker[None],'same') 
        f = convolve2d(f,ker[None].T,'same') #wont work w/ 1d array
        f = f[n:-n,n:-n]
        
        #resymmetrize if working with real image
        if params.complex == 0:
            x = f.flatten()
            x[ceil(h*w/2):] = x[floor(h*w/2):floor(h*w/2)-floor(w/2)-n*w:-1]
            f = x.reshape([int(np.ceil(h/2))+n,w])
        if params.complex == 0:
            Localmax = f == ndimage.maximum_filter(f,size = (3,3))
            for j in range(0,int(np.ceil(h/2))):
                for k in range(0,w):
                    if Localmax[j,k] == 1:
                        plane[j,k,i+1] = 1
                        
        else:
            plane[:,:,i+1] = f == ndimage.maximum_filter(f,size = (3,3))
        
    #Remove Edges 
    plane[0,:,:] = 0
    plane[:,0,:] = 0
    plane[-1,:,:] = 0
    plane[:,-1,:] = 0
    
    #Track maixma through scale-space! 
    minima_locations = []
    minima_persistence = []
    for i in range(0,f.shape[0]):
        for j in range(0,f.shape[1]):
            if plane[i,j,0] == 1:
                minima_locations.append([i,j])
                worm_end = 0
                scale = 1
                while not worm_end:
                    #check 8 connectivity in next range
                    worm_end = 1
                    for p in range(-1,2):
                        for q in range(-1,2):
                            if i+p < 0 or i+p >= f.shape[0] or j+q < 0 or j+q >= f.shape[1] or scale > num_iter:
                                continue
                            if worm_end == 1 and plane[i+p,j+q,scale] == 1:
                                worm_end = 0    
                                scale += 1
                                i = i+p
                                j = j+q
                    if worm_end == 1:
                        minima_persistence.append(scale)
                        
    minima_persistence = np.array(minima_persistence)
    minima_locations = np.array(minima_locations)
    
    #Find threshold
    if params.typeDetect.lower() == 'otsu':    
        thresh = otsu(minima_persistence)
        detected_modes = minima_locations[minima_persistence >= thresh]
    elif params.typeDetect.lower() == 'mean':   
        thresh = np.ceil(np.mean(minima_persistence))
        detected_modes= minima_locations[minima_persistence >= thresh]
    elif params.typeDetect.lower() == 'empiricallaw':
        thresh = empiricalLaw(minima_persistence)
        detected_modes = minima_locations[minima_persistence >= thresh]
    elif params.typeDetect.lower() == 'halfnormal':
        thresh = halfNormal(minima_persistence)
        detected_modes = minima_locations[minima_persistence >= thresh]
    elif params.typeDetect.lower() == 'kmeans':
        clusters = ewtkmeans(minima_persistence,1000)
        upper_cluster = clusters[minima_persistence == max(minima_persistence)][0]
        detected_modes = minima_locations[clusters == upper_cluster]   
    
    #If real, mirror results
    for i in range(0,len(detected_modes)):
        if detected_modes[i,0] == h//2 and detected_modes[i,1] == w//2:
            continue
        detected_modes=np.vstack([detected_modes,[h-detected_modes[i,0]-1,w-detected_modes[i,1]-1]])
    return detected_modes

"""Deprecated! Using scipy's maximum_filter instead"""
def local_min2d(f,conn = 8):
    if conn != 8 and conn!= 4:
        print('Wrong connectivity value, assuming connectivity 8')
        conn = 81
    f2 =  np.pad(f,1,'reflect')
    [h,w] = f2.shape
    minima4 = np.zeros(f2.shape)
    for i in range(0,h):
        minima4[i,:] = localmin(f2[i,:])
    for j in range(0,w):
        minima4[:,j] *= localmin(f2[:,j])
    if conn == 4:
        return minima4[1:-1,1:-1]
    else:
        diags = localminDiags(f,diagdir = 0)*localminDiags(f,diagdir = 1)
        minima = minima4[1:-1,1:-1]*diags
        return minima

def localminDiags(f,diagdir):
    if diagdir == 1:
        f = f[-1::-1,:]
    [h,w] = f.shape
    diagmin = np.zeros(f.size)
    #Set corners equal to 1
    diagmin[-w] = 1; diagmin[w-1] = 1
    for i in range(-h+2,w-1):
        diags = np.diag(f,i)
        diags =  np.pad(diags,1,'reflect')
        diagminima = localmin(diags)
        if i < 0:
            diagmin[-i*w:w**2+2:w+1] = diagminima[1:-1]
        else:
            diagmin[i:(w-i)*(w+1):w+1] = diagminima[1:-1]
    diagmin = diagmin.reshape([h,w])
    if diagdir == 1:
        diagmin = diagmin[-1::-1,:]
    return diagmin