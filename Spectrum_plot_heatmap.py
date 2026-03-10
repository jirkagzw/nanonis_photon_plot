# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# main program starts here

# Import the necessary packages and modules
from dateutil import parser
import inspect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from scipy import signal
import glob, os
import operator
import nanonispy2 as nap
import scipy 
from scipy.optimize import curve_fit
import matplotlib.cm as cm
from sklearn.preprocessing import normalize
from skimage.filters import gaussian, laplace, median
from scipy.signal import convolve2d
from scipy.signal import savgol_filter
from scipy.constants import h, c, e
import matplotlib.ticker as ticker
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from scipy.signal import medfilt
import math
def E(wavelength):
   return (h*c)/(wavelength*e*1E-9)
def WL(E):
   return (h*c)/(E*e*1E-9)
import matplotlib.pyplot as plt
import pickle
import gzip
from matplotlib import gridspec
eV=1239.84193
cm1_per_eV = 8065.544  # 1 eV = 8065.544 cm⁻¹
plt.rcParams["font.family"] = "sans-serif"
cdict = {'red':   [[0.0,  0.0, 0.0],
                   [0.4,  0.1, 0.1],
                   [0.7,  0.24, 0.24],
                   #[0.9,  0.88, 0.88],
                   [1.0,  1.0, 1.0]],
         'green': [[0.0,  0.0, 0.0],
                   [0.3,  0.15, 0.15],
                   [0.75,  0.6, 0.6],
                   #[0.9,  0.88, 0.88],
                   [1.0,  1.0, 1.0]],
         'blue':  [[0.0,  0.0, 0.0],
                   [0.3,  0.38, 0.38],
                   [0.5,  0.6, 0.6],
                   [0.75,  0.83, 0.83],
                   #[0.9,  0.88, 0.88],
                   [1.0,  1.0, 1.0]]}
cmap_cold = matplotlib.colors.LinearSegmentedColormap('Cold1', segmentdata=cdict, N=256)

def filter_image(im,sigma):
    
#    im = normalize(im)
#    im = np.fliplr(im)
#    im = gaussian(im, sigma=5)
#    im = laplace(im)

#    num=3
#    im = convolve2d(im, np.ones((2*num, 1)) / 2 / num, mode='valid')
        
    im = gaussian(im, sigma=sigma)
    return(im)
    
def filter_2ddespike(im,tol): #filter image
    im_filt = median(im).tolist()
    im_res = median(im).tolist()
    im=im.tolist()
    for i in range(len(im)):
        for j in range(len(im[i])):
            try:
                if (im[i][j]/im_filt[i][j]-1)>tol:
                    im_res[i][j]=im_filt[i][j]  
                else:
                    im_res[i][j]=im[i][j]
            except ZeroDivisionError:
                im_res[i][j]=im_filt[i][j]
          #  print(im_res[i][j],"filtered",im[i][j],"original")
   # print(np.array(im_res))
    return np.array(im_res)
                
def despike_2d_adapt(image, kernel_size=5, sigma_thresh=5):
    """
    Removes cosmic rays and other spike-like artifacts from a 2D image.

    This function identifies outlier pixels by comparing them to a median-filtered
    version of the image. A pixel is flagged as a cosmic ray if it exceeds the
    local median by a specified number of standard deviations.

    Parameters:
    ----------
    image : np.ndarray
        The 2D input image as a NumPy array.
    kernel_size : int, optional
        The size of the square kernel used for the median filter.
        A larger kernel is better for removing larger artifacts but may
        be more aggressive on real, extended features. Default is 5.
    sigma_thresh : float, optional
        The number of standard deviations a pixel's value must be above
        the local median to be considered a cosmic ray. Default is 5.

    Returns:
    -------
    np.ndarray
        A new 2D array with the cosmic rays replaced by the values from
        the median-filtered image.
    """
    # Ensure the input is a NumPy array
    image = np.array(image)

    # 1. Create a median-filtered version of the image. This serves as a
    # smooth, "peak-free" model of the image.
    median_image = median_filter(image, size=kernel_size)

    # 2. Calculate the difference between the original image and the median image.
    # This difference map highlights sharp features like cosmic rays.
    difference_image = image - median_image

    # 3. Calculate a robust measure of the standard deviation (noise) from the
    # difference image. We use the median absolute deviation (MAD) scaled to be
    # equivalent to the standard deviation for a normal distribution.
    # We ignore positive outliers (the cosmic rays themselves) for a better estimate.
    # 1.4826 is the scaling factor for normally distributed data.
    noise = 1.4826 * np.median(np.abs(difference_image[difference_image < 0]))

    # Handle the case of zero noise to avoid division by zero
    if noise == 0:
        return image.copy()

    # 4. Identify the cosmic rays. A pixel is a cosmic ray if its value in the
    # difference image is greater than 'sigma_thresh' times the noise.
    cosmic_ray_mask = difference_image > sigma_thresh * noise

    # 5. Create a copy of the original image and replace the cosmic ray pixels
    # with the corresponding values from the median-filtered image.
    cleaned_image = image.copy()
    cleaned_image[cosmic_ray_mask] = median_image[cosmic_ray_mask]

    return cleaned_image

def despike_sigma(array, width=5, sigma_thresh=6, mode='both'):
    array = np.asarray(array, dtype=float)
    med = medfilt(array, kernel_size=width)
    resid = array - med
    mad = np.median(np.abs(resid))
    
    if mode == 'both':
        mask = np.abs(resid) > sigma_thresh * mad
    elif mode == 'pos':
        mask = resid > sigma_thresh * mad
    elif mode == 'neg':
        mask = resid < -sigma_thresh * mad
    else:
        raise ValueError("mode must be 'both', 'pos', or 'neg'")

    clean = array.copy()
    clean[mask] = med[mask]
    return clean, mask
                
def add_key(dict_obj, key, value):
    if key not in dict_obj.keys():
        dict_obj.update({key : value})
# Dictionary of strings and ints
def getsorteddat(path):
    """
        Return two lists containing the opened .sxm data.

        Parameters
        ----------
        path : string
            Path of data.

        Returns
        -------
        file_list: list
            List containing all file names in path.

        filedata_sxm: list
            List containing each .sxm image file opened with nanosipy.

        filedata_dat: list
            List containing each spectroscopy file opened with nanosipy.
            
        filedata_grid: list
            List containing each  grid file opened with nanosipy.
            

        Use Example
        -----
        file_list,filedata_sxm, filedata_dat, filedata_grid = getfiles(path)

    """
    file_list = os.listdir(path)
    filedata = []
    file_numbers = []
    file_names = []
    for file in file_list:
        if file.endswith(".dat") and file.startswith("AALS"): #or file.startswith("AKS") or file.startswith("AALS"):
        #if file.endswith(".dat") and file.startswith("LS"):
            file_name= str(file)
            file_number=int(file_name[(len(file_name)-9):(len(file_name)-4)])
            print(file)
            spec = nap.read.Spec(path + file)
            add_key(spec.header,"Filename",file_name[2:-4])
            try:
                add_key(spec.header,"Bias (V)",spec.header["Bias>Bias (V)"])
            except:
                pass
            try:
                add_key(spec.header,"Number of Accumulations",spec.header["GAN"])
            except KeyError:
                add_key(spec.header,"Number of Accumulations",1)
            filedata.append(spec)
            file_numbers.append(file_number)
            file_names.append(file_name)
            
        
    return file_list, filedata, file_numbers,file_names

def getsorteddatBS(path):
    """
example
        file_list,filedata_sxm, filedata_dat, filedata_grid = getfiles(path)
    """
    file_list = os.listdir(path)
    filedata = []
    file_numbers = []
    file_names = []
    for file in file_list:
        if file.endswith(".dat") and file.startswith("BS"): #or file.startswith("AKS") or file.startswith("AALS"):
        #if file.endswith(".dat") and file.startswith("LS"):
            file_name= str(file)
            file_number=int(file_name[(len(file_name)-9):(len(file_name)-4)])
           # print(file)
            spec = nap.read.Spec(path + file)
            add_key(spec.header,"Filename",file_name[0:-4])
            filedata.append(spec)
            file_numbers.append(file_number)
            file_names.append(file_name)              
    return file_list, filedata, file_numbers,file_names

    
overwrite=False
#path = "C:/Users/jirka/OneDrive - uochb.cas.cz/UOCHAB/data/2025/2025-10-30/" # PTCDA MgO  633 and 532 nm laser 1,2,3 ML 
path = "C:/Users/jirka/OneDrive - uochb.cas.cz/UOCHAB/data/2025/2025-07-08/" # PTCDA Raman-SQDM 633nm laser

pathcopy="C:/Users/Jirka/ownCloud/Documents/fzu/papers/ZnPc-cation/"
#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2020/2020-08-12/"
PIK = "pickle.dat"
pthPIK=path+PIK
if os.path.isfile(pthPIK) and overwrite==False:
    with gzip.open(pthPIK, "rb") as f:
        file_list=pickle.load(f)
        filedata_dat=pickle.load(f)
        file_numbers=pickle.load(f)
        file_names=pickle.load(f)
else:
    file_list, filedata_dat, file_numbers,file_names = getsorteddat(path) #now simple loading from AALS spectroscopy with nanonispy
    with gzip.open(pthPIK, "wb") as f:
        pickle.dump(file_list, f,protocol=-1)
        pickle.dump(filedata_dat, f,protocol=-1)
        pickle.dump(file_numbers, f,protocol=-1)
        pickle.dump(file_names, f,protocol=-1)
        
file_listBS, filedata_datBS, file_numbersBS,file_namesBS = getsorteddatBS(path) #now simple loading from BS spectroscopy with nanonispy
#files_sorted_time = sorted(
 #   filedata_dat,
 #   key=lambda filedata_dat: filedata_dat.header['Start time'])

def writesxm(path,filename, header, scan_par,channels,*argv): #writes nanonis file from map data
    fn = open(path+filename, mode='wb')
    fn.write((":NANONIS_VERSION:\n").encode('utf-8'))
    fn.write(("2\n").encode('utf-8'))
    fn.write((":SCANIT_TYPE:\n").encode('utf-8'))
    fn.write(("\tFLOAT\tMSBFIRST\n").encode('utf-8'))
    for key in scan_par:
        print(key)
        fn.write((":"+key+":"+"\n").encode('utf-8'))
        fn.write((scan_par[key]+"\n").encode('utf-8'))
    for key in header:
        fn.write((":"+key+":"+"\n").encode('utf-8'))
        fn.write((header[key]+"\n").encode('utf-8'))
    fn.write((":DATA_INFO:\n").encode('utf-8'))
    fn.write(("\tChannel\tName\tUnit\tDirection\tCalibration\tOffset\n").encode('utf-8'))
    for j in range (0,len(channels)):
        for i in range (0,len(channels[0])):
            fn.write(("\t").encode('utf-8'))
            fn.write((channels[j][i]).encode('utf-8'))
        fn.write(("\n").encode('utf-8'))   
    fn.write(("\n").encode('utf-8'))
    fn.write((":SCANIT_END:\n\n\n").encode('utf-8'))
    fn.write(bytes([26]))
    fn.write(bytes([4]))
    for d in argv:
        d.astype(">f4").tofile(fn)
        d.astype(">f4").tofile(fn)
  #  d2.astype(">f4").tofile(fn)
 #   d2.astype(">f4").tofile(fn)
 #   d3.astype(">f4").tofile(fn)
 #   d3.astype(">f4").tofile(fn)
  #  d4.astype(">f4").tofile(fn)
   # d4.astype(">f4").tofile(fn)
    #d5.astype(">f4").tofile(fn)
    #d5.astype(">f4").tofile(fn)
    #d6.astype(">f4").tofile(fn)
    #d6.astype(">f4").tofile(fn)
    #for i in range(0,len(data)):
       # print(i)
       # print(len(data[i,:,:]))
       # print(len(data[i,0,:]))
       # print(len(data[i,0,0]))
       # data[i,:,:].tofile(fn)
    #np.save(fn, data)
    
    
    
def load_AALS_data(path, overwrite=False):
    """
    Loads all AALS*.dat files from `path` (using pickle cache if available).
    Returns: (file_list, filedata_dat, file_numbers, file_names)
    """
    import gzip, pickle, os
    PIK = "pickle.dat"
    pthPIK = os.path.join(path, PIK)
    from nanonispy import read

    if os.path.isfile(pthPIK) and not overwrite:
        with gzip.open(pthPIK, "rb") as f:
            file_list = pickle.load(f)
            filedata_dat = pickle.load(f)
            file_numbers = pickle.load(f)
            file_names = pickle.load(f)
    else:
        file_list, filedata_dat, file_numbers, file_names = getsorteddat(path)
        with gzip.open(pthPIK, "wb") as f:
            pickle.dump(file_list, f, protocol=-1)
            pickle.dump(filedata_dat, f, protocol=-1)
            pickle.dump(file_numbers, f, protocol=-1)
            pickle.dump(file_names, f, protocol=-1)
    return file_list, filedata_dat, file_numbers, file_names

def find(lst, key, value):
    for i, dic in enumerate(lst):
        if dic[key] == value:
            return i
    return -1

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def nmtopix(x,lr,rr,ncol):
    res=(x-lr)/(rr-lr)*ncol
    res=int(res)
    if x<lr or res<0:
        return int(0)
    if x>rr or res>ncol:
        return int(ncol)
    else:
        return res
     
def eVtopix(x,lr,rr,ncol):
    res=(eV/x-lr)/(rr-lr)*ncol
    res=int(res)
    if eV/x<lr or res<0:
        return int(ncol-0)
    if eV/x>rr or res>ncol:
        return int(ncol-ncol)
    else:
        return int(ncol-res)
    
def eVtopixOK(x,lr,rr,ncol):
    res=(eV/x-lr)/(rr-lr)*ncol
    res=int(res)
    if eV/x<lr or res<0:
        return int(0)
    if eV/x>rr or res>ncol:
        return int(ncol)
    else:
        return res
    
def despike(array,tol,width,lr,rr,ncol):
    dtm=scipy.signal.medfilt(array, kernel_size=width)
    result=[]
    for i in range(0,len(array)):
        if i<lr or i>rr:
            result.append(array[i])
          #  print(i)
        elif (array[i]/dtm[i]-1)>tol:
            result.append(dtm[i])
        else:
            result.append(array[i])
    return result

def despike_multi(i,tol,width,lr,rr,ncol):
    my_dict=filedata_dat[i].signals
    filtered_dict = dict(filter(lambda item: "Counts nf" in item[0], my_dict.items()))
    x_f=[]
    for key in filtered_dict:
        x_f.append(despike(filtered_dict[key],tol,width,lr,rr,ncol))
    x_f=np.array(x_f)
    dim=len(x_f)
    if dim>2:
        counts_ar=[]
        for j in range (0,len(x_f[0,:])):
            x_med=np.median(x_f[:,j])
            tre=max((max(x_med-320,1))**0.5,10)
            #if tre>10:
            #    print(tre,"tre")
           # print(x_med,"median")
            counts=0
            division=0
            for i in range (0,len(x_f)):
                if x_f[i,j]<x_med+tre+5:
                    counts+=x_f[i,j]
                  #  print(x_f[i,j],"count",i,j)
                    division+=1
                else:
                    pass
            counts_ar.append(counts/division)
          #  print(counts/division,"res")
        counts_ar=despike(counts_ar,tol,width,lr,rr,ncol)
    elif dim==2:
        counts_ar=np.mean(x_f)
    else:
        counts_ar=filedata_dat[i].signals["Counts"]
    return counts_ar

def despike_multi_R(i,**kwargs):
    my_dict=filedata_dat[i].signals
    filtered_dict = dict(filter(lambda item: "Counts nf" in item[0], my_dict.items()))
    x_f=[]
    for key in filtered_dict:
        x_f.append(filtered_dict[key])
    x_f=np.array(x_f)
    dim=len(x_f)
    if dim>2:
        counts_ar=[]
        for j in range (0,len(x_f[0,:])):
            x_med=np.median(x_f[:,j])
            tre=max((max(x_med-320,1))**0.5,10)
            #if tre>10:
            #    print(tre,"tre")
           # print(x_med,"median")
            counts=0
            division=0
            for i in range (0,len(x_f)):
                if x_f[i,j]<x_med+tre+5:
                    counts+=x_f[i,j]
                  #  print(x_f[i,j],"count",i,j)
                    division+=1
                else:
                    pass
            counts_ar.append(counts/division)
          #  print(counts/division,"res")
    elif dim==2:
        counts_ar=np.mean(x_f)
    else:
        if "width" in kwargs:
            width=int(kwargs["width"])
            counts_ar=despike(filedata_dat[i].signals["Counts nf"],0.1,width,0,len(filedata_dat[i].signals['Wavelength (nm)']),len(filedata_dat[i].signals['Wavelength (nm)'])-1)
        else:
            counts_ar=filedata_dat[i].signals["Counts nf"]
    return counts_ar

#def gauss2(x,*p):
 #   A1,mu1,sigma1,A2,mu2,sigma2=p
  #  return A1*np.exp(-(x-mu1)**2/(2.*sigma1**2))+A2*np.exp(-(x-mu2)**2/(2.*sigma2**2))
def gauss2(x,*p):
    A1,mu1,sigma1,off=p
    return A1*np.exp(-(x-mu1)**2/(2.*sigma1**2))+off

def gauss(x,A,mu,sigma,off):
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))+off
def lorentz(x,A,mu,wid,off):
    return (A*0.5*wid)/(np.pi*(((x-mu)**2)+((0.5*wid)**2)))/(2/(np.pi*wid))+off
          
            
def bias_profile_pcol(i1, i2, x1, x2, gamma=1.0, contrast=False, unit="nm", save=True,bg=305,ker_size=5,sig_th=5,center_wl=632.8,plot_two_level=False, two_level_params=None):
    """
    example  
    bias_profile_pcol(i1="LS-PL-3ML-vdep-b-00001", i2="LS-PL-3ML-vdep-b-00051", x1=650, x2=700)
    
    Plot a quick heatmap of raw counts vs wavelength for scans from i1 to i2.

    Parameters:
    -----------
    i1, i2 : int
        Index range of scans to include.
    x1, x2 : float
        Wavelength (or energy) range.
    gamma : float
        Gamma correction for colormap.
    contrast : bool
        If True, normalize contrast per scan.
    unit : str
        'nm' for wavelength, 'eV' for energy.
    save : bool
        If True, saves the figure to path and pathcopy.

    Returns:
    --------
    None
    """
    if isfloat(i1)==True:
        i1=i1
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i1):
                i1=j
                break
    if isfloat(i2)==True:
        i2=i2
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i2):
                i2=j
                break
            
    lr = filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr = filedata_dat[i1].signals['Wavelength (nm)'][-1]
    ncol = len(filedata_dat[i1].signals['Wavelength (nm)']) - 1

    counts = []
    bias=[]
    for i in range(i1, i2 + 1):
        scan_counts = filedata_dat[i].signals['Counts']
        counts.append(scan_counts)
        bias.append(float(filedata_dat[i].header["Bias (V)"]))
    counts = np.array(counts)
    data = despike_2d_adapt(counts,kernel_size=ker_size,sigma_thresh=sig_th)-bg
    if contrast:
        data = (data - np.min(data, axis=1)[:, None]) / np.ptp(data, axis=1)[:, None]
        
    #cmap = matplotlib.cm.get_cmap('seismic')
    cmap = matplotlib.cm.get_cmap('gist_heat_r')
    matplotlib.colors.LinearSegmentedColormap.set_gamma(cmap, gamma)

    fig, ax = plt.subplots()

    if unit == "nm":
        x = filedata_dat[i1].signals['Wavelength (nm)'][nmtopix(x1, lr, rr, ncol):nmtopix(x2, lr, rr, ncol)]
        #y = np.arange(i2 - i1 + 1)
        y = np.array(bias)
        X, Y = np.meshgrid(x, y)
        plt.pcolormesh(X, Y, data[:, nmtopix(x1, lr, rr, ncol):nmtopix(x2, lr, rr, ncol)], cmap=cmap,norm=matplotlib.colors.PowerNorm(gamma=gamma))
        ax.set_xlabel('Wavelength [nm]')
    else:
        x_full = filedata_dat[i1].signals['Wavelength (nm)'][::-1]
        x_ev_full = eV / x_full
        # Find indices corresponding to x1 and x2 in eV
        ev1 = eV / x2  # x2 nm → lower energy
        ev2 = eV / x1  # x1 nm → higher energy
        x_mask = (x_ev_full >= ev1) & (x_ev_full <= ev2)
        x = x_ev_full[x_mask]
        data = data[:, ::-1]
        data = data[:, x_mask]
        #y = np.arange(i2 - i1 + 1)
        y = np.array(bias)
        X, Y = np.meshgrid(x, y)
        mesh=plt.pcolormesh(X, Y, data, cmap=cmap) #vector option
        mesh.set_rasterized(True)
        #plt.imshow(data, extent=[x1, x2, y.min(), y0.max()],
           #aspect='auto', cmap=cmap, origin='lower') #raster option
        ax.set_xlabel('Photon energy (eV)')
        ax.tick_params(axis='x', which='both', top=True, bottom=True)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5)) # <-- Add this line
        if unit=="cm":
            # Add top axis in Raman shift [cm⁻¹]
            def ev_to_raman(eV, E0=eV/center_wl):
                return (E0 - eV) * cm1_per_eV
            
            def raman_to_ev(raman, E0=eV/center_wl):
                return E0 - raman / cm1_per_eV
            
            # Create top axis for Raman shift
            secax = ax.secondary_xaxis('top', functions=(ev_to_raman, raman_to_ev))
            secax.set_xlabel('Raman shift (cm⁻¹)', color='red')
            
            # Set major and minor tick locations
            secax.xaxis.set_major_locator(ticker.AutoLocator())
            secax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))  # 5 minor ticks per major
            
            # Set color for ticks and tick labels
            secax.tick_params(axis='x', which='both', colors='red')  # sets tick lines and labels color
            
            # Optional: explicitly set minor tick label color if they're shown (usually they are not by default)
            for label in secax.get_xticklabels(minor=True):
                label.set_color('red')

    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.set_ylabel('Bias (V)')
    fig.set_size_inches(4.0, 3.0)
    plt.colorbar()
    
    # After plotting the heatmap (unit="eV")
    if unit == "eV" and plot_two_level and two_level_params is not None:
        # Extract two-level parameters
        e = np.array(two_level_params.get('e', [1.3293, 1.3293]))
        u = np.array(two_level_params.get('u', [-0.00082, -0.00082]))
        J = np.array(two_level_params.get('J', [[0,0.0015],[0.0015,0]]))
        
        # Use the same bias axis as the heatmap
        V_array = np.array(bias)
        
        # Compute eigenvalues for each bias
        Evals = np.zeros((len(V_array), 2))
        for i, v in enumerate(V_array):
            Eonsite = e + u * v
            H = np.diag(Eonsite) + J
            Evals[i] = np.linalg.eigvalsh(H)
        
        # Overlay the two-level curves
        plt.plot(Evals[:,0], V_array, lw=2, color='gray', label='Lower eigenvalue')
        plt.plot(Evals[:,1], V_array, lw=2, color='darkred', label='Upper eigenvalue')
        plt.legend(fontsize=6, loc='upper left')

    if save:
        filename = f"{filedata_dat[i1].header['Filename']}-{filedata_dat[i2].header['Filename']}"
        plt.savefig(path + filename + ".png", dpi=300, bbox_inches='tight', transparent=True)
        plt.savefig(path + filename + ".svg", dpi=300, bbox_inches='tight', transparent=True)
     #   plt.savefig(pathcopy + filename + "_quick.svg", dpi=300, bbox_inches='tight')

def bias_profile_pcol_flip(i1, i2, x1, x2, gamma=1.0, contrast=False, unit="nm", save=True, bg=305, ker_size=5, sig_th=5, center_wl=632.8, flip_axes=False,plot_two_level=False, two_level_params=None):
    """
    Plot a quick heatmap of raw counts vs wavelength or energy for scans from i1 to i2.

    Parameters:
    -----------
    i1, i2 : int or str
        Index range of scans to include (or filenames).
    x1, x2 : float
        Wavelength (nm) or photon energy (eV) range.
    gamma : float
        Gamma correction for colormap.
    contrast : bool
        Normalize contrast per scan.
    unit : str
        'nm' for wavelength, 'eV' for energy, 'cm' for Raman shift.
    save : bool
        If True, saves the figure.
    bg : float
        Background value to subtract.
    flip_axes : bool
        If True, flips axes (Bias on X, wavelength/energy on Y).
    """

    # Convert filename to index if needed
    if not isfloat(i1):
        for j in range(len(file_numbers)):
            if filedata_dat[j].header["Filename"] == str(i1):
                i1 = j
                break
    if not isfloat(i2):
        for j in range(len(file_numbers)):
            if filedata_dat[j].header["Filename"] == str(i2):
                i2 = j
                break

    # Get wavelength info
    lr = filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr = filedata_dat[i1].signals['Wavelength (nm)'][-1]
    ncol = len(filedata_dat[i1].signals['Wavelength (nm)']) - 1

    # Collect data
    counts = []
    bias = []
    for i in range(i1, i2 + 1):
        scan_counts = filedata_dat[i].signals['Counts']
        counts.append(scan_counts)
        bias.append(float(filedata_dat[i].header["Bias (V)"]))
    counts = np.array(counts)
    data = despike_2d_adapt(counts, kernel_size=ker_size, sigma_thresh=sig_th) - bg

    if contrast:
        data = (data - np.min(data, axis=1)[:, None]) / np.ptp(data, axis=1)[:, None]

    # Colormap
    cmap = matplotlib.cm.get_cmap('seismic')
    matplotlib.colors.LinearSegmentedColormap.set_gamma(cmap, gamma)

    fig, ax = plt.subplots()

    # Determine plot axis data
    y = np.array(bias)

    if unit == "nm":
        x_idx_start = nmtopix(x1, lr, rr, ncol)
        x_idx_end = nmtopix(x2, lr, rr, ncol)
        x = filedata_dat[i1].signals['Wavelength (nm)'][x_idx_start:x_idx_end]
        data = data[:, x_idx_start:x_idx_end]
    else:
        x_full = filedata_dat[i1].signals['Wavelength (nm)'][::-1]
        x_ev_full = eV / x_full
        ev1 = eV / x2  # x2 nm → lower energy
        ev2 = eV / x1  # x1 nm → higher energy
        x_mask = (x_ev_full >= ev1) & (x_ev_full <= ev2)
        x = x_ev_full[x_mask]
        data = data[:, ::-1]
        data = data[:, x_mask]

    # Handle flipped axes
    if flip_axes:
        X, Y = np.meshgrid(y, x)
        plt.pcolormesh(X, Y, data.T, cmap=cmap)
    else:
        X, Y = np.meshgrid(x, y)
        plt.pcolormesh(X, Y, data, cmap=cmap)

    # Labels and ticks
    if unit == "nm":
        if flip_axes:
            ax.set_xlabel('Bias (V)')
            ax.set_ylabel('Wavelength [nm]')
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        else:
            ax.set_xlabel('Wavelength [nm]')
            ax.set_ylabel('Bias (V)')
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    else:
        if flip_axes:
            ax.set_xlabel('Bias (V)')
            ax.set_ylabel('Photon energy (eV)')
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        else:
            ax.set_xlabel('Photon energy (eV)')
            ax.set_ylabel('Bias (V)')
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

        # Raman shift top axis only when energy is on X
        if unit == "cm" and not flip_axes:
            def ev_to_raman(ev, E0=eV / center_wl):
                return (E0 - ev) * cm1_per_eV

            def raman_to_ev(raman, E0=eV / center_wl):
                return E0 - raman / cm1_per_eV

            secax = ax.secondary_xaxis('top', functions=(ev_to_raman, raman_to_ev))
            secax.set_xlabel('Raman shift (cm⁻¹)', color='red')
            secax.xaxis.set_major_locator(ticker.AutoLocator())
            secax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
            secax.tick_params(axis='x', which='both', colors='red')
            for label in secax.get_xticklabels(minor=True):
                label.set_color('red')
                
    fig.set_size_inches(5, 2.5)
    plt.colorbar()
    
    # After plotting the heatmap (unit="eV")
    if unit == "eV" and plot_two_level and two_level_params is not None:
        # Extract two-level parameters
        e = np.array(two_level_params.get('e', [1.3293, 1.3293]))
        u = np.array(two_level_params.get('u', [-0.00082, -0.00082]))
        J = np.array(two_level_params.get('J', [[0,0.0015],[0.0015,0]]))
        
        # Use the same bias axis as the heatmap
        V_array = np.array(bias)
        
        # Compute eigenvalues for each bias
        Evals = np.zeros((len(V_array), 2))
        for i, v in enumerate(V_array):
            Eonsite = e + u * v
            H = np.diag(Eonsite) + J
            Evals[i] = np.linalg.eigvalsh(H)
        
        # Overlay the two-level curves
        plt.plot(Evals[:,0], V_array, lw=2, color='gray', label='Lower eigenvalue')
        plt.plot(Evals[:,1], V_array, lw=2, color='darkred', label='Upper eigenvalue')
        plt.legend(fontsize=6, loc='upper left')
    
    # Save
    if save:
        filename = f"{filedata_dat[i1].header['Filename']}-{filedata_dat[i2].header['Filename']}"
        plt.savefig(path + filename + ".png", dpi=300, bbox_inches='tight', transparent=True)
        plt.savefig(path + filename + ".svg", dpi=300, bbox_inches='tight', transparent=True) 
        plt.savefig(filename + ".png", dpi=300, bbox_inches='tight', transparent=True)
        plt.savefig(filename + ".svg", dpi=300, bbox_inches='tight', transparent=True)
        
bias_profile_pcol(
    i1="LS-PL-2mer-vdep-lside0-00001",
    i2="LS-PL-2mer-vdep-lside0-00022",
    x1=900,        # photon energy lower bound in eV
    x2=960,        # photon energy upper bound in eV
    unit="eV",
    contrast=True,
    gamma=1.5,
    plot_two_level=True,
    two_level_params={
        "e": [1.3293, 1.3293],          # onsite energies in eV
        "u": [-0.00082, -0.00082],      # Stark slopes (eV/V)
        "J": [[0, 0.0015],[0.0015, 0]]  # 2x2 coupling matrix
    }
)

bias_profile_pcol(
    'LS-PL-2mer-vdep-middle0-00001','LS-PL-2mer-vdep-middle0-00022',
    x1=900,        # photon energy lower bound in eV
    x2=960,        # photon energy upper bound in eV
    unit="eV",
    contrast=True,
    gamma=1.5,
    plot_two_level=True,
    two_level_params={
        "e": [1.3293, 1.3293],               # On-site energies in eV
        "u": [-0.00082, -0.00082],           # Stark slopes (eV/V)
        "J": [[0.0000, 0.0015],              # 2x2 coupling matrix
              [0.0015, 0.0000]]  # 2x2 coupling matrix
    }
)

