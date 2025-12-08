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
path = "C:/Users/jirka/OneDrive - uochb.cas.cz/UOCHAB/data/2025/2025-10-30/" # PTCDA MgO  633 and 532 nm laser 1,2,3 ML 
#path = "C:/Users/jirka/OneDrive - uochb.cas.cz/UOCHAB/data/2025/2025-10-16/" # PTCDA Raman-SQDM 785nm laser
#path = "C:/Users/jirka/OneDrive - uochb.cas.cz/UOCHAB/data/2025/2025-07-31/" # PTCDA Raman-SQDM 633nm laser
#path = "C:/Users/jirka/OneDrive - uochb.cas.cz/UOCHAB/data/2025/2025-07-30/" # PTCDA Raman-SQDM 633nm laser
#path = "C:/Users/jirka/OneDrive - uochb.cas.cz/UOCHAB/data/2025/2025-07-29/" # PTCDA Raman-SQDM 633nm laser
#path = "C:/Users/jirka/OneDrive - uochb.cas.cz/UOCHAB/data/2025/2025-07-03/" # PTCDA Raman-SQDM 633nm laser
#path = "C:/Users/jirka/OneDrive - uochb.cas.cz/UOCHAB/data/2025/2025-07-02/" # PTCDA Raman-SQDM 533nm laser
#path = "C:/Users/jirka/OneDrive - uochb.cas.cz/UOCHAB/data/2025/2025-07-01/" # PTCDA Raman-SQDM 533nm laser
#path = "C:/Users/jirka/OneDrive - uochb.cas.cz/UOCHAB/data/2025/2025-05-07/" # PTCDA dianion NaCl  
#path = "C:/Users/jirka/OneDrive - uochb.cas.cz/UOCHAB/data/2025/2025-05-16/" #hanging PTCDA 633 nm anion Raman
#path = "C:/Users/jirka/OneDrive - uochb.cas.cz/UOCHAB/data/2025/2025-05-14/" #mostly hanging PTCDA 533 nm dianion Raman
#path = "C:/Users/jirka/OneDrive - uochb.cas.cz/UOCHAB/data/2025/2025-04-24/" #hanging PTCDA PL 785 nm
#path = "C:/Users/jirka/OneDrive - uochb.cas.cz/UOCHAB/data/2025/2025-04-22/" #standing PTCDA
#path = "C:/Users/jirka/OneDrive - uochb.cas.cz/UOCHAB/data/2025-03-13/" #lanthanides
#path = "C:/Users/jirka/OneDrive - uochb.cas.cz/UOCHAB/data/2024-01-19/" #PTCDA NaCl PL
#path = "C:/Users/jirka/OneDrive - uochb.cas.cz/UOCHAB/data/2025/2025-02-14 [1]/" #MgPc MgO EL+PL
#path = "C:/Users/jirka/OneDrive - uochb.cas.cz/UOCHAB/data/2025/2025-01-14/" #PTCDA MgO EL+PL
#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2022/2022-05-19/"
#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2022/2022-05-18/"
#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2022/2022-05-19/"
#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2021/2021-11-08/"
#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2022/2022-02-18/"
#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2022/2022-02-16/"
#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2021/2021-06-30/"
#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2021/2022-02-18_2/"
#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2021/2021-07-02/"
#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2021/2021-11-10/"
#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2021/2021-03-18/"
#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2020/2020-12-06/"
#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2020/2020-12-20/"
#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2020/2020-09-15/"
#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2020/2020-09-17/"
#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2020/2020-09-17/cor1/"
#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2020/2020-08-28/"
#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2020/2020-08-25/"
#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2020/2020-07-29/"
#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2019/2019-12-14/"
#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2019/2019-09-23/" #fig 1 spectra
#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2019/2019-09-23/maps/" #fig 2 show
#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2019/2019-09-19/" #dimer maps
#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2019/2019-11-18/" #frustrated trimer maps
#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2020/2020-03-27/" 
#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2020/2020-03-24/" 
#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2020/2020-03-22/" 
#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2020/2020-03-25/" 
#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2019/2019-12-22/" # metal tip map
#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2020/2020-04-04/" # histograms
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


def bgcorplot(i,i_bg,x1,x2,unit,multi):  #converted 
    
    if isfloat(i)==True:
        i=i
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i):
                i=j
                break
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bg):
                i_bg=j
                break
            
    lr=filedata_dat[i].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i].signals['Wavelength (nm)'][-1]
    ncol=len(filedata_dat[i].signals['Wavelength (nm)'])-1
    n=float(filedata_dat[i].header["Number of Accumulations"])
    n_bg=float(filedata_dat[i_bg].header["Number of Accumulations"])
    fig, ax1 = plt.subplots() #osy 
    if unit=="nm":
        ax1.plot(filedata_dat[i].signals['Wavelength (nm)'],filedata_dat[i].signals['Counts']-n/n_bg*filedata_dat[i_bg].signals['Counts'],"",color="red", label=str(filedata_dat[i].header["Filename"])+" "+str(filedata_dat[i].header["Number of Accumulations"])+" " +str(i)+"B"+str(filedata_dat[i].header["Bias (V)"]))
        ax1.set_xlabel('wavelenght [nm]')
        ax1.set_xlim((x1, x2)) 
    if unit=="eV":
          ax1.plot(eV/filedata_dat[i].signals['Wavelength (nm)'],filedata_dat[i].signals['Counts']-n/n_bg*filedata_dat[i_bg].signals['Counts'],"", label=str(filedata_dat[i].header["Filename"])+" "+str(filedata_dat[i].header["Number of Accumulations"])+" " +str(i)+"B"+str(filedata_dat[i].header["Bias (V)"]))
          ax1.set_xlabel('Energy [eV]')
          ax1.set_xlim((eV/x1, eV/x2)) 
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_xlim((x1, x2))
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax22=False
       
    
    # get the primary axis x tick locations in plot units
    x1nm=math.floor(E(x1)*10)/10
    x2nm=math.ceil(E(x2)*10)/10
    x1nm_min=math.floor(E(x1)*50)/50
    x2nm_min=math.ceil(E(x2)*50)/50             
    xtl=np.linspace(x1nm,x2nm,int(round(abs(x2nm-x1nm)*10))+1)
    xtl_min=np.linspace(x1nm_min,x2nm_min,int(round(abs(x2nm_min-x1nm_min)*50))+1)
    xtickloc=[WL(x) for x in xtl]
    xtickloc_min=[WL(x) for x in xtl_min]
    ax1.set_ylabel('Photon intensity [a.u.]')
    #ax1.set_xlim((x1, x2)) 
    ax1.tick_params('y')
    #plt.legend(loc=4)
    ax1.set_yticklabels([])
    if ax22==True:
        ax2 = ax1.twiny()
        ax2.set_xticks(xtickloc)
        ax2.set_xticks(xtickloc_min, minor=True)
        # calculate new values for the second axis tick labels, format them, and set them
        x2labels = ['{:.3g}'.format(E(x)) for x in xtickloc]
        ax2.set_xticklabels(x2labels)
        # force the bounds to be the same
        ax2.set_xlim(ax1.get_xlim()) 
        ax2.set_xlabel('Energy [eV]') 
    ax1.grid(True,linestyle=':')
    x_size=4
    y_size=3.65
    fig.set_size_inches(x_size*multi/2.54,y_size*multi/2.54)
    plt.savefig(path+str(filedata_dat[i].header["Filename"])+"V"+str(filedata_dat[i].header["Bias (V)"])+".png", dpi=400, bbox_inches = 'tight',transparent=True)
            
#bgcorplot(1,2,500,800,"nm")
def bgcorplot_2(i,i2,i_bg,x1,x2,unit,save,path):  #converted  
    if isfloat(i)==True:
        i=i
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i):
                i=j
                break
            
    if isfloat(i2)==True:
        i2=i2
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i2):
                i2=j
                break
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bg):
                i_bg=j
                break
    lr=filedata_dat[i].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i].signals['Wavelength (nm)'][-1]
    ncol=len(filedata_dat[i].signals['Wavelength (nm)'])-1 
    n=float(filedata_dat[i].header["Number of Accumulations"])
    n2=float(filedata_dat[i2].header["Number of Accumulations"])
    n_bg=float(filedata_dat[i_bg].header["Number of Accumulations"])
    fig, ax1 = plt.subplots() #osy 
    if unit=="nm":
        ax1.plot(filedata_dat[i].signals['Wavelength (nm)'],(filedata_dat[i].signals['Counts']-n/n_bg*filedata_dat[i_bg].signals['Counts'])/abs(float(filedata_dat[i].header['Current avg. (A)'])*1E12),"", label="ZnPc",color="blue")
        ax1.plot(filedata_dat[i2].signals['Wavelength (nm)'],(filedata_dat[i2].signals['Counts']-n2/n_bg*filedata_dat[i_bg].signals['Counts'])/abs(float(filedata_dat[i2].header['Current avg. (A)'])*1E12),"", label="CuPc",color="red")
        ax1.set_xlabel('Wavelength [nm]')
        ax1.set_xlim((x1, x2)) 
        ax2 = ax1.twiny()
        ax2.set_xlim(E(x1),E(x2))
        ax2.set_xlabel('Energy [eV]')
        ax1.set_xlim((x1, x2)) 
    if unit=="eV":
          ax1.plot((eV/filedata_dat[i].signals['Wavelength (nm)'])-1.9156,(filedata_dat[i].signals['Counts']-n/n_bg*filedata_dat[i_bg].signals['Counts'])/np.amax(filedata_dat[i].signals['Counts'][nmtopix(640,lr,rr,ncol):nmtopix(750,lr,rr,ncol)]-n/n_bg*filedata_dat[i_bg].signals['Counts'][nmtopix(640,lr,rr,ncol):nmtopix(750,lr,rr,ncol)])+0.2,"", label="ZnPc 1.9156 eV",color="blue")
          ax1.plot((eV/filedata_dat[i2].signals['Wavelength (nm)'])-1.5187,(filedata_dat[i2].signals['Counts']-n2/n_bg*filedata_dat[i_bg].signals['Counts'])/np.amax(filedata_dat[i2].signals['Counts'][nmtopix(800,lr,rr,ncol):nmtopix(850,lr,rr,ncol)]-n2/n_bg*filedata_dat[i_bg].signals['Counts'][nmtopix(800,lr,rr,ncol):nmtopix(850,lr,rr,ncol)]),"",  label="ZnPc+ 1.5187 eV",color="red")
          ax1.set_xlabel('E-E($S_1$)[eV]')
          ax1.set_xlim((x1, x2)) 
          #majtick_spacing=0.01
          #ax1.xaxis.set_major_locator(ticker.MultipleLocator(majtick_spacing))
          #mintick_spacing=0.005
          #ax1.xaxis.set_minor_locator(ticker.MultipleLocator(mintick_spacing))
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Photon intensity [a.u.]')
    #ax1.set_ylim((0, 1.25)) 
    ax1.tick_params('y')
    plt.legend(loc="best")
    ax1.grid(True,linestyle=':')
    
    fig.set_size_inches(3.2*1.5,3*1.5)
    if save==True:
        plt.savefig(path+str(filedata_dat[i].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"V"+str(filedata_dat[i].header["Bias (V)"])+".png", dpi=400, bbox_inches = 'tight')

def bgcorplot_3(i,i2,i_bg,x1,x2,unit,save,path):  #converted  
    if isfloat(i)==True:
        i=i
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i):
                i=j
                break
            
    if isfloat(i2)==True:
        i2=i2
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i2):
                i2=j
                break
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bg):
                i_bg=j
                break
    lr=filedata_dat[i].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i].signals['Wavelength (nm)'][-1]
    ncol=len(filedata_dat[i].signals['Wavelength (nm)'])-1 
    n=float(filedata_dat[i].header["Number of Accumulations"])
    n2=float(filedata_dat[i2].header["Number of Accumulations"])
    n_bg=float(filedata_dat[i_bg].header["Number of Accumulations"])
    fig, ax1 = plt.subplots() #osy 
    if unit=="nm":
        ax1.plot(filedata_dat[i].signals['Wavelength (nm)'],(filedata_dat[i].signals['Counts']-n/n_bg*filedata_dat[i_bg].signals['Counts']), label="Bias=-2.8 V",color="blue")
        ax1.plot(filedata_dat[i2].signals['Wavelength (nm)'],(filedata_dat[i2].signals['Counts']-n2/n_bg*filedata_dat[i_bg].signals['Counts']), label="Bias=+2.8 V",color="red")
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_xlim((x1, x2)) 
        plt.legend(loc="best")
        ax2 = ax1.twiny()
        # get the primary axis x tick locations in plot units
        xtickloc = ax1.get_xticks() 
        # set the second axis ticks to the same locations
        ax2.set_xticks(xtickloc)
        # calculate new values for the second axis tick labels, format them, and set them
        x2labels = ['{:.3g}'.format(x) for x in E(xtickloc)]
        ax2.set_xticklabels(x2labels)
        # force the bounds to be the same
        ax2.set_xlim(ax1.get_xlim()) 
        ax2.set_xlabel('Energy (eV)')
    if unit=="eV":
          ax1.plot((eV/filedata_dat[i].signals['Wavelength (nm)']),(filedata_dat[i].signals['Counts']-n/n_bg*filedata_dat[i_bg].signals['Counts']),"", label="ZnPc 1.9156 eV",color="blue")
          ax1.plot((eV/filedata_dat[i2].signals['Wavelength (nm)']),(filedata_dat[i2].signals['Counts']-n2/n_bg*filedata_dat[i_bg].signals['Counts']),"",  label="ZnPc+ 1.5187 eV",color="red")
          ax1.set_xlabel('Energy (eV)')
          ax1.set_xlim((x1, x2)) 
          #majtick_spacing=0.01
          #ax1.xaxis.set_major_locator(ticker.MultipleLocator(majtick_spacing))
          #mintick_spacing=0.005
          #ax1.xaxis.set_minor_locator(ticker.MultipleLocator(mintick_spacing))
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Photon intensity (a.u.)')
    plt.legend()
    #ax1.set_ylim((0, 1.25)) 
    ax1.tick_params('y')
    ax1.grid(True,linestyle=':')
    
    fig.set_size_inches(3.2*1.5,3*1.5)
    if save==True:
        plt.savefig(path+str(filedata_dat[i].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"V"+str(filedata_dat[i].header["Bias (V)"])+".png", dpi=400, bbox_inches = 'tight')

def bgcorplot_div(i,i_bg,i_div,i_bgdiv,x1,x2,unit,save,path,label,norm,**kwargs):  #converted  
    if isfloat(i)==True:
        i=i
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i):
                i=j
                break
            
   # if isfloat(i2)==True:
     #   i2=i2
   #else:
     #   for j in range (0,len(file_numbers)):
      #      if filedata_dat[j].header["Filename"]==str(i2):
      #          i2=j
      #          break
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bg):
                i_bg=j
                break

    if isfloat(i_div)==True:
        i_div=i_div
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_div):
                i_div=j
                break
    if isfloat(i_bgdiv)==True:
        i_bgdiv=i_bgdiv
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bgdiv):
                i_bgdiv=j
                break
            

    lr=filedata_dat[i].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i].signals['Wavelength (nm)'][-1]    
    ncol=len(filedata_dat[i].signals['Wavelength (nm)'])-1
    n=float(filedata_dat[i].header["Number of Accumulations"])
   # n2=float(filedata_dat[i2].header["Number of Accumulations"])
    n_bg=float(filedata_dat[i_bg].header["Number of Accumulations"])
    if "bg" in kwargs:
        bgcounts=float(kwargs["bg"])
        bg=np.full(ncol+1,bgcounts)
    else:
        bg=savgol_filter(filedata_dat[i_bg].signals['Counts'],31,0)
        
    fig, ax1 = plt.subplots() #osy 

    norm_spec=savgol_filter(filedata_dat[i_div].signals['Counts'],31,0) - np.min(filedata_dat[i_div].signals['Counts'])#savgol_filter(filedata_dat[i_bgdiv].signals['Counts'],31,0)
    if norm==True:
       # data=(filedata_dat[i].signals['Counts nf']-n/n_bg*savgol_filter(filedata_dat[i_bg].signals['Counts'],31,0))/norm_spec
        data=(filedata_dat[i].signals['Counts']-n/n_bg*savgol_filter(despike(bg,0.1,7,nmtopix(700,lr,rr,ncol),nmtopix(940,lr,rr,ncol),ncol),31,0))/norm_spec
      #  data2=(filedata_dat[i+1].signals['Counts']-n/n_bg*savgol_filter(bg,31,0))/norm_spec
       # data3=(filedata_dat[i+2].signals['Counts']-n/n_bg*savgol_filter(bg,31,0))/norm_spec
        #data=despike(data,0.1,7,nmtopix(700,lr,rr,ncol),nmtopix(940,lr,rr,ncol))    
    else:
       # data=(filedata_dat[i].signals['Counts nf']-n/n_bg*savgol_filter(filedata_dat[i_bg].signals['Counts'],31,0))
        if "nf" in kwargs:
            try:
                data=(filedata_dat[i].signals['Counts nf'])
            except:
                data=(filedata_dat[i].signals['Counts nf 1'])
            if type(kwargs["nf"]) == int or type(kwargs["nf"]) == float:
                data=despike(data,0.1,int(kwargs["nf"]),380,400,ncol)
            data=data-n/n_bg*savgol_filter(bg,31,0)
        else:
            data=(despike_multi(i,0.2,7,lr,rr,ncol)-n/n_bg*savgol_filter(bg,31,0))
            data2=(filedata_dat[i].signals['Counts']-n/n_bg*savgol_filter(bg,31,0))
       # data2=(filedata_dat[i+1].signals['Counts']-n/n_bg*savgol_filter(bg,31,0))
        #data3=(filedata_dat[i+2].signals['Counts']-n/n_bg*savgol_filter(bg,31,0))
    if "savgol" in kwargs:
            data=savgol_filter(data,int(kwargs["savgol"]),0)
    if unit=="nm":
        ax1.plot(filedata_dat[i].signals['Wavelength (nm)'],(data),"", label=label,color="blue")
        #ax1.plot(filedata_dat[i].signals['Wavelength (nm)'],(norm_spec),"", label="plasmon",color="red")
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_xlim((x1, x2)) 
        ax2 = ax1.twiny()
        # get the primary axis x tick locations in plot units
        xtickloc = ax1.get_xticks() 
        # set the second axis ticks to the same locations
        ax2.set_xticks(xtickloc)
        # calculate new values for the second axis tick labels, format them, and set them
        x2labels = ['{:.3g}'.format(x) for x in E(xtickloc)]
        ax2.set_xticklabels(x2labels)
        # force the bounds to be the same
        ax2.set_xlim(ax1.get_xlim()) 
        ax2.set_xlabel('Energy (eV)')
    if unit=="eV":
            lw=1
            normc=np.max(savgol_filter(data,15,0))
            print(normc)
           # ax1.plot(eV/filedata_dat[i].signals['Wavelength (nm)'],savgol_filter(data,15,0)/normc,"", label=label,linewidth=lw,color="blue")
           # ax1.plot(eV/filedata_dat[i].signals['Wavelength (nm)'],(data)/normc,"", label=label,linewidth=lw*0.3,color="blue")
            ax1.plot(eV/filedata_dat[i].signals['Wavelength (nm)'],(data),"", label=label,color="blue",linewidth=lw)
          #  ax1.plot(eV/filedata_dat[i].signals['Wavelength (nm)'],(data2),"", label=label,color="red",linewidth=lw)
            #ax1.plot(eV/filedata_dat[i].signals['Wavelength (nm)'],(data),"", marker='.',label=label,color="blue",linestyle="None")
           # ax1.plot(eV/filedata_dat[i+1].signals['Wavelength (nm)'],(data2),"", label=label,color="red",linewidth=lw)
           # ax1.plot(eV/filedata_dat[i+2].signals['Wavelength (nm)'],(data3),"", label=label,color="black",linewidth=lw)
             #ax1.plot(eV/filedata_dat[i].signals['Wavelength (nm)'],(data),"",color="red",label=label)
            ## ax1.plot(eV/filedata_dat[i].signals['Wavelength (nm)'],norm_spec*np.amax(data[nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])*0.5/np.amax(norm_spec[nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])/10,"",color="black",ls="--",label="plasmon",linewidth=lw)
             #ax1.plot(filedata_dat[i].signals['Wavelength (nm)'],(norm_spec),"", label="plasmon",color="red")
            if "fit" in kwargs:
               # ax1.plot(eV/filedata_dat[i].signals['Wavelength (nm)'],lorentz(eV/filedata_dat[i].signals['Wavelength (nm)'],621.341,1.39405,0.00083182,0),"", label=label,color="black",linewidth=lw) #H2Pc 00077 3ML
                #ax1.plot(eV/filedata_dat[i].signals['Wavelength (nm)'],lorentz(eV/filedata_dat[i].signals['Wavelength (nm)'],921.683,1.51759,0.000650313,0),"", label=label,color="black",linewidth=lw) #ZnPc idep00001 4ML
                ax1.plot(eV/filedata_dat[i].signals['Wavelength (nm)'],lorentz(eV/filedata_dat[i].signals['Wavelength (nm)'],196.362,1.52001,0.00120571,0),"", label=label,color="black",linewidth=lw)   #ZnPc 3ML LS00003
                  
            ax1.set_xlabel('Photon energy (eV)')
            f= open(path+str(filedata_dat[i].header["Filename"])+"V"+str(filedata_dat[i].header["Bias (V)"])+"-"+label+".txt","w+")
            f.write("Photon energy (eV)"+'\t'+"Photon intensity ZnPc [a.u]" +"\n")#+'\t'+"Photon intensity plasmon [a.u]"+"\n")
            for index in range(len(data)-1,-1,-1):
                f.write(str("{0:.5f}".format(eV/filedata_dat[i].signals['Wavelength (nm)'][index])) +"\t" +str("{0:.5f}".format(data[index]))+"\n")#+"\t" +str("{0:.5f}".format(norm_spec[index]*np.amax(data)*0.5/np.amax(norm_spec[nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])))  + "\n")
             #majtick_spacing=0.1
             #ax1.xaxis.set_major_locator(ticker.MultipleLocator(majtick_spacing))
             #mintick_spacing=0.05
            f.close() 
            ax1.set_xlim((E(x2), E(x1)))
            ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax2 = ax1.twiny()
            # get the primary axis x tick locations in plot units
            if "maj_int2" in kwargs:
                maj_int2=float(kwargs["maj_int2"])
            else:
                maj_int2=100
            if "min_int2" in kwargs:
                min_int2=float(kwargs["min_int2"])
            else:
                min_int2=20
           # maj_int2=100
            #min_int2=20
            x1nm=math.ceil((x1)/maj_int2)*maj_int2
            x2nm=math.floor((x2)/maj_int2)*maj_int2
            x1nm_min=math.ceil(x1/min_int2)*min_int2
            x2nm_min=math.floor(x2/min_int2)*min_int2

            xtl=np.linspace(x1nm,x2nm,int(round(abs(x2nm-x1nm)/maj_int2))+1)

            xtl_min=np.linspace(x1nm_min,x2nm_min,int(round(abs(x2nm_min-x1nm_min)/min_int2))+1)

            xtickloc=[E(x) for x in xtl]
            xtickloc_min=[E(x) for x in xtl_min]
            # set the second axis ticks to the same locations
            ax2.set_xticks(xtickloc)
            ax2.set_xticks(xtickloc_min, minor=True)
            # calculate new values for the second axis tick labels, format them, and set them
            x2labels = ['{:.0f}'.format(WL(x)) for x in xtickloc]
            ax2.set_xticklabels(x2labels)
            # force the bounds to be the same
            ax2.set_xlim(ax1.get_xlim()) 
            ax2.set_xlabel('Wavelength (nm)') 
            
    if unit=="cm":
            lw=1
            lwl=632.8
            ax1.plot(1E7/lwl-1E7/filedata_dat[i].signals['Wavelength (nm)'],(data),"", label=label,color="blue",linewidth=lw)
            ax1.set_xlabel(r'Raman shift  ($\mathrm{cm}^-1$)')
            ax1.set_xlim((x1,x2)) 

    ax1.set_ylabel('Normalized intensity (arb. u.)')
    ax1.set_ylim((-10,1.1*np.max(data[nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)]))) 
  #  ax1.set_ylim((-0.1,1.2)) 
    ax1.tick_params('y')
   # ax1.yaxis.set_ticks_position('both')
   # ax1.xaxis.set_ticks_position('both')
    if "maj_int" in kwargs:
       maj_int=float(kwargs["maj_int"])
    else:
        maj_int=0.05
    if "min_int" in kwargs:
       min_int=float(kwargs["min_int"])
    else:
        min_int=0.01
    ax1.xaxis.set_major_locator(plt.MultipleLocator(maj_int))
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(min_int))
   ## ax1.yaxis.set_major_locator(plt.MultipleLocator(5))
   ## ax1.yaxis.set_minor_locator(plt.MultipleLocator(2.5))
    #plt.legend(loc="best")
    #ax1.grid(True,linestyle=':')
  #  ax1.set_ylim((-0.05, 5.5)) 
   ## ax1.set_ylim((0-np.amax(data[nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])/300, np.amax(data[nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])/10+np.amax(data[nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])/300)) 
    fig.set_size_inches(4.5,3)
    if save==True:
        plt.savefig(path+str(filedata_dat[i].header["Filename"])+"V"+str(filedata_dat[i].header["Bias (V)"])+str(x2)+"-"+str(x1)+"-"+label+".png", dpi=400, bbox_inches = 'tight',transparent=True)
        plt.savefig(path+str(filedata_dat[i].header["Filename"])+"V"+str(filedata_dat[i].header["Bias (V)"])+str(x2)+"-"+str(x1)+"-"+label+".svg", dpi=400, bbox_inches = 'tight',transparent=True)
#bgcorplot(1,2,500,800,"nm")

def bgcorplot_divb(i,i_bg,i_div,i_bgdiv,x1,x2,unit,save,path,label,norm,**kwargs):  #converted  
    if isfloat(i)==True:
        i=i
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].basename==str(i):
                i=j
                break
            
   # if isfloat(i2)==True:
     #   i2=i2
   #else:
     #   for j in range (0,len(file_numbers)):
      #      if filedata_dat[j].header["Filename"]==str(i2):
      #          i2=j
      #          break
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].basename==str(i_bg):
                i_bg=j
                break

    if isfloat(i_div)==True:
        i_div=i_div
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].basename==str(i_div):
                i_div=j
                break
    if isfloat(i_bgdiv)==True:
        i_bgdiv=i_bgdiv
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].basename==str(i_bgdiv):
                i_bgdiv=j
                break
            

    lr=filedata_dat[i].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i].signals['Wavelength (nm)'][-1]    
    ncol=len(filedata_dat[i].signals['Wavelength (nm)'])-1
    n=float(filedata_dat[i].header["Number of Accumulations"])
   # n2=float(filedata_dat[i2].header["Number of Accumulations"])
    n_bg=float(filedata_dat[i_bg].header["Number of Accumulations"])
    if "bg" in kwargs:
        bgcounts=float(kwargs["bg"])
        bg=np.full(ncol+1,bgcounts)
    else:
        bg=savgol_filter(filedata_dat[i_bg].signals['Counts'],31,0)
        
    fig, ax1 = plt.subplots() #osy 

    norm_spec=savgol_filter(filedata_dat[i_div].signals['Counts'],31,0) - np.min(filedata_dat[i_div].signals['Counts'])#savgol_filter(filedata_dat[i_bgdiv].signals['Counts'],31,0)
    if norm==True:
       # data=(filedata_dat[i].signals['Counts nf']-n/n_bg*savgol_filter(filedata_dat[i_bg].signals['Counts'],31,0))/norm_spec
        data=(filedata_dat[i].signals['Counts']-n/n_bg*savgol_filter(despike(bg,0.1,7,nmtopix(700,lr,rr,ncol),nmtopix(940,lr,rr,ncol),ncol),31,0))/norm_spec
      #  data2=(filedata_dat[i+1].signals['Counts']-n/n_bg*savgol_filter(bg,31,0))/norm_spec
       # data3=(filedata_dat[i+2].signals['Counts']-n/n_bg*savgol_filter(bg,31,0))/norm_spec
        #data=despike(data,0.1,7,nmtopix(700,lr,rr,ncol),nmtopix(940,lr,rr,ncol))    
    else:
       # data=(filedata_dat[i].signals['Counts nf']-n/n_bg*savgol_filter(filedata_dat[i_bg].signals['Counts'],31,0))
        if "nf" in kwargs:
            data=(filedata_dat[i].signals['Counts nf'])
            if type(kwargs["nf"]) == int or type(kwargs["nf"]) == float:
                data=despike(data,0.1,int(kwargs["nf"]),380,400,ncol)
            data=data-n/n_bg*savgol_filter(bg,31,0)
        else:
            data=(despike_multi(i,0.2,7,lr,rr,ncol)-n/n_bg*savgol_filter(bg,31,0))
            data2=(filedata_dat[i].signals['Counts']-n/n_bg*savgol_filter(bg,31,0))
       # data2=(filedata_dat[i+1].signals['Counts']-n/n_bg*savgol_filter(bg,31,0))
        #data3=(filedata_dat[i+2].signals['Counts']-n/n_bg*savgol_filter(bg,31,0))
    if unit=="nm":
        ax1.plot(filedata_dat[i].signals['Wavelength (nm)'],(data),"", label=label,color="blue")
        #ax1.plot(filedata_dat[i].signals['Wavelength (nm)'],(norm_spec),"", label="plasmon",color="red")
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_xlim((x1, x2)) 
        ax2 = ax1.twiny()
        # get the primary axis x tick locations in plot units
        xtickloc = ax1.get_xticks() 
        # set the second axis ticks to the same locations
        ax2.set_xticks(xtickloc)
        # calculate new values for the second axis tick labels, format them, and set them
        x2labels = ['{:.3g}'.format(x) for x in E(xtickloc)]
        ax2.set_xticklabels(x2labels)
        # force the bounds to be the same
        ax2.set_xlim(ax1.get_xlim()) 
        ax2.set_xlabel('Energy (eV)')
    if unit=="eV":
            lw=1
            ax1.plot(eV/filedata_dat[i].signals['Wavelength (nm)'],(data),"", label=label,color="blue",linewidth=lw)
          #  ax1.plot(eV/filedata_dat[i].signals['Wavelength (nm)'],(data2),"", label=label,color="red",linewidth=lw)
            #ax1.plot(eV/filedata_dat[i].signals['Wavelength (nm)'],(data),"", marker='.',label=label,color="blue",linestyle="None")
           # ax1.plot(eV/filedata_dat[i+1].signals['Wavelength (nm)'],(data2),"", label=label,color="red",linewidth=lw)
           # ax1.plot(eV/filedata_dat[i+2].signals['Wavelength (nm)'],(data3),"", label=label,color="black",linewidth=lw)
             #ax1.plot(eV/filedata_dat[i].signals['Wavelength (nm)'],(data),"",color="red",label=label)
            ## ax1.plot(eV/filedata_dat[i].signals['Wavelength (nm)'],norm_spec*np.amax(data[nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])*0.5/np.amax(norm_spec[nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])/10,"",color="black",ls="--",label="plasmon",linewidth=lw)
             #ax1.plot(filedata_dat[i].signals['Wavelength (nm)'],(norm_spec),"", label="plasmon",color="red")
            if "fit" in kwargs:
               # ax1.plot(eV/filedata_dat[i].signals['Wavelength (nm)'],lorentz(eV/filedata_dat[i].signals['Wavelength (nm)'],621.341,1.39405,0.00083182,0),"", label=label,color="black",linewidth=lw) #H2Pc 00077 3ML
                #ax1.plot(eV/filedata_dat[i].signals['Wavelength (nm)'],lorentz(eV/filedata_dat[i].signals['Wavelength (nm)'],921.683,1.51759,0.000650313,0),"", label=label,color="black",linewidth=lw) #ZnPc idep00001 4ML
                ax1.plot(eV/filedata_dat[i].signals['Wavelength (nm)'],lorentz(eV/filedata_dat[i].signals['Wavelength (nm)'],196.362,1.52001,0.00120571,0),"", label=label,color="black",linewidth=lw)   #ZnPc 3ML LS00003
                  
            ax1.set_xlabel('Photon energy (eV)')
            f= open(path+str(filedata_dat[i].header["Filename"])+"V"+str(filedata_dat[i].header["Bias (V)"])+"-"+label+".txt","w+")
            f.write("Photon energy (eV)"+'\t'+"Photon intensity ZnPc [a.u]" +"\n")#+'\t'+"Photon intensity plasmon [a.u]"+"\n")
            for index in range(len(data)-1,-1,-1):
                f.write(str("{0:.5f}".format(eV/filedata_dat[i].signals['Wavelength (nm)'][index])) +"\t" +str("{0:.5f}".format(data[index]))+"\n")#+"\t" +str("{0:.5f}".format(norm_spec[index]*np.amax(data)*0.5/np.amax(norm_spec[nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])))  + "\n")
             #majtick_spacing=0.1
             #ax1.xaxis.set_major_locator(ticker.MultipleLocator(majtick_spacing))
             #mintick_spacing=0.05
            f.close() 
            ax1.set_xlim((E(x2), E(x1)))
            ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax2 = ax1.twiny()
            # get the primary axis x tick locations in plot units
            if "maj_int2" in kwargs:
                maj_int2=float(kwargs["maj_int2"])
            else:
                maj_int2=100
            if "min_int2" in kwargs:
                min_int2=float(kwargs["min_int2"])
            else:
                min_int2=20
           # maj_int2=100
            #min_int2=20
            x1nm=math.ceil((x1)/maj_int2)*maj_int2
            x2nm=math.floor((x2)/maj_int2)*maj_int2
            x1nm_min=math.ceil(x1/min_int2)*min_int2
            x2nm_min=math.floor(x2/min_int2)*min_int2

            xtl=np.linspace(x1nm,x2nm,int(round(abs(x2nm-x1nm)/maj_int2))+1)

            xtl_min=np.linspace(x1nm_min,x2nm_min,int(round(abs(x2nm_min-x1nm_min)/min_int2))+1)

            xtickloc=[E(x) for x in xtl]
            xtickloc_min=[E(x) for x in xtl_min]
            # set the second axis ticks to the same locations
            ax2.set_xticks(xtickloc)
            ax2.set_xticks(xtickloc_min, minor=True)
            # calculate new values for the second axis tick labels, format them, and set them
            x2labels = ['{:.0f}'.format(WL(x)) for x in xtickloc]
            ax2.set_xticklabels(x2labels)
            # force the bounds to be the same
            ax2.set_xlim(ax1.get_xlim()) 
            ax2.set_xlabel('Wavelength (nm)') 
            if "maj_int" in kwargs:
               maj_int=float(kwargs["maj_int"])
            else:
                maj_int=0.05
            if "min_int" in kwargs:
               min_int=float(kwargs["min_int"])
            else:
                min_int=0.01
            ax1.xaxis.set_major_locator(plt.MultipleLocator(maj_int))
            ax1.xaxis.set_minor_locator(plt.MultipleLocator(min_int))
    if unit=="cm":
            lw=1
            lwl=632.8
            ax1.plot(1E7/lwl-1E7/filedata_dat[i].signals['Wavelength (nm)'],(data),"", label=label,color="blue",linewidth=lw)
            ax1.set_xlabel(r'Raman shift  ($\mathrm{cm}^-1$)')
            ax1.set_xlim((x1,x2)) 
            
    ax1.set_ylabel('Photon intensity (counts)')
    ax1.set_ylim((-10,1.1*np.max(data[nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)]))) 
    ax1.tick_params('y')
   # ax1.yaxis.set_ticks_position('both')
   # ax1.xaxis.set_ticks_position('both')

   ## ax1.yaxis.set_major_locator(plt.MultipleLocator(5))
   ## ax1.yaxis.set_minor_locator(plt.MultipleLocator(2.5))
    #plt.legend(loc="best")
    #ax1.grid(True,linestyle=':')
  #  ax1.set_ylim((-0.05, 5.5)) 
   ## ax1.set_ylim((0-np.amax(data[nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])/300, np.amax(data[nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])/10+np.amax(data[nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])/300)) 
    fig.set_size_inches(6,3)
    if save==True:
        plt.savefig(path+str(filedata_dat[i].header["Filename"])+"V"+str(filedata_dat[i].header["Bias (V)"])+str(x2)+"-"+str(x1)+"-"+label+".png", dpi=400, bbox_inches = 'tight',transparent=True)
        plt.savefig(path+str(filedata_dat[i].header["Filename"])+"V"+str(filedata_dat[i].header["Bias (V)"])+str(x2)+"-"+str(x1)+"-"+label+".svg", dpi=400, bbox_inches = 'tight',transparent=True)
#bgcorplot(1,2,500,800,"nm")

def bgcorplot_div_list(i_list,i_bg,i_div,i_bgdiv,x1,x2,unit,save,path,label,norm,**kwargs):  #converted  
# """
# example
#bgcorplot_div_list(["LS-zdep-4ML00001","LS-zdep-4ML00002","LS-zdep-4ML00003","LS-zdep-4ML00004","LS-zdep-4ML00005"],"LS-man00002","LS-man00002","LS-man00002",880,913,"eV",True,path,"div",False,bg=317,maj_int=0.01,min_int=0.002,maj_int2=10,min_int2=2,offset=700)
# """
    ind_list=[]
    data_ar=[]
    for index in range (0,len(i_list)):
        
        if isfloat(i_list[index])==True:
            i=i_list[index]
            ind_list.append(i)
        else:
            for j in range (0,len(file_numbers)):
                if filedata_dat[j].header["Filename"]==str(i_list[index]):
                    i=j
                    ind_list.append(i)
                    
                    break
            
   # if isfloat(i2)==True:
     #   i2=i2
   #else:
     #   for j in range (0,len(file_numbers)):
      #      if filedata_dat[j].header["Filename"]==str(i2):
      #          i2=j
      #          break
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bg):
                i_bg=j
                break

    if isfloat(i_div)==True:
        i_div=i_div
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_div):
                i_div=j
                break
    if isfloat(i_bgdiv)==True:
        i_bgdiv=i_bgdiv
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bgdiv):
                i_bgdiv=j
                break
            
    i=ind_list[0]
    lr=filedata_dat[i].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i].signals['Wavelength (nm)'][-1]    
    ncol=len(filedata_dat[i].signals['Wavelength (nm)'])-1
    n=float(filedata_dat[i].header["Number of Accumulations"])
   # n2=float(filedata_dat[i2].header["Number of Accumulations"])
    n_bg=float(filedata_dat[i_bg].header["Number of Accumulations"])
    if "bg" in kwargs:
        bgcounts=float(kwargs["bg"])
        bg=np.full(ncol+1,bgcounts)
    else:
        bg=savgol_filter(filedata_dat[i_bg].signals['Counts'],31,0)
        
    fig, ax1 = plt.subplots() #osy 

    norm_spec=savgol_filter(filedata_dat[i_div].signals['Counts'],31,0) - savgol_filter(filedata_dat[i_bgdiv].signals['Counts'],31,0)
    if norm==True:
        for i1 in ind_list:
            data=(filedata_dat[i1].signals['Counts']-n/n_bg*savgol_filter(despike(bg,0.1,7,nmtopix(700,lr,rr,ncol),nmtopix(940,lr,rr,ncol),ncol),31,0))/norm_spec
            data_ar.append(data)
    else:
       # data=(filedata_dat[i].signals['Counts nf']-n/n_bg*savgol_filter(filedata_dat[i_bg].signals['Counts'],31,0))
        if "nf" in kwargs:
            for i1 in ind_list:
                data=(filedata_dat[i1].signals['Counts nf 1']-n/n_bg*savgol_filter(bg,31,0))
                data_ar.append(data)
            if type(kwargs["nf"]) == int or type(kwargs["nf"]) == float:
                for i2 in range(0,len(data_ar)):
                    data_ar[i2]=despike(data_ar[i2],0.1,int(kwargs["nf"]),lr,rr,ncol)
        else:
            for i1 in ind_list:
                data=(despike_multi(i1,0.2,7,lr,rr,ncol)-n/n_bg*savgol_filter(bg,31,0))
               # data=(filedata_dat[i1].signals['Counts']-n/n_bg*savgol_filter(bg,31,0))# with a properly working filter in labview (after 2021-11-25)
                data_ar.append(data)

    if unit=="nm":
        ax1.plot(filedata_dat[i].signals['Wavelength (nm)'],(data),"", label=label,color="blue")
        #ax1.plot(filedata_dat[i].signals['Wavelength (nm)'],(norm_spec),"", label="plasmon",color="red")
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_xlim((x1, x2)) 
        ax2 = ax1.twiny()
        # get the primary axis x tick locations in plot units
        xtickloc = ax1.get_xticks() 
        # set the second axis ticks to the same locations
        ax2.set_xticks(xtickloc)
        # calculate new values for the second axis tick labels, format them, and set them
        x2labels = ['{:.3g}'.format(x) for x in E(xtickloc)]
        ax2.set_xticklabels(x2labels)
        # force the bounds to be the same
        ax2.set_xlim(ax1.get_xlim()) 
        ax2.set_xlabel('Energy (eV)')
    if "offset" in kwargs:
       offset=float(kwargs["offset"])
    else:
        offset=100
    if unit=="eV":
            lw=0.8
            d_ind=0
            
            if "avg" in kwargs:
                normc=np.max(savgol_filter(np.mean(data_ar,axis=0),15,0))
                print(normc)
                ax1.plot(eV/filedata_dat[i1].signals['Wavelength (nm)'],savgol_filter(np.mean(data_ar,axis=0),15,0)/normc,"", label=str(i1),linewidth=lw,color="blue")
                ax1.plot(eV/filedata_dat[i1].signals['Wavelength (nm)'],(np.mean(data_ar,axis=0))/normc,"", label=str(i1),linewidth=lw*0.3,color="blue")
                data_mean=np.mean(data_ar,axis=0)
                f= open(path+str(filedata_dat[i].header["Filename"])+"V"+str(filedata_dat[i].header["Bias (V)"])+"-"+label+".txt","w+")
                f.write("Photon energy (eV)"+'\t'+"Photon intensity ZnPc [a.u]" +"\n")
                for index in range(len(data_mean)-1,-1,-1):
                    f.write(str("{0:.5f}".format(eV/filedata_dat[i].signals['Wavelength (nm)'][index])) +"\t" +str("{0:.5f}".format((data_mean[index])))+"\n")
                f.close() 
            else:
                for i1 in ind_list:
                    print(i1," ",d_ind)
                    if d_ind==0:
                        col="blue"
                    else:
                        col="red"
                  #  print(despike_multi(i1,0.2,7,lr,rr,ncol,),"dimension despike")
                    #ax1.plot(eV/filedata_dat[i1].signals['Wavelength (nm)'],(data_ar[d_ind]+d_ind*offset),"", label=str(i1),linewidth=lw,color="blue")
                    #print(np.max(np.array(data_ar[d_ind][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])))
            #        ax1.plot(eV/filedata_dat[i1].signals['Wavelength (nm)'],(np.array(data_ar[d_ind])/np.max(np.array(data_ar[d_ind][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)]))+d_ind*offset),"", label=str(i1),linewidth=lw,color="blue")
                    ax1.plot(eV/filedata_dat[i1].signals['Wavelength (nm)'],(np.array(data_ar[d_ind])/np.max(np.array(data_ar[d_ind]))+d_ind*offset),"", label=str(i1),linewidth=lw,color="blue")
                    #ax1.plot(eV/filedata_dat[i1].signals['Wavelength (nm)'],(data_ar[d_ind]/max(data_ar[d_ind])),"", label=str(i1),linewidth=lw,alpha=0.5)
                    d_ind+=1
            if "fit" in kwargs:
               # ax1.plot(eV/filedata_dat[i].signals['Wavelength (nm)'],lorentz(eV/filedata_dat[i].signals['Wavelength (nm)'],621.341,1.39405,0.00083182,0),"", label=label,color="black",linewidth=lw) #H2Pc 00077 3ML
                #ax1.plot(eV/filedata_dat[i].signals['Wavelength (nm)'],lorentz(eV/filedata_dat[i].signals['Wavelength (nm)'],921.683,1.51759,0.000650313,0),"", label=label,color="black",linewidth=lw) #ZnPc idep00001 4ML
                ax1.plot(eV/filedata_dat[i].signals['Wavelength (nm)'],lorentz(eV/filedata_dat[i].signals['Wavelength (nm)'],196.362,1.52001,0.00120571,0),"", label=label,color="black",linewidth=lw)   #ZnPc 3ML LS00003
            ax1.set_xlabel('Photon energy (eV)')
           # f= open(path+str(filedata_dat[i].header["Filename"])+"V"+str(filedata_dat[i].header["Bias (V)"])+"-"+label+".txt","w+")
           # f.write("Photon energy (eV)"+'\t'+"Photon intensity ZnPc [a.u]" +"\n")#+'\t'+"Photon intensity plasmon [a.u]"+"\n")
           # for index in range(len(data)-1,-1,-1):
           #     f.write(str("{0:.5f}".format(eV/filedata_dat[i].signals['Wavelength (nm)'][index])) +"\t" +str("{0:.5f}".format(data[index]))+"\n")#+"\t" +str("{0:.5f}".format(norm_spec[index]*np.amax(data)*0.5/np.amax(norm_spec[nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])))  + "\n")
             #majtick_spacing=0.1
             #ax1.xaxis.set_major_locator(ticker.MultipleLocator(majtick_spacing))
             #mintick_spacing=0.05
           # f.close() 
            ax1.set_xlim((E(x2), E(x1)))
            ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax2 = ax1.twiny()
            # get the primary axis x tick locations in plot units
            if "maj_int2" in kwargs:
                maj_int2=float(kwargs["maj_int2"])
            else:
                maj_int2=100
            if "min_int2" in kwargs:
                min_int2=float(kwargs["min_int2"])
            else:
                min_int2=20
           # maj_int2=100
            #min_int2=20
            x1nm=math.ceil((x1)/maj_int2)*maj_int2
            x2nm=math.floor((x2)/maj_int2)*maj_int2
            x1nm_min=math.ceil(x1/min_int2)*min_int2
            x2nm_min=math.floor(x2/min_int2)*min_int2

            xtl=np.linspace(x1nm,x2nm,int(round(abs(x2nm-x1nm)/maj_int2))+1)

            xtl_min=np.linspace(x1nm_min,x2nm_min,int(round(abs(x2nm_min-x1nm_min)/min_int2))+1)

            xtickloc=[E(x) for x in xtl]
            xtickloc_min=[E(x) for x in xtl_min]
            # set the second axis ticks to the same locations
            ax2.set_xticks(xtickloc)
            ax2.set_xticks(xtickloc_min, minor=True)
            # calculate new values for the second axis tick labels, format them, and set them
            x2labels = ['{:.0f}'.format(WL(x)) for x in xtickloc]
            ax2.set_xticklabels(x2labels)
            # force the bounds to be the same
            ax2.set_xlim(ax1.get_xlim()) 
            ax2.set_xlabel('Wavelength (nm)') 

            ax1.set_ylabel('Normalized intensity (arb. u.)')
           # ax1.set_ylim((-0.1, 8.1)) 
            ax1.tick_params('y')
           # ax1.yaxis.set_ticks_position('both')
           # ax1.xaxis.set_ticks_position('both')
            if "maj_int" in kwargs:
               maj_int=float(kwargs["maj_int"])
            else:
                maj_int=0.05
            if "min_int" in kwargs:
               min_int=float(kwargs["min_int"])
            else:
                min_int=0.01
            ax1.xaxis.set_major_locator(plt.MultipleLocator(maj_int))
            ax1.xaxis.set_minor_locator(plt.MultipleLocator(min_int))
    if unit=="cm":
        d_ind=0
        for i1 in ind_list :
            lw=1
            lwl=632.8
         #   ax1.plot(1E7/lwl-1E7/filedata_dat[i].signals['Wavelength (nm)'],(data),"", label=label,color="blue",linewidth=lw)
         #   ax1.plot(1E7/lwl-1E7/filedata_dat[i].signals['Wavelength (nm)'],(np.array(data_ar[d_ind])/np.max(np.array(data_ar[d_ind]))+d_ind*offset),"", label=str(i1),linewidth=lw,color="blue") normalized
            ax1.plot(1E7/lwl-1E7/filedata_dat[i].signals['Wavelength (nm)'],(np.array(data_ar[d_ind])+d_ind*offset),"", label=str(i1),linewidth=lw,color="blue")
            d_ind+=1
        ax1.set_xlabel(r'Raman shift  ($\mathrm{cm}^-1$)')
        ax1.set_xlim((x1,x2)) 
            
    ax1.set_ylabel('Normalized intensity (arb. u.)')
  #  ax1.set_ylabel('Photon intensity (counts)')
   # ax1.set_ylim((-10,1.1*np.max(data[nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)]))) 
    ax1.tick_params('y')
   ## ax1.yaxis.set_major_locator(plt.MultipleLocator(5))
   ## ax1.yaxis.set_minor_locator(plt.MultipleLocator(2.5))
    #plt.legend(loc="best")
   # ax1.grid(True,linestyle=':')
   # ax1.set_ylim((-0.1, 1.2)) 
   ## ax1.set_ylim((0-np.amax(data[nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])/300, np.amax(data[nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])/10+np.amax(data[nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])/300)) 
  #  fig.set_size_inches(3,3)
    #fig.set_size_inches(4.5,6)
    fig.set_size_inches(4,3)
    if save==True:
        plt.savefig(path+str(filedata_dat[i].header["Filename"])+"V"+str(filedata_dat[i].header["Bias (V)"])+str(x2)+"-"+str(x1)+"-"+label+".png", dpi=600, bbox_inches = 'tight',transparent=True)
        plt.savefig(path+str(filedata_dat[i].header["Filename"])+"V"+str(filedata_dat[i].header["Bias (V)"])+str(x2)+"-"+str(x1)+"-"+label+".svg", dpi=400, bbox_inches = 'tight',transparent=True)
#bgcorplot(1,2,500,800,"nm")
    df=True
    if df==True:
        # Extract frequency shifts and bias voltages from headers
        freq_shifts = []
        bias_voltages = []
        for idx in ind_list:
            hdr = filedata_dat[idx].header
            freq_shift = float(hdr.get("OC M1 Freq. Shift (Hz)", "nan"))
            bias = float(hdr.get("Bias (V)", "nan"))
            if not (np.isnan(freq_shift) or np.isnan(bias)):
                freq_shifts.append(freq_shift)
                bias_voltages.append(bias)
    
        # Add a new inset axis or new figure
        ax_freq = fig.add_axes([0.6, 0.6, 0.3, 0.3])  # inset (x0, y0, width, height)
        ax_freq.plot(bias_voltages, freq_shifts, 'o-', color='green')
        ax_freq.set_xlabel("Bias (V)", fontsize=8)
        ax_freq.set_ylabel("Freq. Shift (Hz)", fontsize=8)
        ax_freq.tick_params(labelsize=8)
        ax_freq.grid(True, linestyle=':', linewidth=0.5)


def bgcorplot_stitch(i_start, i_stop, i_bg, i_div, i_bgdiv, x1, x2, unit, save, path, label, norm, gridpoints=10000,plot_original=False, **kwargs):  
    """ 
    Example:
    bgcorplot_stitch("LS-zdep-4ML00001", "LS-zdep-4ML00005", "LS-man00002", "LS-man00002", "LS-man00002",
                     880, 913, "eV", True, path, "div", False, bg=317, maj_int=0.01, min_int=0.002, maj_int2=10, min_int2=2, offset=700)
    """
    
    ind_list = []
    data_ar=[]
    
    # Find indices of i_start and i_stop
    i_start_index, i_stop_index = None, None
    
    for j in range(len(file_numbers)):
        if filedata_dat[j].header["Filename"] == str(i_start):
            i_start_index = j
        if filedata_dat[j].header["Filename"] == str(i_stop):
            i_stop_index = j
        if i_start_index is not None and i_stop_index is not None:
            break
    
    # Ensure valid indices
    if i_start_index is None or i_stop_index is None:
        raise ValueError("Start or Stop spectrum not found in filedata_dat")

    # Append indices from start to stop
    if i_start_index <= i_stop_index:
        ind_list = list(range(i_start_index, i_stop_index + 1))
    else:
        ind_list = list(range(i_start_index, i_stop_index - 1, -1))  # Handle reverse order if needed

    # Find indices for background and division spectra
    def find_index(spec):
        if isfloat(spec):
            return spec
        for j in range(len(file_numbers)):
            if filedata_dat[j].header["Filename"] == str(spec):
                return j
        raise ValueError(f"Filename {spec} not found in filedata_dat")

    i_bg = find_index(i_bg)
    i_div = find_index(i_div)
    i_bgdiv = find_index(i_bgdiv)
    
    # First spectrum in the range
    i=ind_list[0]
    lr=filedata_dat[i].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i].signals['Wavelength (nm)'][-1]    
    ncol=len(filedata_dat[i].signals['Wavelength (nm)'])-1
    n=float(filedata_dat[i].header["Number of Accumulations"])
   # n2=float(filedata_dat[i2].header["Number of Accumulations"])
    n_bg=float(filedata_dat[i_bg].header["Number of Accumulations"])
    if "bg" in kwargs:
        bgcounts=float(kwargs["bg"])
        bg=np.full(ncol+1,bgcounts)
    else:
        bg=savgol_filter(filedata_dat[i_bg].signals['Counts'],31,0)
        
    fig, ax1 = plt.subplots() #osy 

    norm_spec=savgol_filter(filedata_dat[i_div].signals['Counts'],31,0) - savgol_filter(filedata_dat[i_bgdiv].signals['Counts'],31,0)
    if norm==True:
        for i1 in ind_list:
            data=(filedata_dat[i1].signals['Counts']-n/n_bg*savgol_filter(despike(bg,0.1,7,nmtopix(700,lr,rr,ncol),nmtopix(940,lr,rr,ncol),ncol),31,0))/norm_spec
            data_ar.append(data)
    else:
       # data=(filedata_dat[i].signals['Counts nf']-n/n_bg*savgol_filter(filedata_dat[i_bg].signals['Counts'],31,0))
        if "nf" in kwargs:
            for i1 in ind_list:
                data=(filedata_dat[i1].signals['Counts nf']-n/n_bg*savgol_filter(bg,31,0))
                data_ar.append(data)
            if type(kwargs["nf"]) == int or type(kwargs["nf"]) == float:
                for i2 in range(0,len(data_ar)):
                    data_ar[i2]=despike(data_ar[i2],0.1,int(kwargs["nf"]),lr,rr,ncol)
        else:
            for i1 in ind_list:
                data=(despike_multi(i1,0.2,7,lr,rr,ncol)-n/n_bg*savgol_filter(bg,31,0))
               # data=(filedata_dat[i1].signals['Counts']-n/n_bg*savgol_filter(bg,31,0))# with a properly working filter in labview (after 2021-11-25)
                data_ar.append(data)

    if unit=="nm":
        ax1.plot(filedata_dat[i].signals['Wavelength (nm)'],(data),"", label=label,color="blue")
        #ax1.plot(filedata_dat[i].signals['Wavelength (nm)'],(norm_spec),"", label="plasmon",color="red")
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_xlim((x1, x2)) 
        ax2 = ax1.twiny()
        # get the primary axis x tick locations in plot units
        xtickloc = ax1.get_xticks() 
        # set the second axis ticks to the same locations
        ax2.set_xticks(xtickloc)
        # calculate new values for the second axis tick labels, format them, and set them
        x2labels = ['{:.3g}'.format(x) for x in E(xtickloc)]
        ax2.set_xticklabels(x2labels)
        # force the bounds to be the same
        ax2.set_xlim(ax1.get_xlim()) 
        ax2.set_xlabel('Energy (eV)')
    if "offset" in kwargs:
       offset=float(kwargs["offset"])
    else:
        offset=100
    if unit == "eV":
            lw = 0.8
            all_wavelengths = []
            all_intensities = []
        
            # Load data
            for i1 in ind_list:
                wavelength = filedata_dat[i1].signals['Wavelength (nm)']
                intensity = data_ar[ind_list.index(i1)]  # Corresponding intensity data
                all_wavelengths.append(wavelength)
                all_intensities.append(intensity)
        
            # Plot all original spectra for sanity check (if enabled)
            if plot_original:
                for i, (wavelength, intensity) in enumerate(zip(all_wavelengths, all_intensities)):
                    ax1.plot(eV / wavelength, intensity, label=f"Spectrum {i}", alpha=0.5)
        
            # Define a full common wavelength grid
            min_wavelength = min(w[0] for w in all_wavelengths)  # Smallest starting wavelength
            max_wavelength = max(w[-1] for w in all_wavelengths)  # Largest ending wavelength
            full_grid = np.linspace(min_wavelength, max_wavelength, gridpoints)  # Adjust grid density
        
            # Interpolate each spectrum onto the full grid
            interpolated_spectra = []
            for wavelength, intensity in zip(all_wavelengths, all_intensities):
                interp_func = interp1d(wavelength, intensity, bounds_error=False, fill_value=0)  # Extrapolation handled
                interpolated_spectra.append(interp_func(full_grid))
        
            # Take maximum intensity at each point
            stitched_spectrum = np.nanmax(interpolated_spectra, axis=0)
            
            col="blue"
            
            f= open(path+str(filedata_dat[i_start_index].header["Filename"])+"V"+str(filedata_dat[i_start_index].header["Bias (V)"])+"-"+label+".txt","w+")
            f.write("Photon energy (eV)"+'\t'+"Photon intensity (Counts)" +"\n")#+'\t'+"Photon intensity plasmon [a.u]"+"\n")
            for index in range(len(stitched_spectrum)-1,-1,-1):
                f.write(str("{0:.5f}".format(eV/full_grid[index])) +"\t" +str("{0:.5f}".format(stitched_spectrum[index]))+"\n")
            f.close()
            
            ax1.plot(eV/full_grid,stitched_spectrum,"", label=str(i1),linewidth=lw,color="blue")
            
            
            if "fit" in kwargs:
                ax1.plot(eV/filedata_dat[i].signals['Wavelength (nm)'],lorentz(eV/filedata_dat[i].signals['Wavelength (nm)'],196.362,1.52001,0.00120571,0),"", label=label,color="black",linewidth=lw)   #ZnPc 3ML LS00003
            ax1.set_xlabel('Photon energy (eV)')

            ax1.set_xlim((E(x2), E(x1)))
            ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax2 = ax1.twiny()
            # get the primary axis x tick locations in plot units
            if "maj_int2" in kwargs:
                maj_int2=float(kwargs["maj_int2"])
            else:
                maj_int2=100
            if "min_int2" in kwargs:
                min_int2=float(kwargs["min_int2"])
            else:
                min_int2=20
           # maj_int2=100
            #min_int2=20
            x1nm=math.ceil((x1)/maj_int2)*maj_int2
            x2nm=math.floor((x2)/maj_int2)*maj_int2
            x1nm_min=math.ceil(x1/min_int2)*min_int2
            x2nm_min=math.floor(x2/min_int2)*min_int2

            xtl=np.linspace(x1nm,x2nm,int(round(abs(x2nm-x1nm)/maj_int2))+1)

            xtl_min=np.linspace(x1nm_min,x2nm_min,int(round(abs(x2nm_min-x1nm_min)/min_int2))+1)

            xtickloc=[E(x) for x in xtl]
            xtickloc_min=[E(x) for x in xtl_min]
            # set the second axis ticks to the same locations
            ax2.set_xticks(xtickloc)
            ax2.set_xticks(xtickloc_min, minor=True)
            # calculate new values for the second axis tick labels, format them, and set them
            x2labels = ['{:.0f}'.format(WL(x)) for x in xtickloc]
            ax2.set_xticklabels(x2labels)
            # force the bounds to be the same
            ax2.set_xlim(ax1.get_xlim()) 
            ax2.set_xlabel('Wavelength (nm)') 

            ax1.set_ylabel('Normalized intensity (arb. u.)')
           # ax1.set_ylim((-0.1, 8.1)) 
            ax1.tick_params('y')
           # ax1.yaxis.set_ticks_position('both')
           # ax1.xaxis.set_ticks_position('both')
            if "maj_int" in kwargs:
               maj_int=float(kwargs["maj_int"])
            else:
                maj_int=0.05
            if "min_int" in kwargs:
               min_int=float(kwargs["min_int"])
            else:
                min_int=0.01
            ax1.xaxis.set_major_locator(plt.MultipleLocator(maj_int))
            ax1.xaxis.set_minor_locator(plt.MultipleLocator(min_int))
    if unit=="cm":
        if "laser_nm" in kwargs:
            lwl=float(kwargs["laser_nm"])
        else:
            lwl=632.8
            
        if "offset_nm" in kwargs:
            offset_nm=float(kwargs["offset_nm"])
        else:
            offset_nm=0
            
        lw=1
        all_wavelengths = []
        all_intensities = []
        for i1 in ind_list:
            wavelength = filedata_dat[i1].signals['Wavelength (nm)']
            intensity = data_ar[ind_list.index(i1)]  # Corresponding intensity data
            all_wavelengths.append(wavelength)
            all_intensities.append(intensity)
        
        # Plot all original spectra for sanity check (if enabled)
        if plot_original:
            for i, (wavelength, intensity) in enumerate(zip(all_wavelengths, all_intensities)):
                ax1.plot(eV / wavelength, intensity, label=f"Spectrum {i}", alpha=0.5)
        
        # Define a full common wavelength grid
        min_wavelength = min(w[0] for w in all_wavelengths)  # Smallest starting wavelength
        max_wavelength = max(w[-1] for w in all_wavelengths)  # Largest ending wavelength
        full_grid = np.linspace(min_wavelength, max_wavelength, gridpoints)  # Adjust grid density
        
        # Interpolate each spectrum onto the full grid
        interpolated_spectra = []
        for wavelength, intensity in zip(all_wavelengths, all_intensities):
            interp_func = interp1d(wavelength, intensity, bounds_error=False, fill_value=0)  # Extrapolation handled
            interpolated_spectra.append(interp_func(full_grid))
        
        # Take maximum intensity at each point
        stitched_spectrum = np.nanmax(interpolated_spectra, axis=0)
        
        col="blue"
        
        
        cm_scale=(1E7/lwl-1E7/(full_grid-offset_nm))
        ax1.plot(cm_scale,stitched_spectrum,"", label=str(i1),linewidth=lw,color="blue")
        
        # Define the mapping functions for the top axis
        def cm1_to_ev(x):
            λ = 1e7 / (1e7 / lwl - x)  # convert cm⁻¹ to nm
            return 1240 / λ            # convert nm to eV
        
        def ev_to_cm1(e):
            λ = 1240 / e               # convert eV to nm
            return 1e7 / lwl - 1e7 / λ # convert nm to cm⁻¹
        
        # Add top axis
        if "sec_axis" in kwargs:
            secax = ax1.secondary_xaxis('top', functions=(cm1_to_ev, ev_to_cm1))
            secax.set_xlabel("Energy (eV)")
        
        ax1.set_xlabel(r'Raman shift  ($\mathrm{cm}^{-1}$)')
        ax1.set_xlim((x1,x2)) 
        ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
                 
        ax1.set_ylabel('Intensity (counts)')
         
        ax1.tick_params('y')

    fig.set_size_inches(6,3)
    if save==True:
        plt.savefig(path+str(filedata_dat[i].header["Filename"])+"V"+str(filedata_dat[i].header["Bias (V)"])+str(x2)+"-"+str(x1)+"-"+label+".png", dpi=600, bbox_inches = 'tight',transparent=True)
        plt.savefig(path+str(filedata_dat[i].header["Filename"])+"V"+str(filedata_dat[i].header["Bias (V)"])+str(x2)+"-"+str(x1)+"-"+label+".svg", dpi=400, bbox_inches = 'tight',transparent=True)
#bgcorplot(1,2,500,800,"nm")

def bgcorplot_stitch_2(i_start, i_stop, i_bg_start, i_bg_stop,
                     x1, x2, unit, save, path, label, norm,bg_sub=True,
                     gridpoints=10000, plot_original=False, despike_w=5, **kwargs):
    """ 
    Example:
    bgcorplot_stitch("LS-zdep-4ML00001", "LS-zdep-4ML00005",
                     "LS-bg00001", "LS-bg00005",
                     880, 913, "eV", True, path, "div", False,
                     bg=317, maj_int=0.01, min_int=0.002, maj_int2=10, min_int2=2, offset=700)
    """

    ind_list = []
    data_ar = []
    bg_data_ar = []
    cor_data_ar=[]

    # --- Find indices of i_start and i_stop ---
    i_start_index, i_stop_index = None, None
    for j in range(len(file_numbers)):
        if filedata_dat[j].header["Filename"] == str(i_start):
            i_start_index = j
        if filedata_dat[j].header["Filename"] == str(i_stop):
            i_stop_index = j
        if i_start_index is not None and i_stop_index is not None:
            break

    if i_start_index is None or i_stop_index is None:
        raise ValueError("Start or Stop spectrum not found in filedata_dat")

    if i_start_index <= i_stop_index:
        ind_list = list(range(i_start_index, i_stop_index + 1))
    else:
        ind_list = list(range(i_start_index, i_stop_index - 1, -1))

    # --- Find indices for background start/stop ---
    i_bg_start_index, i_bg_stop_index = None, None
    for j in range(len(file_numbers)):
        if filedata_dat[j].header["Filename"] == str(i_bg_start):
            i_bg_start_index = j
        if filedata_dat[j].header["Filename"] == str(i_bg_stop):
            i_bg_stop_index = j
        if i_bg_start_index is not None and i_bg_stop_index is not None:
            break

    if i_bg_start_index is None or i_bg_stop_index is None:
        raise ValueError("Background start or stop spectrum not found in filedata_dat")

    if i_bg_start_index <= i_bg_stop_index:
        bg_ind_list = list(range(i_bg_start_index, i_bg_stop_index + 1))
    else:
        bg_ind_list = list(range(i_bg_start_index, i_bg_stop_index - 1, -1))

    # --- Common setup from the first spectrum ---
    i = ind_list[0]
    lr = filedata_dat[i].signals['Wavelength (nm)'][0]
    rr = filedata_dat[i].signals['Wavelength (nm)'][-1]
    ncol = len(filedata_dat[i].signals['Wavelength (nm)']) - 1
    n = float(filedata_dat[i].header["Number of Accumulations"])

    fig, ax1 = plt.subplots()

    # --- Load main spectra ---
    for i1, i2 in zip(ind_list, bg_ind_list):
        counts = filedata_dat[i1].signals['Counts nf']
        bg_counts = filedata_dat[i2].signals['Counts nf']
        #counts = filedata_dat[i1].signals['Counts']
        #bg_counts = filedata_dat[i2].signals['Counts']
        data_ar.append(counts)
        bg_data_ar.append(bg_counts)
        if despike_w is not None:
            cor_data,_ = despike_sigma(counts-bg_counts+500, width=despike_w, sigma_thresh=6, mode='both')
            cor_data-=500
        else:
            cor_data=counts-bg_counts
        cor_data_ar.append(cor_data)


    # --- Stitch main spectra ---
    all_wavelengths = [filedata_dat[i1].signals['Wavelength (nm)'] for i1 in ind_list]
    if bg_sub==True:  
        all_intensities = cor_data_ar
    else:
        all_intensities=np.array(data_ar)-305
    min_wavelength = min(w[0] for w in all_wavelengths)
    max_wavelength = max(w[-1] for w in all_wavelengths)
    full_grid = np.linspace(min_wavelength, max_wavelength, gridpoints)

    interpolated_spectra = []
    for wavelength, intensity in zip(all_wavelengths, all_intensities):
        interp_func = interp1d(wavelength, intensity, bounds_error=False, fill_value=0)
        interpolated_spectra.append(interp_func(full_grid))
    stitched_spectrum = np.nanmax(interpolated_spectra, axis=0)
    stitched_corrected=stitched_spectrum
    # --- Plot and save as before ---
    if unit == "eV":
        lw = 0.8
        ax1.plot(eV / full_grid, stitched_corrected, "", label=label, linewidth=lw, color="blue")

        f = open(path + str(filedata_dat[i_start_index].header["Filename"]) +
                 "V" + str(filedata_dat[i_start_index].header["Bias (V)"]) +
                 "-" + label + ".txt", "w+")
        f.write("Photon energy (eV)\tPhoton intensity (Counts)\n")
        for index in range(len(stitched_corrected)-1, -1, -1):
            f.write(f"{eV/full_grid[index]:.5f}\t{stitched_corrected[index]:.5f}\n")
        f.close()

        ax1.set_xlabel('Photon energy (eV)')
        ax1.set_ylabel('Intensity (arb. u.)')
        ax1.set_xlim((E(x2), E(x1)))
    
    if unit=="cm":
        if "laser_nm" in kwargs:
            lwl=float(kwargs["laser_nm"])
        else:
            lwl=632.8
            
        if "offset_nm" in kwargs:
            offset_nm=float(kwargs["offset_nm"])
        else:
            offset_nm=0
            
        lw=1
        
        col="blue"
        
        cm_scale=(1E7/lwl-1E7/(full_grid-offset_nm))
        ax1.plot(cm_scale,stitched_spectrum,"", label=str(i1),linewidth=lw,color="blue")
        
        # Define the mapping functions for the top axis
        def cm1_to_ev(x):
            λ = 1e7 / (1e7 / lwl - x)  # convert cm⁻¹ to nm
            return 1240 / λ            # convert nm to eV
        
        def ev_to_cm1(e):
            λ = 1240 / e               # convert eV to nm
            return 1e7 / lwl - 1e7 / λ # convert nm to cm⁻¹
        
        # Add top axis
        if "sec_axis" in kwargs:
            secax = ax1.secondary_xaxis('top', functions=(cm1_to_ev, ev_to_cm1))
            secax.set_xlabel("Energy (eV)")
        
        ax1.set_xlabel(r'Raman shift  ($\mathrm{cm}^{-1}$)')
        ax1.set_xlim((x1,x2)) 
        ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
                 
        ax1.set_ylabel('Intensity (counts)')
         
        ax1.tick_params('y')
        
    fig.set_size_inches(6, 3)
    if save:
        plt.savefig(path + str(filedata_dat[i].header["Filename"]) +
                    "V" + str(filedata_dat[i].header["Bias (V)"]) +
                    str(x2) + "-" + str(x1) + "-" + label + ".png",
                    dpi=600, bbox_inches='tight', transparent=True)
        plt.savefig(path + str(filedata_dat[i].header["Filename"]) +
                    "V" + str(filedata_dat[i].header["Bias (V)"]) +
                    str(x2) + "-" + str(x1) + "-" + label + ".svg",
                    dpi=400, bbox_inches='tight', transparent=True)


def stitch_spectra(file_list):
    """
    Reads spectra from a list of files, stitches them together by interpolating onto a common wavelength grid,
    and plots the final stitched spectrum.

    Parameters:
        file_list (list of str): List of file paths containing spectral data.

    Assumes each file contains two columns: wavelength (nm) and intensity.
    """
    all_wavelengths = []
    all_intensities = []

    # Load data from each file
    for file in file_list:
        data = np.loadtxt(file)
        wavelength, intensity = data[:, 0], data[:, 1]
        all_wavelengths.append(wavelength)
        all_intensities.append(intensity)

    # Define a common wavelength grid based on min/max range
    min_wavelength = max(w[0] for w in all_wavelengths)  # Highest starting wavelength
    max_wavelength = min(w[-1] for w in all_wavelengths)  # Lowest ending wavelength
    common_grid = np.linspace(min_wavelength, max_wavelength, 1000)  # Adjust grid density as needed

    # Interpolate each spectrum onto the common grid
    interpolated_spectra = []
    for wavelength, intensity in zip(all_wavelengths, all_intensities):
        interp_func = interp1d(wavelength, intensity, bounds_error=False, fill_value=0)
        interpolated_spectra.append(interp_func(common_grid))

    # Combine spectra (average where they overlap)
    stitched_spectrum = np.nanmean(interpolated_spectra, axis=0)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(common_grid, stitched_spectrum, label="Stitched Spectrum", color='b')
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.title("Stitched Spectrum")
    plt.legend()
    plt.show()

    return common_grid, stitched_spectrum  # Returns the final stitched spectrum data


def bgcorplot_div2(i,i_bg,i_div,i_bgdiv,x1,x2,unit,save,path,label,norm):  #converted  
    if isfloat(i)==True:
        i=i
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i):
                i=j
                break
            
   # if isfloat(i2)==True:
     #   i2=i2
   #else:
     #   for j in range (0,len(file_numbers)):
      #      if filedata_dat[j].header["Filename"]==str(i2):
      #          i2=j
      #          break
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bg):
                i_bg=j
                break

    if isfloat(i_div)==True:
        i_div=i_div
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_div):
                i_div=j
                break
    if isfloat(i_bgdiv)==True:
        i_bgdiv=i_bgdiv
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bgdiv):
                i_bgdiv=j
                break
            

    lr=filedata_dat[i].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i].signals['Wavelength (nm)'][-1]    
    ncol=len(filedata_dat[i].signals['Wavelength (nm)'])-1
    n=float(filedata_dat[i].header["Number of Accumulations"])
   # n2=float(filedata_dat[i2].header["Number of Accumulations"])
    n_bg=float(filedata_dat[i_bg].header["Number of Accumulations"])
    fig, ax1 = plt.subplots() #osy 

    norm_spec=savgol_filter(filedata_dat[i_div].signals['Counts'],31,0) - savgol_filter(filedata_dat[i_bgdiv].signals['Counts'],31,0)
    if norm==True:
       # data=(filedata_dat[i].signals['Counts nf']-n/n_bg*savgol_filter(filedata_dat[i_bg].signals['Counts'],31,0))/norm_spec
        data=(filedata_dat[i].signals['Counts']-n/n_bg*savgol_filter(despike(filedata_dat[i_bg].signals['Counts'],0.1,7,nmtopix(700,lr,rr,ncol),nmtopix(940,lr,rr,ncol)),31,0))/norm_spec
        data2=(filedata_dat[i+1].signals['Counts']-n/n_bg*savgol_filter(filedata_dat[i_bg].signals['Counts'],31,0))/norm_spec
        #data=despike(data,0.1,7,nmtopix(700,lr,rr,ncol),nmtopix(940,lr,rr,ncol))    
    else:
       # data=(filedata_dat[i].signals['Counts nf']-n/n_bg*savgol_filter(filedata_dat[i_bg].signals['Counts'],31,0))
        data=(filedata_dat[i].signals['Counts']-n/n_bg*savgol_filter(filedata_dat[i_bg].signals['Counts'],31,0))
        data2=(filedata_dat[i_div].signals['Counts']-n/n_bg*savgol_filter(filedata_dat[i_bg].signals['Counts'],31,0))
    if unit=="nm":
        ax1.plot(filedata_dat[i].signals['Wavelength (nm)'],(data),"", label=label,color="blue")
        #ax1.plot(filedata_dat[i].signals['Wavelength (nm)'],(norm_spec),"", label="plasmon",color="red")
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_xlim((x1, x2)) 
        ax2 = ax1.twiny()
        # get the primary axis x tick locations in plot units
        xtickloc = ax1.get_xticks() 
        # set the second axis ticks to the same locations
        ax2.set_xticks(xtickloc)
        # calculate new values for the second axis tick labels, format them, and set them
        x2labels = ['{:.3g}'.format(x) for x in E(xtickloc)]
        ax2.set_xticklabels(x2labels)
        # force the bounds to be the same
        ax2.set_xlim(ax1.get_xlim()) 
        ax2.set_xlabel('Energy (eV)')
    if unit=="eV":
          lw=1
          ax1.plot(eV/filedata_dat[i].signals['Wavelength (nm)'],(data),"", label=label,color="blue",linewidth=lw)
          ax1.plot(eV/filedata_dat[i_div].signals['Wavelength (nm)'],(data2)/2,"", label=label,color="red",linewidth=lw)
          ##ax1.plot(eV/filedata_dat[i+1].signals['Wavelength (nm)'],(data2)/10,"", label=label,color="red",linewidth=lw)
          #ax1.plot(eV/filedata_dat[i].signals['Wavelength (nm)'],(data),"",color="red",label=label)
         ## ax1.plot(eV/filedata_dat[i].signals['Wavelength (nm)'],norm_spec*np.amax(data[nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])*0.5/np.amax(norm_spec[nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])/10,"",color="black",ls="--",label="plasmon",linewidth=lw)
          #ax1.plot(filedata_dat[i].signals['Wavelength (nm)'],(norm_spec),"", label="plasmon",color="red")
          ax1.set_xlabel('Photon energy (eV)')
          ax1.set_xlim((eV/x2, eV/x1)) 
          f= open(pathcopy+str(filedata_dat[i].header["Filename"])+"V"+str(filedata_dat[i].header["Bias (V)"])+"-"+label+".txt","w+")
          f.write("Photon energy (eV)"+'\t'+"Photon intensity ZnPc [a.u]" +'\t'+"Photon intensity plasmon [a.u]"+"\n")
         #3##3 for index in range(len(data)-1,-1,-1):
         ##     f.write(str("{0:.5f}".format(eV/filedata_dat[i].signals['Wavelength (nm)'][index])) +"\t" +str("{0:.5f}".format(data[index])) +"\t" +str("{0:.5f}".format(norm_spec[index]*np.amax(data)*0.5/np.amax(norm_spec[nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])))  + "\n")
          #majtick_spacing=0.1
          #ax1.xaxis.set_major_locator(ticker.MultipleLocator(majtick_spacing))
          #mintick_spacing=0.05
          f.close() 
         ## x1int=np.sum(data[nmtopix(645,lr,rr,ncol):nmtopix(660,lr,rr,ncol)])
        # # x2int=np.sum(data[nmtopix(815,lr,rr,ncol):nmtopix(830,lr,rr,ncol)])
        #  ratio=x2int/x1int
#          print(x1int,"x1 intensity")
       #   print(x2int,"x2 intensity")
      #    print(ratio,"x2/x1 ratio intensity")
          
          #ax1.xaxis.set_minor_locator(ticker.MultipleLocator(mintick_spacing))
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Photon intensity (conuts)')
    #ax1.set_ylim((0, 700)) 
    ax1.tick_params('y')
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    ax1.xaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(0.01))
   ## ax1.yaxis.set_major_locator(plt.MultipleLocator(5))
   ## ax1.yaxis.set_minor_locator(plt.MultipleLocator(2.5))
    #plt.legend(loc="best")
    #ax1.grid(True,linestyle=':')
    #ax1.set_ylim((0, 1)) 
   ## ax1.set_ylim((0-np.amax(data[nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])/300, np.amax(data[nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])/10+np.amax(data[nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])/300)) 
    fig.set_size_inches(4,4)
    if save==True:
        plt.savefig(path+str(filedata_dat[i].header["Filename"])+"V"+str(filedata_dat[i].header["Bias (V)"])+str(x2)+"-"+str(x1)+"-"+label+".png", dpi=400, bbox_inches = 'tight',transparent=True)
        plt.savefig(pathcopy+str(filedata_dat[i].header["Filename"])+"V"+str(filedata_dat[i].header["Bias (V)"])+str(x2)+"-"+str(x1)+"-"+label+".svg", dpi=400, bbox_inches = 'tight',transparent=True)
#bgcorplot(1,2,500,800,"nm")

def bgcordiv_txt(i,i_bg,i_div,i_bgdiv,x1,x2,unit,save,path,label,norm,**kwargs):  #txt in energy scale  
#(j,"LS-CHmap-d00002",0,0,807,827,"eV",True,path,"div",False)  
    if isfloat(i)==True:
        i=i
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i):
                i=j
                break
            
   # if isfloat(i2)==True:
     #   i2=i2
   #else:
     #   for j in range (0,len(file_numbers)):
      #      if filedata_dat[j].header["Filename"]==str(i2):
      #          i2=j
      #          break
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bg):
                i_bg=j
                break

    if isfloat(i_div)==True:
        i_div=i_div
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_div):
                i_div=j
                break
    if isfloat(i_bgdiv)==True:
        i_bgdiv=i_bgdiv
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bgdiv):
                i_bgdiv=j
                break
            

    lr=filedata_dat[i].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i].signals['Wavelength (nm)'][-1]    
    ncol=len(filedata_dat[i].signals['Wavelength (nm)'])-1
    n=float(filedata_dat[i].header["Number of Accumulations"])
   # n2=float(filedata_dat[i2].header["Number of Accumulations"])
    n_bg=float(filedata_dat[i_bg].header["Number of Accumulations"])
    if "bg" in kwargs:
        bgcounts=float(kwargs["bg"])
        bg=np.full(ncol+1,bgcounts)
    else:
        bg=savgol_filter(filedata_dat[i_bg].signals['Counts'],31,0)

    norm_spec=savgol_filter(filedata_dat[i_div].signals['Counts'],31,0) - np.min(filedata_dat[i_div].signals['Counts'])#savgol_filter(filedata_dat[i_bgdiv].signals['Counts'],31,0)
    if norm==True:
        data=(filedata_dat[i].signals['Counts']-n/n_bg*savgol_filter(despike(bg,0.1,7,nmtopix(700,lr,rr,ncol),nmtopix(940,lr,rr,ncol),ncol),31,0))/norm_spec
    else:
        if "nf" in kwargs:
            data=(filedata_dat[i].signals['Counts nf'])
            if type(kwargs["nf"]) == int or type(kwargs["nf"]) == float:
                data=despike(data,0.1,int(kwargs["nf"]),380,400,ncol)
            data=data-n/n_bg*savgol_filter(bg,31,0)
        else:
            data=(despike_multi(i,0.2,7,lr,rr,ncol)-n/n_bg*savgol_filter(bg,31,0))
         #   print(n/n_bg*savgol_filter(bg,31,0))
            data2=(filedata_dat[i].signals['Counts']-n/n_bg*savgol_filter(bg,31,0))
    if unit=="eV":
            f= open(path+str(filedata_dat[i].header["Filename"])+"V"+str(filedata_dat[i].header["Bias (V)"])+"-"+label+".txt","w+")
            f.write("Photon energy (eV)"+'\t'+"Photon intensity ZnPc (a.u)" +"\n")#+'\t'+"Photon intensity plasmon [a.u]"+"\n")
            for index in range(len(data)-1,-1,-1):
                f.write(str("{0:.5f}".format(eV/filedata_dat[i].signals['Wavelength (nm)'][index])) +"\t" +str("{0:.5f}".format(data2[index]))+"\n")#+"\t" +str("{0:.5f}".format(norm_spec[index]*np.amax(data)*0.5/np.amax(norm_spec[nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])))  + "\n")
            f.close() 
          
#bgcorplot(1,2,500,800,"nm")
def bgcorplot_div_zero(i,i_bg,i_div,i_bgdiv,x1,x2,unit,save,path,label,norm,zero1,zero2):  #converted  
    if isfloat(i)==True:
        i=i
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i):
                i=j
                break
            
   # if isfloat(i2)==True:
     #   i2=i2
   #else:
     #   for j in range (0,len(file_numbers)):
      #      if filedata_dat[j].header["Filename"]==str(i2):
      #          i2=j
      #          break
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bg):
                i_bg=j
                break

    if isfloat(i_div)==True:
        i_div=i_div
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_div):
                i_div=j
                break
    if isfloat(i_bgdiv)==True:
        i_bgdiv=i_bgdiv
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bgdiv):
                i_bgdiv=j
                break
            

    lr=filedata_dat[i].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i].signals['Wavelength (nm)'][-1]    
    ncol=len(filedata_dat[i].signals['Wavelength (nm)'])-1
    n=float(filedata_dat[i].header["Number of Accumulations"])
   # n2=float(filedata_dat[i2].header["Number of Accumulations"])
    n_bg=float(filedata_dat[i_bg].header["Number of Accumulations"])
    fig, ax1 = plt.subplots() #osy 

    norm_spec=savgol_filter(filedata_dat[i_div].signals['Counts'],31,0) - savgol_filter(filedata_dat[i_bgdiv].signals['Counts'],31,0)
    if norm==True:
        data=(filedata_dat[i].signals['Counts nf']-n/n_bg*savgol_filter(filedata_dat[i_bg].signals['Counts'],31,0))/norm_spec
        data=despike(data,0.3,7,eVtopixOK(1.0,lr,rr,ncol),eVtopixOK(zero2-0.1,lr,rr,ncol),ncol) 
        data=despike(data,0.3,7,eVtopixOK(zero2+0.1,lr,rr,ncol),eVtopixOK(zero1-0.1,lr,rr,ncol),ncol)
        data=despike(data,0.3,7,eVtopixOK(zero1+0.1,lr,rr,ncol),eVtopixOK(3,lr,rr,ncol),ncol)
        #data=despike(data,0.1,7,nmtopix(700,lr,rr,ncol),nmtopix(940,lr,rr,ncol))    
    else:
        data=(filedata_dat[i].signals['Counts nf']-n/n_bg*savgol_filter(filedata_dat[i_bg].signals['Counts'],31,0))
    if unit=="nm":
        ax1.plot(filedata_dat[i].signals['Wavelength (nm)'],(data),"", label=label,color="blue")
        #ax1.plot(filedata_dat[i].signals['Wavelength (nm)'],(norm_spec),"", label="plasmon",color="red")
        ax1.set_xlabel('wavelength [nm]')
        ax1.set_xlim((x1, x2)) 
        ax2 = ax1.twiny()
        # get the primary axis x tick locations in plot units
        xtickloc = ax1.get_xticks() 
        # set the second axis ticks to the same locations
        ax2.set_xticks(xtickloc)
        # calculate new values for the second axis tick labels, format them, and set them
        x2labels = ['{:.3g}'.format(x) for x in E(xtickloc)]
        ax2.set_xticklabels(x2labels)
        # force the bounds to be the same
        ax2.set_xlim(ax1.get_xlim()) 
        ax2.set_xlabel('Energy [eV]')
    if unit=="eV":
          ax1.plot((eV/filedata_dat[i].signals['Wavelength (nm)'])-zero1,(data/np.amax(data[eVtopixOK(zero1+x2,lr,rr,ncol):eVtopixOK(zero1+x1,lr,rr,ncol)])),"", label=label,color="blue")
          ax1.plot((eV/filedata_dat[i].signals['Wavelength (nm)'])-zero2,(data/np.amax(data[eVtopixOK(zero2+x2,lr,rr,ncol):eVtopixOK(zero2+x1,lr,rr,ncol)])),"", label=label,color="red")
          #ax1.plot(eV/filedata_dat[i].signals['Wavelength (nm)'],(data),"",color="red",label=label)
          #ax1.plot(eV/filedata_dat[i].signals['Wavelength (nm)'],norm_spec*np.amax(data)*0.5/np.amax(norm_spec[nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)]),"",color="black",ls="--",label="plasmon")
          #ax1.plot(filedata_dat[i].signals['Wavelength (nm)'],(norm_spec),"", label="plasmon",color="red")
         # print(nmtopix(eV/(zero1+x1),lr,rr,ncol),nmtopix(eV/(zero1+x2),lr,rr,ncol))
         # print(eVtopixOK(zero1+x1,lr,rr,ncol),eVtopixOK(zero1+x2,lr,rr,ncol))
         # print(nmtopix(eV/(zero2+x1),lr,rr,ncol),nmtopix(eV/(zero2+x2),lr,rr,ncol))
         # print(eVtopixOK(zero2+x1,lr,rr,ncol),eVtopixOK(zero2+x2,lr,rr,ncol))
         # print(np.amax(data[eVtopixOK(zero1+x1,lr,rr,ncol):eVtopixOK(zero1+x2,lr,rr,ncol)]))
          ax1.set_xlabel('Photon energy [eV]')
          ax1.set_xlim((x1, x2)) 
          ax1.set_ylim(0,0.1)
          #f= open(path+str(filedata_dat[i].header["Filename"])+"V"+str(filedata_dat[i].header["Bias (V)"])+"-"+label+".txt","w+")
          #f.write("Photon energy [eV]"+'\t'+"Photon intensity ZnPc [a.u]" +'\t'+"Photon intensity plasmon [a.u]"+"\n")
          #for index in range(len(data)-1,-1,-1):
            #  f.write(str("{0:.5f}".format(eV/filedata_dat[i].signals['Wavelength (nm)'][index])) +"\t" +str("{0:.5f}".format(data[index])) +"\t" +str("{0:.5f}".format(norm_spec[index]*np.amax(data)*0.5/np.amax(norm_spec[nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])))  + "\n")
          #majtick_spacing=0.1
          #ax1.xaxis.set_major_locator(ticker.MultipleLocator(majtick_spacing))
          #mintick_spacing=0.05
          #f.close() 
          #ax1.xaxis.set_minor_locator(ticker.MultipleLocator(mintick_spacing))
    if unit=="wn":
          #ax1.plot(((eV/filedata_dat[i].signals['Wavelength (nm)'][::-1])-zero1)*(-8065.54),(data[::-1]/np.amax(data[eVtopixOK(zero1+x2,lr,rr,ncol):eVtopixOK(zero1+x1,lr,rr,ncol)])),"", label=label,color="blue")
          #ax1.plot(((eV/filedata_dat[i].signals['Wavelength (nm)'][::-1])-zero2)*(-8065.54),(data[::-1]/np.amax(data[eVtopixOK(zero2+x2,lr,rr,ncol):eVtopixOK(zero2+x1,lr,rr,ncol)])),"", label=label,color="red")
          x3=0.2
          x4=-0.1
          ax1.plot(((eV/filedata_dat[i].signals['Wavelength (nm)'][eVtopixOK(zero1+x2,lr,rr,ncol):eVtopixOK(zero1+x1+x3,lr,rr,ncol)])-zero1)*(-8065.54),(data[eVtopixOK(zero1+x2,lr,rr,ncol):eVtopixOK(zero1+x1+x3,lr,rr,ncol)]/np.amax(data[eVtopixOK(zero1+x2,lr,rr,ncol):eVtopixOK(zero1+x1,lr,rr,ncol)])),"", label="ZnPc",color="blue")
          ax1.plot(((eV/filedata_dat[i].signals['Wavelength (nm)'][eVtopixOK(zero2+x2,lr,rr,ncol):eVtopixOK(zero2+x1+x3,lr,rr,ncol)])-zero2)*(-8065.54),(data[eVtopixOK(zero2+x2,lr,rr,ncol):eVtopixOK(zero2+x1+x3,lr,rr,ncol)]/np.amax(data[eVtopixOK(zero2+x2,lr,rr,ncol):eVtopixOK(zero2+x1,lr,rr,ncol)])),"", label="ZnPc$^+$",color="red")
          ax1.plot(((eV/filedata_dat[i].signals['Wavelength (nm)'][eVtopixOK(zero1+x2+x4,lr,rr,ncol):eVtopixOK(zero1+x1,lr,rr,ncol)])-zero1)*(-8065.54),(data[eVtopixOK(zero1+x2+x4,lr,rr,ncol):eVtopixOK(zero1+x1,lr,rr,ncol)]/(1/8*np.amax(data[eVtopixOK(zero1+x2,lr,rr,ncol):eVtopixOK(zero1+x1,lr,rr,ncol)]))),"", label=label,color="blue")
          ax1.plot(((eV/filedata_dat[i].signals['Wavelength (nm)'][eVtopixOK(zero2+x2+x4,lr,rr,ncol):eVtopixOK(zero2+x1,lr,rr,ncol)])-zero2)*(-8065.54),(data[eVtopixOK(zero2+x2+x4,lr,rr,ncol):eVtopixOK(zero2+x1,lr,rr,ncol)]/(1/8*np.amax(data[eVtopixOK(zero2+x2,lr,rr,ncol):eVtopixOK(zero2+x1,lr,rr,ncol)]))),"", label=label,color="red")

          ax1.set_xlabel('Energy shift [cm$^{-1}$]')
          ax1.set_xlim((x2*(-8065.54), x1*(-8065.54))) 
          ax1.set_xlim((-200,1680)) 
          ax1.set_ylim(0,0.1)
          ax1.set_ylim(0,1.1)
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Photon intensity [a.u.]')
    #ax1.set_ylim((0, 700)) 
    ax1.tick_params('y')
    plt.legend(loc="best")
    #ax1.grid(True,linestyle=':')
    #ax1.set_ylim((-2, 5)) 
    #ax1.set_ylim((-2, np.amax(data)+2)) 
    fig.set_size_inches(5*1,3*1)
    if save==True:
        plt.savefig(path+str(filedata_dat[i].header["Filename"])+"V"+str(filedata_dat[i].header["Bias (V)"])+str(x2)+"-"+str(x1)+"-"+label+".svg", dpi=400, bbox_inches = 'tight',transparent=True)
#bgcorplot(1,2,500,800,"nm")

#print(float(ar_dict[99].get("Current avg. (A)"))*1E12,"ahoj")

def bgcorplot_multi_filter(i1,i2,i_bg,x1,x2,unit,save,path):  #converted  
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
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bg):
                i_bg=j
                break
    lr=filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i1].signals['Wavelength (nm)'][-1]  
    ncol=len(filedata_dat[i1].signals['Wavelength (nm)'])-1
    n=float(filedata_dat[i1].header["Number of Accumulations"])
    n2=float(filedata_dat[i2].header["Number of Accumulations"])
    n_bg=float(filedata_dat[i_bg].header["Number of Accumulations"])
    fig, ax1 = plt.subplots() #osy 
    data=[]
    data_nf=[]
    dif=[]
    data_filt=[]
    if unit=="nm":
        for i in range(i1,i2+1):
            data_nf.append(filedata_dat[i].signals['Counts nf'])
            data.append(despike(filedata_dat[i].signals['Counts nf'],0.1,7,lr,rr,ncol))
            ax1.plot(filedata_dat[i].signals['Wavelength (nm)'],filedata_dat[i].signals['Counts nf'],"", label="i="+str(i))
        if (i2-i1)==1:
            dif=abs(np.array(data[1])-np.array(data[0]))
            for i in range(0,len(data[0])):
                if dif[i]>np.median(dif)+10:
                    d=min(data[0][i],data[1][i])
                else:
                    d=(data[0][i]+data[1][i])/2
                data_filt.append(d)    
        else:
            for i in range(0,len(data[0])):
                med=np.median(data, axis=1)
                minimum=np.min(data[:][i])
                number=0
                d=0
                for j in range (0,len(data)):
                    if data[j][i]<=med+10 or data[j][i]<minimum+10 :
                        d=d+data[j][i]
                        n=n+1
                data_filt.append(d/number)    
        data_filt=despike(data_filt,0.1,7,lr,rr,ncol)          
        ax1.plot(filedata_dat[i1].signals['Wavelength (nm)'],data_filt,"", label="filter",color="black") 
            
        ax1.set_xlabel('Wavelength [nm]')
        ax1.set_xlim((x1, x2)) 
        ax2 = ax1.twiny()
        ax2.set_xlim(E(x1),E(x2))
        ax2.set_xlabel('Energy [eV]')
    if unit=="eV":
        ax1.plot(eV/filedata_dat[i].signals['Wavelength (nm)'],filedata_dat[i].signals['Counts nf']-n/n_bg*filedata_dat[i_bg].signals['Counts'],"", label=str(filedata_dat[i].header["Filename"])+" "+str(filedata_dat[i].header["Number of Accumulations"])+" " +str(i)+"B"+str(filedata_dat[i].header["Bias (V)"]))
        ax1.plot(eV/filedata_dat[i2].signals['Wavelength (nm)'],filedata_dat[i2].signals['Counts nf']-n2/n_bg*filedata_dat[i_bg].signals['Counts'],"", label=str(filedata_dat[i2].header["Filename"])+" "+str(filedata_dat[i2].header["Number of Accumulations"])+" " +str(i2)+"B"+str(filedata_dat[i2].header["Bias (V)"]))
        ax1.set_xlabel('Energy [eV]')
        ax1.set_xlim((eV/x1, eV/x2)) 
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Photon intensity [counts/pA]') 
    ax1.tick_params('y')
    plt.legend(loc="best")
    ax1.grid(True,linestyle=':')
   # ax1.set_ylim((-5, 250)) 
    fig.set_size_inches(3.2,3)
    if save==True:
        plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"V"+str(filedata_dat[i1].header["Bias (V)"])+".png", dpi=400, bbox_inches = 'tight')
            

def bgcorplot_multi(i,i2,i_bg,x1,x2,unit,save,path):  #converted  
    if isfloat(i)==True:
        i=i
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i):
                i=j
                break
            
    if isfloat(i2)==True:
        i2=i2
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i2):
                i2=j
                break
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bg):
                i_bg=j
                break
    lr=filedata_dat[i].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i].signals['Wavelength (nm)'][-1]  
    ncol=len(filedata_dat[i].signals['Wavelength (nm)'])-1
    n=float(filedata_dat[i].header["Number of Accumulations"])
    n2=float(filedata_dat[i2].header["Number of Accumulations"])
    n_bg=float(filedata_dat[i_bg].header["Number of Accumulations"])
    fig, ax1 = plt.subplots() #osy 
    if unit=="nm":
        ax1.plot(filedata_dat[i].signals['Wavelength (nm)'],(filedata_dat[i].signals['Counts nf']-n/n_bg*filedata_dat[i_bg].signals['Counts'])/abs(float(filedata_dat[i].header['Current avg. (A)'])*1E12),"", label="ZnPc",color="blue")
        ax1.plot(filedata_dat[i2].signals['Wavelength (nm)'],(filedata_dat[i2].signals['Counts nf']-n2/n_bg*filedata_dat[i_bg].signals['Counts'])/abs(float(filedata_dat[i2].header['Current avg. (A)'])*1E12),"", label="CuPc",color="red")
        ax1.set_xlabel('Wavelength [nm]')
        ax1.set_xlim((x1, x2)) 
        ax2 = ax1.twiny()
        ax2.set_xlim(E(x1),E(x2))
        ax2.set_xlabel('Energy [eV]')
    if unit=="eV":
        ax1.plot(eV/filedata_dat[i].signals['Wavelength (nm)'],filedata_dat[i].signals['Counts nf']-n/n_bg*filedata_dat[i_bg].signals['Counts'],"", label=str(filedata_dat[i].header["Filename"])+" "+str(filedata_dat[i].header["Number of Accumulations"])+" " +str(i)+"B"+str(filedata_dat[i].header["Bias (V)"]))
        ax1.plot(eV/filedata_dat[i2].signals['Wavelength (nm)'],filedata_dat[i2].signals['Counts nf']-n2/n_bg*filedata_dat[i_bg].signals['Counts'],"", label=str(filedata_dat[i2].header["Filename"])+" "+str(filedata_dat[i2].header["Number of Accumulations"])+" " +str(i2)+"B"+str(filedata_dat[i2].header["Bias (V)"]))
        ax1.set_xlabel('Energy [eV]')
        ax1.set_xlim((eV/x1, eV/x2)) 
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Photon intensity [counts/pA]') 
    ax1.tick_params('y')
    plt.legend(loc="best")
    ax1.grid(True,linestyle=':')
    ax1.set_ylim((-5, 250)) 
    fig.set_size_inches(3.2,3)
    if save==True:
        plt.savefig(path+str(filedata_dat[i].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"V"+str(filedata_dat[i].header["Bias (V)"])+".png", dpi=400, bbox_inches = 'tight')
            

def bgcormultiplot(i1,i2,i_bg,x1,x2,unit,save,norm):

    cur=1.0
        
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
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bg):
                i_bg=j
                break

    n_bg=float(filedata_dat[i_bg].header["Number of Accumulations"])
    lr=filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i1].signals['Wavelength (nm)'][-1]   
    ncol=len(filedata_dat[i1].signals['Wavelength (nm)'])-1
    for i in range(i1,i2+1):
        n=float(filedata_dat[i].header["Number of Accumulations"])
        if norm==True:
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
            print(cur)
        fig, ax1 = plt.subplots() #osy 
        if unit=="nm":
            ax1.plot(filedata_dat[i].signals['Wavelength (nm)'],filedata_dat[i].signals['Counts nf']-n/n_bg*filedata_dat[i_bg].signals['Counts'],"", label=str(filedata_dat[i].header["Filename"])+" "+str(filedata_dat[i].header["Number of Accumulations"])+" " +str(i)+"B"+str(filedata_dat[i].header["Bias (V)"]))
            ax1.set_xlabel('wavelength [nm]')
            ax1.set_xlim((x1, x2)) 
            ax2 = ax1.twiny()
            # get the primary axis x tick locations in plot units
            xtickloc = ax1.get_xticks() 
            # set the second axis ticks to the same locations
            ax2.set_xticks(xtickloc)
            # calculate new values for the second axis tick labels, format them, and set them
            x2labels = ['{:.3g}'.format(x) for x in E(xtickloc)]
            ax2.set_xticklabels(x2labels)
            # force the bounds to be the same
            ax2.set_xlim(ax1.get_xlim()) 
            ax2.set_xlabel('Energy [eV]')
        if unit=="eV":
            ax1.plot(eV/filedata_dat[i].signals['Wavelength (nm)'],filedata_dat[i].signals['Counts nf']-n/n_bg*filedata_dat[i_bg].signals['Counts'],"", label=str(filedata_dat[i].header["Filename"])+" "+str(filedata_dat[i].header["Number of Accumulations"])+" " +str(i)+"B"+str(filedata_dat[i].header["Bias (V)"]))
            ax1.set_xlabel('Energy [eV]')
            ax1.set_xlim((eV/x1, eV/x2)) 
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel(' counts ')
        if norm==True:
            ax1.set_ylabel(' counts/pA ')
        #ax1.set_xlim((x1, x2)) 
        ax1.tick_params('y')
        plt.legend()
        ax1.grid(True,linestyle=':')
        
        fig.set_size_inches(4.4,4.4)
        if save==True:
              plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"V"+str(filedata_dat[i1].header["Bias (V)"])+".png", dpi=400, bbox_inches = 'tight',transparent=True)
            
def multiplot(i1,i2,x1,x2,unit,save,norm):
    cur=1.0
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
    lr=filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i1].signals['Wavelength (nm)'][-1]
    ncol=len(filedata_dat[i1].signals['Wavelength (nm)'])-1
    for i in range(i1,i2+1): 
        if norm==True:
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        fig, ax1 = plt.subplots() #osy 
        if unit=="nm":
            ax1.plot(filedata_dat[i].signals['Wavelength (nm)'],filedata_dat[i].signals['Counts']/cur,"",  label=str(filedata_dat[i].header["Filename"])+" "+str(filedata_dat[i].header["Number of Accumulations"])+" " +str(i)+"B"+str(filedata_dat[i].header["Bias (V)"]))
            ax1.set_xlabel('wavelenght [nm]')
            ax1.set_xlim((x1, x2)) 
        if unit=="eV":
                ax1.plot(eV/filedata_dat[i].signals['Wavelength (nm)'],filedata_dat[i].signals['Counts']/cur,"",  label=str(filedata_dat[i].header["Filename"])+" "+str(filedata_dat[i].header["Number of Accumulations"])+" " +str(i)+"B"+str(filedata_dat[i].header["Bias (V)"]))
                ax1.set_xlabel('Energy [eV]')
                ax1.set_xlim((eV/x1, eV/x2)) 
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel(' counts ')
        if norm==True:
            ax1.set_ylabel(' counts/pA ')
        #ax1.set_xlim((x1, x2)) 
        ax1.tick_params('y')
        plt.legend()
        ax1.grid(True,linestyle=':')
        
        fig.set_size_inches(4.4,4.4)       
        if save==True:
             plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"V"+str(filedata_dat[i1].header["Bias (V)"])+".png", dpi=400, bbox_inches = 'tight',transparent=True)
            

def npower(x,a,x0,n,b):
    return a*(x-x0)**n + b
def linfunc(x,a,b):
    return a*x+b
def cubfunc(x,a,b,c,d):
    return a*x**3 +b*x**2+c*x+d
def npower2(x,a,n):
    return a*(x)**n 


def freqdep(i1,i2,i_bg,x1,x2,unit,save,norm):
    cur2=[]
    scounts=[]
    bias=[]
    bias2=[]
    zrel=[]
    freq=[]
    filetransfer="C:/Users/Jirka/ownCloud/Documents/fzu/data/2021/2021-04-07/LS-freqsweepcor-a00001cal_function.txt"# 2021-04-07
    #filetransfer="C:/Users/Jirka/ownCloud/Documents/fzu/data/2020/2020-12-16_2/LS-freqsweepcor-b00001cal_function.txt"# 2020-12-17
    #filetransfer="C:/Users/Jirka/ownCloud/Documents/fzu/data/2020/2020-09-15/KS-freqsweepcor00002_00001cal_function.txt"# 1
    #filetransfer="C:/Users/Jirka/ownCloud/Documents/fzu/data/2020/2020-07-29/LS-freqsweep00108cal_function.txt"# 1
    #filetransfer="C:/Users/Jirka/ownCloud/Documents/fzu/data/2020/2020-08-28/LS-freqsweepcor00004cal_function.txt"
    #filetransfer="C:/Users/Jirka/ownCloud/Documents/fzu/data/2020/2020-08-25/LS-freqsweepcor00004cal_function.txt"
    try:
        fr,transraw,trans = np.loadtxt(filetransfer, skiprows=1, dtype=None,unpack=True) 
    except ValueError:
        fr,trans = np.loadtxt(filetransfer, skiprows=1, dtype=None,unpack=True) 
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
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bg):
                i_bg=j
    lr=filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i1].signals['Wavelength (nm)'][-1]       
    ncol=len(filedata_dat[i1].signals['Wavelength (nm)'])-1
    for i in range(i1,i2+1):
        zrel.append((float(filedata_dat[i].header["Z avg. (m)"])-float(filedata_dat[i1].header["Z avg. (m)"]))*1E9)
        bias.append(filedata_dat[i1].header["Bias (V)"])
        freq.append(float(filedata_dat[i].header["genarb frequency (MHz)"]))
        n=float(filedata_dat[i].header["Number of Accumulations"])
        n_bg=n=float(filedata_dat[i_bg].header["Number of Accumulations"])
        cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        bias2.append(float(filedata_dat[i].header["Bias (V)"]))
        cur2.append(cur)
        scounts.append((sum(filedata_dat[i].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])-n/n_bg*np.sum(filedata_dat[i_bg].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])))  
    fig, ax1 = plt.subplots() #osy
    ax2 = ax1.twinx()
    ax1.set_xlabel('Frequency [MHz]')
    ax1.set_ylabel('Integrated counts '+str(x1)+"-"+str(x2)+" nm")
    #plt.xlim(xmin=0)
    ax1.plot(freq,scounts, label=str(filedata_dat[i1].header["Filename"])+" "+str(filedata_dat[i2].header["Filename"])+str(filedata_dat[i1].header["genarb frequency (MHz)"])+"-"+str(filedata_dat[i2].header["genarb frequency (MHz)"])+"MHz",marker='.',linestyle="None")
    ax1.plot(freq,savgol_filter(scounts,3,0),linestyle="--",color="black")
    #if abs(zrel[0]-zrel[-1])>0.0001:
    ax2.plot(fr,trans, label="calibration" ,marker="^",linestyle="None",color="orange")
    ax2.plot(freq,zrel,linestyle="--",color="blue")
    #    ax2.set_ylabel('Zrel [nm]')
   # else:
    #    ax2.plot(bias2,cur2, label="I(V) dependence" ,marker="^",linestyle="None",color="orange")
    #    ax2.set_ylabel('Current [pA]')    
   # ax1.set_ylim((1000, 1500)) 
    ax1.set_xlim((freq[0], freq[-1])) 
    #ax1.set_ylim((5000, 8000))
    #ax1.set_ylim((0, max(savgol_filter(scounts,3,0))*1.1)) 
    plt.legend()

    ax1.grid(True,linestyle=':')
    fig.set_size_inches(4*4.4,2*4.4)
    f= open(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"r"+str(x1)+"-"+str(x2)+str(filedata_dat[i1].header["genarb frequency (MHz)"])+"-"+str(filedata_dat[i2].header["genarb frequency (MHz)"])+"MHz-freqdep.txt","w+")
    f.write("Frequency [MHz]"+'\t'+"ODMR signal [counts]"+'\t'+"Transfer function [fraction]"+"\n")
    for i in range (0,len(freq)):
        for j in range (0,len(fr)):
            if abs(float(freq[i])-float(fr[j]))<0.1:
                f.write(str("{0:.1f}".format(float(freq[i])))+'\t'+str("{0:.0f}".format(float(scounts[i])))+'\t'+str("{0:.4f}".format(float(trans[j])))+"\n")
                break
            elif j==len(fr)-1:
                f.write(str("{0:.1f}".format(float(freq[i])))+'\t'+str("{0:.0f}".format(float(scounts[i])))+'\t'+str("{0:.4f}".format(0))+"\n")
    f.close()
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"r"+str(x1)+"-"+str(x2)+str(filedata_dat[i1].header["genarb frequency (MHz)"])+"-"+str(filedata_dat[i2].header["genarb frequency (MHz)"])+"MHz-freqdep.png", dpi=400, bbox_inches = 'tight') # nazev souboru 
def vdep(i1,i2,i_bg,x1,x2,unit,save,norm):
    cur2=[]
    scounts=[]
    bias=[]
    bias2=[]
    zrel=[]
        
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
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bg):
                i_bg=j
    lr=filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i1].signals['Wavelength (nm)'][-1]       
    ncol=len(filedata_dat[i1].signals['Wavelength (nm)'])-1
    for i in range(i1,i2+1):
        zrel.append((float(filedata_dat[i].header["Z avg. (m)"])-float(filedata_dat[i1].header["Z avg. (m)"]))*1E9)
        bias.append(filedata_dat[i1].header["Bias (V)"])
        n=float(filedata_dat[i].header["Number of Accumulations"])
        n_bg=n=float(filedata_dat[i_bg].header["Number of Accumulations"])
        cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        bias2.append(float(filedata_dat[i].header["Bias (V)"]))
        cur2.append(cur)
        scounts.append((sum(filedata_dat[i].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])-n/n_bg*np.sum(filedata_dat[i_bg].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])))  
    fig, ax1 = plt.subplots() #osy
    ax2 = ax1.twinx()
    ax1.set_xlabel('Bias [V]')
    ax1.set_ylabel('Integrated counts '+str(x1)+"-"+str(x2)+" nm")
    #plt.xlim(xmin=0)
    ax1.plot(bias2,scounts, label=str(filedata_dat[i1].header["Filename"])+" "+str(filedata_dat[i2].header["Filename"])+" "+"B "+str(filedata_dat[i1].header["Bias (V)"]),marker='x',linestyle="None")
    if abs(zrel[0]-zrel[-1])>0.0001:
        ax2.plot(bias2,zrel, label="Z(V) dependence" ,marker="^",linestyle="None",color="orange")
        ax2.set_ylabel('Zrel [nm]')
    else:
        ax2.plot(bias2,cur2, label="I(V) dependence" ,marker="^",linestyle="None",color="orange")
        ax2.set_ylabel('Current [pA]')    

    plt.legend()
    #popt, pcov=curve_fit(npower2,cur2,scounts)
    #print(popt[1])
    #ax1.plot(cur2,npower(cur2,popt[0],popt[1],popt[2],popt[3]), label="fit")
    #ax1.plot(cur2,npower2(cur2,popt[0],popt[1]), label="fit")
    ax1.grid(True,linestyle=':')
    fig.set_size_inches(4.4,4.4)
    
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"r"+str(x1)+"-"+str(x2)+"Vdep.png", dpi=400, bbox_inches = 'tight') # nazev souboru 

def vdep2(i1,i2,i_bg,x1,x2,y1,y2,unit,save,norm,ratio):
    cur2=[]
    scounts=[]
    scounts_c=[]
    bias=[]
    bias2=[]
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
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bg):
                i_bg=j
    lr=filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i1].signals['Wavelength (nm)'][-1]   
    ncol=len(filedata_dat[i1].signals['Wavelength (nm)'])-1     
    for i in range(i1,i2+1):
        bias.append(filedata_dat[i1].header["Bias (V)"])
        n=float(filedata_dat[i].header["Number of Accumulations"])
        n_bg=n=float(filedata_dat[i_bg].header["Number of Accumulations"])
        cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        bias2.append(float(filedata_dat[i].header["Bias (V)"]))
        cur2.append(cur)
        scounts.append((sum(filedata_dat[i].signals['Counts nf'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])-n/n_bg*np.sum(filedata_dat[i_bg].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])))  
        scounts_c.append(ratio*(sum(filedata_dat[i].signals['Counts nf'][nmtopix(y1,lr,rr,ncol):nmtopix(y2,lr,rr,ncol)])-n/n_bg*np.sum(filedata_dat[i_bg].signals['Counts'][nmtopix(y1,lr,rr,ncol):nmtopix(y2,lr,rr,ncol)])))
    fig, ax1 = plt.subplots() #osy 
    ax1.set_xlabel('Bias [V]')
    ax1.set_ylabel('Integrated counts '+str(x1)+"-"+str(x2)+" nm")
    #plt.xlim(xmin=0)
    ax1.plot(bias2,scounts, label=str(filedata_dat[i1].header["Filename"])+" "+str(filedata_dat[i2].header["Filename"])+" "+str(x1)+"-"+str(x2),marker='x',linestyle="None")
    ax1.plot(bias2,scounts_c, label=str(filedata_dat[i1].header["Filename"])+" "+str(filedata_dat[i2].header["Filename"])+" "+str(y1)+"-"+str(y2),marker='x',linestyle="None")
    plt.legend()
    print(scounts)
    print(scounts_c)
    #popt, pcov=curve_fit(npower2,cur2,scounts)
    #print(popt[1])
    #ax1.plot(cur2,npower(cur2,popt[0],popt[1],popt[2],popt[3]), label="fit")
    #ax1.plot(cur2,npower2(cur2,popt[0],popt[1]), label="fit")
    ax1.grid(True,linestyle=':')
    fig.set_size_inches(4.4,4.4)
    
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"r1"+str(x1)+"-"+str(x2)+"r2"+str(y1)+"-"+str(y2)+"Vdep.png", dpi=400, bbox_inches = 'tight') # nazev souboru 

def vdep3(i1,i2,i_bg,i_div,i_bgdiv,x1,x2,y1,y2,unit,save,norm,ratio):
    cur2=[]
    scounts=[]
    scounts_c=[]
    bias=[]
    bias2=[]
    zrel=[]
        
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
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bg):
                i_bg=j
    lr=filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i1].signals['Wavelength (nm)'][-1] 
    ncol=len(filedata_dat[i1].signals['Wavelength (nm)'])-1       
    for i in range(i1,i2+1):
        zrel.append((float(filedata_dat[i].header["Z avg. (m)"])-float(filedata_dat[i1].header["Z avg. (m)"]))*1E9)
        bias.append(filedata_dat[i1].header["Bias (V)"])
        n=float(filedata_dat[i].header["Number of Accumulations"])
        n_bg=n=float(filedata_dat[i_bg].header["Number of Accumulations"])
        cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        bias2.append(float(filedata_dat[i].header["Bias (V)"]))
        cur2.append(cur)
        scounts.append((sum(filedata_dat[i].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])-n/n_bg*np.sum(filedata_dat[i_bg].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])))  
        scounts_c.append(ratio*(sum(filedata_dat[i].signals['Counts'][nmtopix(y1,lr,rr,ncol):nmtopix(y2,lr,rr,ncol)])-n/n_bg*np.sum(filedata_dat[i_bg].signals['Counts'][nmtopix(y1,lr,rr,ncol):nmtopix(y2,lr,rr,ncol)])))
    fig, ax1 = plt.subplots() #osy 
    ax1.set_xlabel('Bias [V]')
    ax1.set_ylabel(r'Integrated counts$ \times 1000$')
    ax2 = ax1.twinx()
    
    #plt.xlim(xmin=0)
    scounts=np.array(scounts)
    scounts_c=np.array(scounts_c)
    ax1.plot(bias2,scounts*0.001, label=str(x1)+"-"+str(x2)+" nm",marker='x',linestyle="None",color="blue")
    ax1.plot(bias2,scounts_c*0.001, label=str(y1)+"-"+str(y2)+" nm",marker='x',linestyle="None",color="red")
    if abs(zrel[0]-zrel[-1])>0.0001:
        ax2.plot(bias2,zrel, label="Z(V) dependence" ,marker="^",linestyle="None",color="orange")
        ax2.set_ylabel('Zrel [nm]')
    else:
        ax2.plot(bias2,cur2, label="I(V) dependence" ,marker="^",linestyle="None",color="orange")
        ax2.set_ylabel('Current [pA]')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2,loc="best")
    print(scounts)
    print(scounts_c)
    #popt, pcov=curve_fit(npower2,cur2,scounts)
    #print(popt[1])
    #ax1.plot(cur2,npower(cur2,popt[0],popt[1],popt[2],popt[3]), label="fit")
    #ax1.plot(cur2,npower2(cur2,popt[0],popt[1]), label="fit")
    ax1.grid(True,linestyle=':')
    fig.set_size_inches(9/2.6, 10/2.6)
    
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"r1"+str(x1)+"-"+str(x2)+"r2"+str(y1)+"-"+str(y2)+"Vdep.png", dpi=400, bbox_inches = 'tight') # nazev souboru 

def vdep4(i1,i2,i_bg,i_div,i_bgdiv,x1,x2,y1,y2,unit,save,norm):
    cur2=[]
    scounts=[]
    scounts_c=[]
    bias=[]
    bias2=[]
    zrel=[]
        
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
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bg):
                i_bg=j
    if isfloat(i_div)==True:
        i_div=i_div
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_div):
                i_div=j
                break
    if isfloat(i_bgdiv)==True:
        i_bgdiv=i_bgdiv
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bgdiv):
                i_bgdiv=j
                break


    lr=filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i1].signals['Wavelength (nm)'][-1]
    ncol=len(filedata_dat[i1].signals['Wavelength (nm)'])-1
    norm_spec=savgol_filter(filedata_dat[i_div].signals['Counts'],31,0) - savgol_filter(filedata_dat[i_bgdiv].signals['Counts'],31,0)       
    for i in range(i1,i2+1):
        zrel.append((float(filedata_dat[i].header["Z avg. (m)"])-float(filedata_dat[i1].header["Z avg. (m)"]))*1E9)
        bias.append(filedata_dat[i1].header["Bias (V)"])
        n=float(filedata_dat[i].header["Number of Accumulations"])
        n_bg=n=float(filedata_dat[i_bg].header["Number of Accumulations"])
        cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        bias2.append(float(filedata_dat[i].header["Bias (V)"]))
        cur2.append(cur)
        data=(filedata_dat[i].signals['Counts']-n/n_bg*savgol_filter(filedata_dat[i_bg].signals['Counts'],31,0))/norm_spec
#        data=despike(data,0.1,7,nmtopix(700,lr,rr,ncol),nmtopix(940,lr,rr,ncol))   
        
        
        scounts.append(sum(data[nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])) 
        scounts_c.append(sum(data[nmtopix(y1,lr,rr,ncol):nmtopix(y2,lr,rr,ncol)])) 
    fig, ax1 = plt.subplots() #osy 
    ax1.set_xlabel('Bias (V)')
    ax1.set_ylabel(r'Norm. integrated counts (a.u.)')
    ax2 = ax1.twinx()
    
    #plt.xlim(xmin=0)
    scounts=np.array(scounts)
    scounts_c=np.array(scounts_c)
    ax1.plot(bias2,scounts, label=str(x1)+"-"+str(x2)+" nm",marker='x',linestyle="None",color="blue")
    ax1.plot(bias2,scounts_c, label=str(y1)+"-"+str(y2)+" nm",marker='x',linestyle="None",color="red")
    print(scounts_c[-1]/scounts[-1], "ratio")
    f= open(pathcopy+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"r1"+str(x1)+"-"+str(x2)+"r2"+str(y1)+"-"+str(y2)+"Vdep4.txt","w+")
    if abs(zrel[0]-zrel[-1])>0.0001:
        ax2.plot(bias2,np.array(zrel)*10, label="Z(V) dependence" ,marker="^",linestyle="None",color="black",markersize=1)
        ax2.set_ylabel(r'$\mathrm{Z_{rel}}$ ($\mathrm{ \AA}$)')
        f.write("Bias"+'\t'+"Plasmon normalized X1 intensity (a.u.)"+'\t'+"Plasmon normalized X2 intensity (a.u.)" +'\t'+"Z_rel (Angstroms)"+"\n")
        lastparameter=np.array(zrel)*10
    else:
        ax2.plot(bias2,cur2, label="I(V) dependence" ,marker="^",linestyle="None",color="black",markersize=1)
        ax2.set_ylabel('Current [pA]')
        f.write("Bias"+'\t'+"Plasmon normalized X1 intensity (a.u.)"+'\t'+"Plasmon normalized X2 intensity (a.u.)" +'\t'+"Current (pA)"+"\n")
        lastparameter=cur2
    for index in range(0,len(bias2)):
        f.write(str("{0:.2f}".format(bias2[index])) +"\t" +str("{0:.5f}".format(scounts[index]))  +"\t" +str("{0:.5f}".format(scounts_c[index])) +"\t" +str("{0:.5f}".format(lastparameter[index])) + "\n")
        #f.write("Bias"+'\t'+"Plasmon normalized X1 intensity (a.u.)"+'\t'+"Plasmon normalized X2 intensity (a.u.)" +'\t'+"Current (pA)"+"\n")
    f.close() 
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    #ax2.legend(lines + lines2, labels + labels2,loc="best")
    print(scounts)
    print(scounts_c)
    #popt, pcov=curve_fit(npower2,cur2,scounts)
    #print(popt[1])
    #ax1.plot(cur2,npower(cur2,popt[0],popt[1],popt[2],popt[3]), label="fit")
    #ax1.plot(cur2,npower2(cur2,popt[0],popt[1]), label="fit")
    ax1.xaxis.set_major_locator(plt.MultipleLocator(0.4))
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax2.yaxis.set_major_locator(plt.MultipleLocator(1))
    ax2.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
    ax1.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax1.yaxis.set_minor_locator(plt.MultipleLocator(2.5))
    plt.gca().invert_xaxis()
    #ax1.xaxis.set_major_locator(plt.LinearLocator(-3.2,-2.2))
    #ax1.grid(True,linestyle=':')
    ax1.xaxis.set_ticks_position('both')
    ax1.tick_params(axis='both',which='both',direction='in')
    ax2.tick_params(axis='both',which='both',direction='in')
    mult=1.4
    fig.set_size_inches(mult*2,mult*2)
    ax1.set_xlim((-2.175, -3.225)) 
    
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"r1"+str(x1)+"-"+str(x2)+"r2"+str(y1)+"-"+str(y2)+"Vdep4.png", dpi=400, bbox_inches = 'tight') # nazev souboru 
            plt.savefig(pathcopy+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"r1"+str(x1)+"-"+str(x2)+"r2"+str(y1)+"-"+str(y2)+"Vdep4.svg", dpi=400, bbox_inches = 'tight',transparent=True) # nazev souboru 

def idep(i1,i2,i_bg,x1,x2,unit,save,norm):
    cur2=[]
    scounts=[]
    bias=[]
        
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
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bg):
                i_bg=j
    lr=filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i1].signals['Wavelength (nm)'][-1]
    ncol=len(filedata_dat[i1].signals['Wavelength (nm)'])-1        
    for i in range(i1,i2+1):
        bias.append(filedata_dat[i1].header["Bias (V)"])
        n=float(filedata_dat[i].header["Number of Accumulations"])
        n_bg=n=float(filedata_dat[i_bg].header["Number of Accumulations"])
        cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        
        cur2.append(cur)
        scounts.append((sum(filedata_dat[i].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])-n/n_bg*np.sum(filedata_dat[i_bg].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])))  
    fig, ax1 = plt.subplots() #osy 
    ax1.set_xlabel('Current [pA]')
    ax1.set_ylabel('Integrated counts '+str(x1)+"-"+str(x2)+" nm")
    #plt.xlim(xmin=0)
    ax1.plot(cur2,scounts, label=str(filedata_dat[i1].header["Filename"])+" "+str(filedata_dat[i2].header["Filename"])+"-"+str(x1)+"-"+str(x2),marker='x',linestyle="None")
    popt, pcov=curve_fit(npower2,cur2,scounts)
    print(popt[1])
   # ax1.plot(cur2,npower(cur2,popt[0],popt[1],popt[2],popt[3]), label="fit")
    ax1.plot(cur2,npower2(cur2,popt[0],popt[1]), label="fit exp="+str("{0:.4f}".format(popt[1])))
    ax1.grid(True,linestyle=':')
    fig.set_size_inches(4.4,4.4)
    plt.legend()
    
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"-"+str(x1)+"-"+str(x2)+".png", dpi=400, bbox_inches = 'tight') # nazev souboru 
            
def ivdep(i1,i2,i_bg,x1,x2,unit,save,norm):
    cur2=[]
    scounts=[]
    bias=[]
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
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bg):
                i_bg=j
    lr=filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i1].signals['Wavelength (nm)'][-1]
    ncol=len(filedata_dat[i1].signals['Wavelength (nm)'])-1      
    for i in range(i1,i2+1):
        bias.append(filedata_dat[i].header["Bias (V)"])
        n=float(filedata_dat[i].header["Number of Accumulations"])
        n_bg=n=float(filedata_dat[i_bg].header["Number of Accumulations"])
        cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        cur2.append(cur)
        scounts.append((sum(filedata_dat[i].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])-n/n_bg*np.sum(filedata_dat[i_bg].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])))  
    
    fig, ax1 = plt.subplots()
    cax=ax1.scatter(cur2, scounts, c=bias, cmap=cm.coolwarm, label="data")
    ax1.set_title('I-V plasmon dependence')
    # Make the y-axis label, ticks and tick labels match the line color.
   # cbar = fig.colorbar(cax, ticks=[1.9, 2.0, 2.1, 2.2, 2.3,2.4,2.5])
    cbar = fig.colorbar(cax, ticks=[-2.2, -2.3, -2.4, -2.5, -2.6, -2.7, -2.8, -2.9, -3.0])
    #cbar.ax.set_yticklabels(['1.9','2.0' ,'2.1', '2.2', '2.3','2.4','2.5'])
    #cbar.ax.set_yticklabels(['-2.2','-2.3' ,'-2.4', '-2.5', '-2.6','-2.7', '-2.8', '-2.9', '-3.0'])
    plt.xlim(xmin=60)
    plt.xlim(xmax=180)
    cbar.set_label(label='Sample bias [V]')     
    ax1.set_xlabel('Current [pA]')
    ax1.set_ylabel('Integrated counts '+str(x1)+"-"+str(x2)+" nm")
    #ax1.plot(cur2,npower2(cur2,popt[0],popt[1]), label="fit")
    ax1.grid(True,linestyle=':')
    fig.set_size_inches(4.4,4.4)
    
    if save==True:
        plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"-"+"B "+str(filedata_dat[i1].header["Bias (V)"])+".png", dpi=400, bbox_inches = 'tight') # nazev souboru 
            
def profile_hm(i1,i2,i_bg,s,e,x1,x2,aspect,contrast,unit,save,norm):
    cur2=[]
    scounts=[]
    bias=[]
    #ar_cor= np.copy(ar_np)
    counts_cor=[]
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
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bg):
                i_bg=j
    lr=filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i1].signals['Wavelength (nm)'][-1]
    ncol=len(filedata_dat[i1].signals['Wavelength (nm)'])-1
    if i_bg<-0.1 and i_bg> -2:
        n_bg=0
        for j in range(i1+s,i1+s+e+1):
            n_bg=n_bg+float(filedata_dat[j].header["Number of Accumulations"])
        if (s-e)==0:
            n_bg=np.Inf
        else:
            n_bg=n_bg/(s-e)
    elif i_bg<-2.1:
        n_bg=0
        d_bg=0
        sum_bg=0
        for j in range(i1,i2):
            if (abs(float(filedata_dat[j].header["Current avg. (A)"]))*1E12)<5:
                n_bg=float(filedata_dat[j].header["Number of Accumulations"])
                sum_bg=sum_bg+np.sum(filedata_dat[j].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])#/n_bg
                d_bg=d_bg+1
        print(d_bg)
        sum_bg=sum_bg/d_bg
    else:
        n_bg=float(filedata_dat[i_bg].header["Number of Accumulations"])
        
    for i in range(i1,i2+1):
        bias.append(filedata_dat[i].header["Bias (V)"])
        n=float(filedata_dat[i].header["Number of Accumulations"])
        n_bg=n=float(filedata_dat[i_bg].header["Number of Accumulations"])
        try:
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"])+float(filedata_dat[i].header["Current avg. (A) end"]))/2*1E12
        except KeyError:
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        cur2.append(cur)
        counts_cor.append(np.subtract(filedata_dat[i].signals['Counts'], n/n_bg*filedata_dat[i_bg].signals['Counts']))
    #fig, ax1 = plt.subplots()
    #plt.imshow(ar_np[i1:i2+1,1,nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)], cmap='viridis',interpolation='None',extent=[x1,x2,i1,i2+1])
    #plt.colorbar
    #fig.set_size_inches(4.4,4.4)
    counts_cor=np.array(counts_cor)
    fig, ax1 = plt.subplots()
    #plt.imshow(counts_cor[i1:i2+1,nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)], cmap='viridis',interpolation='None',extent=[x1,x2,i1,i2+1],origin="lower")
    if contrast==True:
        data = (counts_cor - np.min(counts_cor, axis=1)[:, np.newaxis]) / np.ptp(counts_cor, axis=1)[:, np.newaxis]
    else:
        data=counts_cor
    if unit=="nm":
        plt.imshow(data[:,nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)], cmap='afmhot',interpolation='None',extent=[x1,x2,i1,i2+1],aspect=aspect,origin="lower")
    else:
        data2=data[:,::-1]
        plt.imshow(data2[:,eVtopix(x1,lr,rr,ncol):eVtopix(x2,lr,rr,ncol)], cmap='afmhot',interpolation='None',extent=[x1,x2,i1,i2+1],aspect=(x2-x1)*aspect/(eVtopix(x2,lr,rr,ncol)-eVtopix(x1,lr,rr,ncol)),origin="lower")
    plt.colorbar
    print((eVtopix(x2,lr,rr,ncol)-eVtopix(x1,lr,rr,ncol))/(i2+1-i1))
    fig.set_size_inches(4.4,4.4)
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+".png", dpi=400, bbox_inches = 'tight') # nazev souboru

def profile_pcol(i1,i2,i_bg,s,e,x1,x2,gamma,contrast,unit,save,norm):
    cur2=[]
    scounts=[]
    bias=[]
    x_coord=[]
    y_coord=[]
    d=[]
    counts_cor=[]
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
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bg):
                i_bg=j
    lr=filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i1].signals['Wavelength (nm)'][-1]
    ncol=len(filedata_dat[i1].signals['Wavelength (nm)'])-1
    if i_bg<-0.1 and i_bg> -2:
        n_bg=0
        for j in range(i1+s,i1+s+e+1):
            n_bg=n_bg+float(filedata_dat[j].header["Number of Accumulations"])
        if (s-e)==0:
            n_bg=np.Inf
        else:
            n_bg=n_bg/(s-e)
    elif i_bg<-2.1:
        n_bg=0
        d_bg=0
        sum_bg=0
        for j in range(i1,i2):
            if (abs(float(filedata_dat[j].header["Current avg. (A)"]))*1E12)<5:
                n_bg=float(filedata_dat[j].header["Number of Accumulations"])
                sum_bg=sum_bg+np.sum(filedata_dat[j].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])#/n_bg
                d_bg=d_bg+1
        print(d_bg)
        sum_bg=sum_bg/d_bg
    else:
        n_bg=float(filedata_dat[i_bg].header["Number of Accumulations"])
        f= open(pathcopy+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"diagonal","w+")
        f.write("diagonal (nm)\n")
    for i in range(i1,i2+1):
        x_coord.append(float(filedata_dat[i].header['X (m)']))
        y_coord.append(float(filedata_dat[i].header['Y (m)']))
        bias.append(filedata_dat[i].header["Bias (V)"])
        n=float(filedata_dat[i].header["Number of Accumulations"])
        n_bg=n=float(filedata_dat[i_bg].header["Number of Accumulations"])
        try:
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"])+float(filedata_dat[i].header["Current avg. (A) end"]))/2*1E12
        except KeyError:
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        cur2.append(cur)
        counts_cor.append(np.subtract(filedata_dat[i].signals['Counts'], n/n_bg*savgol_filter(filedata_dat[i_bg].signals['Counts'],21,1)))
    #fig, ax1 = plt.subplots()
    #plt.imshow(ar_np[i1:i2+1,1,nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)], cmap='viridis',interpolation='None',extent=[x1,x2,i1,i2+1])
    #plt.colorbar
    #fig.set_size_inches(4.4,4.4)
    x0=np.median(x_coord)
    y0=np.median(y_coord)
    for i in range(0,i2+1-i1):
        if i<(i2+1-i1)/2:
            c=-1E9*((float(x_coord[i])-x0)**2+((float(y_coord[i])-y0)**2))**0.5
        else:
            c=+1E9*((float(x_coord[i])-x0)**2+((float(y_coord[i])-y0)**2))**0.5
        d.append(c)  
        f.write(str("{0:.4f}".format(c)+"\n"))
    print(d)
    counts_cor=np.array(counts_cor)
    fig, ax1 = plt.subplots()
    #plt.imshow(counts_cor[i1:i2+1,nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)], cmap='viridis',interpolation='None',extent=[x1,x2,i1,i2+1],origin="lower")
    if contrast==True:
        data = (counts_cor - np.min(counts_cor, axis=1)[:, np.newaxis]) / np.ptp(counts_cor, axis=1)[:, np.newaxis]
    else:
        data=counts_cor
    cmap=matplotlib.cm.get_cmap('seismic')
    matplotlib.colors.LinearSegmentedColormap.set_gamma(cmap, gamma)
    if unit=="nm":
        x = filedata_dat[i1].signals['Wavelength (nm)'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)]
        #y = np.arange(0.5,i2-i1+1.5)
        y=d
        X,Y = np.meshgrid(x,y)
        plt.pcolormesh(X,Y,data[:,nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)], cmap=cmap)
        ax1.set_xlabel('Wavelength [nm]')
    else:
        data2=data[:,::-1]
        x = filedata_dat[i1].signals['Wavelength (nm)'][::-1]
        x=eV/x[eVtopix(x1,lr,rr,ncol):eVtopix(x2,lr,rr,ncol)]
        #y = np.arange(0.5,i2-i1+1.5)
        y=d
        X,Y = np.meshgrid(x,y)
        plt.pcolormesh(X,Y,data2[:,eVtopix(x1,lr,rr,ncol):eVtopix(x2,lr,rr,ncol)], cmap=cmap)
       # ax1.set_xlabel('Energy (eV)')
        ax1.xaxis.set_ticks_position('top')
       # ax1.xaxis.set_label_position('top')
   # ax1.set_ylabel('Diagonal profile (nm)')
    plt.colorbar
    #print((eVtopix(x2,lr,rr,ncol)-eVtopix(x1,lr,rr,ncol))/(i2+1-i1))
    fig.set_size_inches(1.8,1.8)
    f.close()
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+".png", dpi=400, bbox_inches = 'tight',transparent=True) # nazev souboru
            plt.savefig(pathcopy+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+".svg", dpi=400, bbox_inches = 'tight')
            
            
            
            
def bias_profile_pcol(i1, i2, x1, x2, gamma=1.0, contrast=False, unit="nm", save=True,bg=305,ker_size=5,sig_th=5,center_wl=632.8):
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
        
    cmap = matplotlib.cm.get_cmap('seismic')
    matplotlib.colors.LinearSegmentedColormap.set_gamma(cmap, gamma)

    fig, ax = plt.subplots()

    if unit == "nm":
        x = filedata_dat[i1].signals['Wavelength (nm)'][nmtopix(x1, lr, rr, ncol):nmtopix(x2, lr, rr, ncol)]
        #y = np.arange(i2 - i1 + 1)
        y = np.array(bias)
        X, Y = np.meshgrid(x, y)
        plt.pcolormesh(X, Y, data[:, nmtopix(x1, lr, rr, ncol):nmtopix(x2, lr, rr, ncol)], cmap=cmap)
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

    if save:
        filename = f"{filedata_dat[i1].header['Filename']}-{filedata_dat[i2].header['Filename']}"
        plt.savefig(path + filename + ".png", dpi=300, bbox_inches='tight', transparent=True)
        plt.savefig(path + filename + ".svg", dpi=300, bbox_inches='tight', transparent=True)
     #   plt.savefig(pathcopy + filename + "_quick.svg", dpi=300, bbox_inches='tight')

def bias_profile_pcol_flip(i1, i2, x1, x2, gamma=1.0, contrast=False, unit="nm", save=True, bg=305, ker_size=5, sig_th=5, center_wl=632.8, flip_axes=False):
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

    fig.set_size_inches(4.0, 3.0)
    plt.colorbar()

    # Save
    if save:
        filename = f"{filedata_dat[i1].header['Filename']}-{filedata_dat[i2].header['Filename']}"
        plt.savefig(path + filename + ".png", dpi=300, bbox_inches='tight', transparent=True)
        plt.savefig(path + filename + ".svg", dpi=300, bbox_inches='tight', transparent=True)     
def z_profile_pcol(i1, i2, x1, x2, gamma=1.0, contrast=False, unit="nm", save=True,bg=305,ker_size=5,sig_th=5,center_wl=632.8):
    """
    example  
    quick_profile_pcol(i1="LS-PL-3ML-vdep-b-00001", i2="LS-PL-3ML-vdep-b-00051", x1=650, x2=700)
    
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
    z=[]
    for i in range(i1, i2 + 1):
        scan_counts = filedata_dat[i].signals['Counts']
        counts.append(scan_counts)
        z.append(float(filedata_dat[i].header["Z (m)"]))
    counts = np.array(counts)
    data = despike_2d_adapt(counts,kernel_size=ker_size,sigma_thresh=sig_th)-bg
    if contrast:
        data = (data - np.min(data, axis=1)[:, None]) / np.ptp(data, axis=1)[:, None]
        
    cmap = matplotlib.cm.get_cmap('seismic')
    matplotlib.colors.LinearSegmentedColormap.set_gamma(cmap, gamma)

    fig, ax = plt.subplots()
    y = (np.array(z)-np.min(z))*1e12
    if unit == "nm":
        x = filedata_dat[i1].signals['Wavelength (nm)'][nmtopix(x1, lr, rr, ncol):nmtopix(x2, lr, rr, ncol)]
        #y = np.arange(i2 - i1 + 1)
        X, Y = np.meshgrid(x, y)
        plt.pcolormesh(X, Y, data[:, nmtopix(x1, lr, rr, ncol):nmtopix(x2, lr, rr, ncol)], cmap=cmap)
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
        X, Y = np.meshgrid(x, y)
        plt.pcolormesh(X, Y, data, cmap=cmap)
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
    ax.set_ylabel('Z (pm)')
    fig.set_size_inches(4.0, 3.0)
    plt.colorbar()

    if save:
        filename = f"{filedata_dat[i1].header['Filename']}-{filedata_dat[i2].header['Filename']}"
        plt.savefig(path + filename + ".png", dpi=300, bbox_inches='tight', transparent=True)
        plt.savefig(path + filename + ".svg", dpi=300, bbox_inches='tight', transparent=True)
     #   plt.savefig(pathcopy + filename + "_quick.svg", dpi=300, bbox_inches='tight')
    
def profile_wf(i1,i2,i_bg,s,e,x1,x2,aspect,contrast,unit,save,norm):
    cur2=[]
    scounts=[]
    bias=[]
    #ar_cor= np.copy(ar_np)
    counts_cor=[]
    sum_bg=np.zeros(len(filedata_dat[0].signals['Counts']))
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
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bg):
                i_bg=j
    lr=filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i1].signals['Wavelength (nm)'][-1]
    ncol=len(filedata_dat[i1].signals['Wavelength (nm)'])-1
    if i_bg<-0.1:
        n_bg=0
        for j in range(i1+s,i1+s+e+1):
            n_bg=float(filedata_dat[j].header["Number of Accumulations"])
            a=filedata_dat[j].signals['Counts']/n_bg
            sum_bg=np.add(a,sum_bg)
        if abs(e-s)>0.1:
            sum_bg=sum_bg/(e-s+1)
        else:
            sum_bg=np.zeros(len(filedata_dat[0].signals['Counts'])) 
    elif i_bg<-2.1:
        n_bg=0
        d_bg=0
        for j in range(i1,i2):
            if (abs(float(filedata_dat[j].header["Current avg. (A)"]))*1E12)<5:
                n_bg=float(filedata_dat[j].header["Number of Accumulations"])
                sum_bg=sum_bg+np.sum(filedata_dat[j].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])#/n_bg
                d_bg=d_bg+1
        print(d_bg)
        sum_bg=sum_bg/d_bg
    else:
        n_bg=float(filedata_dat[i_bg].header["Number of Accumulations"])
        sum_bg=filedata_dat[i_bg].signals['Counts']/n_bg
    fig, ax1 = plt.subplots() #osy   
    j=0
    increment=350
    if norm==True:
        increment=increment/70
    for i in range(i1,i2+1):
        
        bias.append(filedata_dat[i].header["Bias (V)"])
        n=float(filedata_dat[i].header["Number of Accumulations"])
        try:
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"])+float(filedata_dat[i].header["Current avg. (A) end"]))/2*1E12
        except KeyError:
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        cur2.append(cur)
        if norm==False:
            counts_cor.append(np.subtract(filedata_dat[i].signals['Counts'], n*sum_bg))
        else:
            counts_cor.append((np.subtract(filedata_dat[i].signals['Counts'], n*sum_bg))/cur)
        ax1.plot(filedata_dat[i].signals['Wavelength (nm)'],counts_cor[-1]+j,"",color=matplotlib.cm.get_cmap('brg')((i-i1)/(i2-i1) ))
        j=j+increment
    #fig, ax1 = plt.subplots()
    #plt.imshow(ar_np[i1:i2+1,1,nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)], cmap='viridis',interpolation='None',extent=[x1,x2,i1,i2+1])
    #plt.colorbar
    #fig.set_size_inches(4.4,4.4)

    #ax3 = ax1.twiny()
    if norm==False:
        ax1.set_ylabel('Photon intensity [counts]')
    else:
        ax1.set_ylabel('Photon intensity [counts/pA]')
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_xlim((x1, x2))
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax2 = ax1.twiny()
    # get the primary axis x tick locations in plot units
    x1nm=math.floor(E(x1)*10)/10
    x2nm=math.ceil(E(x2)*10)/10
    x1nm_min=math.floor(E(x1)*50)/50
    x2nm_min=math.ceil(E(x2)*50)/50             
    xtl=np.linspace(x1nm,x2nm,int(round(abs(x2nm-x1nm)*10))+1)
    xtl_min=np.linspace(x1nm_min,x2nm_min,int(round(abs(x2nm_min-x1nm_min)*50))+1)
    xtickloc=[WL(x) for x in xtl]
    xtickloc_min=[WL(x) for x in xtl_min]
    #xtickloc = ax1.get_xticks() 
    #print(xtl)
    # set the second axis ticks to the same locations
    ax2.set_xticks(xtickloc)
    ax2.set_xticks(xtickloc_min, minor=True)
    # calculate new values for the second axis tick labels, format them, and set them
    x2labels = ['{:.3g}'.format(E(x)) for x in xtickloc]
    ax2.set_xticklabels(x2labels)
    # force the bounds to be the same
    ax2.set_xlim(ax1.get_xlim()) 
    ax2.set_xlabel('Energy [eV]')    

    ax1.grid(True,linestyle=':')
    fig.set_size_inches(6/2.5,8/2.5)
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"wf.png", dpi=400, bbox_inches = 'tight',transparent=True) # nazev souboru 

def profile_wf3d(i1,i2,i_bg,s,e,x1,x2,aspect,contrast,unit,save,norm):
    from mpl_toolkits.mplot3d import Axes3D
    cur2=[]
    scounts=[]
    bias=[]
    #ar_cor= np.copy(ar_np)
    counts_cor=[]
    wl=[]
    sum_bg=np.zeros(len(filedata_dat[0].signals['Counts']))
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
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bg):
                i_bg=j
    lr=filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i1].signals['Wavelength (nm)'][-1]
    ncol=len(filedata_dat[i1].signals['Wavelength (nm)'])-1
    if i_bg<-0.1:
        n_bg=0
        for j in range(i1+s,i1+s+e+1):
            n_bg=float(filedata_dat[j].header["Number of Accumulations"])
            a=filedata_dat[j].signals['Counts']/n_bg
            sum_bg=np.add(a,sum_bg)
        if abs(e-s)>0.1:
            sum_bg=sum_bg/(e-s+1)
        else:
            sum_bg=np.zeros(len(filedata_dat[0].signals['Counts'])) 
    else:
        n_bg=float(filedata_dat[i_bg].header["Number of Accumulations"])
        sum_bg=filedata_dat[i_bg].signals['Counts']/n_bg
    fig= plt.figure() #osy 
    ax1 = fig.add_subplot(111, projection='3d')
    j=0
    increment=200
    if norm==True:
        increment=increment/70
    for i in range(i1,i2+1):
        j=j+increment
        bias.append(filedata_dat[i].header["Bias (V)"])
        n=float(filedata_dat[i].header["Number of Accumulations"])
        try:
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"])+float(filedata_dat[i].header["Current avg. (A) end"]))/2*1E12
        except KeyError:
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        cur2.append(cur)
        wl.append(filedata_dat[i].signals['Wavelength (nm)'])
        if norm==False:
            counts_cor.append(np.subtract(filedata_dat[i].signals['Counts'], n*sum_bg))
        else:
            counts_cor.append((np.subtract(filedata_dat[i].signals['Counts'], n*sum_bg))/cur)
        #ax1.plot(filedata_dat[i].signals['Wavelength (nm)'],counts_cor[-1]+j,"",color=matplotlib.cm.get_cmap('brg')((i-i1)/(i2-i1) ))
    number=np.arange(i1,i2+1,1)
    #X,Y = np.meshgrid(wl,number)
    print(nmtopix(x1,lr,rr,ncol))
    print(nmtopix(x2,lr,rr,ncol))
    Z=np.array(counts_cor)
    X,Y = np.meshgrid(wl[0][:],number)
    print(np.shape(X))
    print(np.shape(wl))
    print(np.shape(Z))
    ax1.plot_surface(X, Y, Z, rstride=1, cstride=1000, color='red', shade=True, lw=5)
    #fig, ax1 = plt.subplots()
    #plt.imshow(ar_np[i1:i2+1,1,nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)], cmap='viridis',interpolation='None',extent=[x1,x2,i1,i2+1])
    #plt.colorbar
    #fig.set_size_inches(4.4,4.4)
   # ax1.set_xlim((x1, x2)) 
    if norm==False:
        ax1.set_ylabel('Photon intensity [counts]')
    else:
        ax1.set_ylabel('Photon intensity [counts/pA]')
    ax1.set_xlabel('Wavelength [nm]')
   # ax1.set_xlim((x1, x2)) 
    

    #ax1.grid(True,linestyle=':')
    fig.set_size_inches(4.4,4.4)
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"wf.png", dpi=400, bbox_inches = 'tight') # nazev souboru 


def raman_wfplot(i1,i2,bg,unit,save,**kwargs):
    """ example raman_wfplot("LS-PTCDA-up-g4-b00002","LS-PTCDA-up-g4-b00141",300,"cm",True,gamma=0.3,tol2d=3)
       example raman_wfplot("LS-PTCDA-updown-b00001","LS-PTCDA-updown-b00362",300,"cm",True,gamma=0.3,tol2d=3,BS=("BS-updown-b00001","BS-updown-b00181"),norm=True)
    """
    
    cur2=[]
    Z2=[] #ab. height
    scounts=[]
    scounts2=[]
    bias=[]
    ncounts=[]
    ncounts2=[]
    sum_bg=0
    sum_bgar=[]
    sum_bg2=0
    counts_cor=[]
    dIdV=[]
    dIdV_norm=[]
    V=[]
    cur_mod=[]
    BS=False
    appendix=""
    if isfloat(i1)==True: #load AALS start
        i1=i1
    else:
        for j in range (0,len(file_numbers)):
            if str(filedata_dat[j].header["Filename"])==str(i1):
                i1=j
                break
    if isfloat(i2)==True: #load AALS end
        i2=i2
    else:
        for j in range (0,len(file_numbers)):
            if str(filedata_dat[j].header["Filename"])==str(i2):
                i2=j
                break
    if "BS" in kwargs: #Load BS optional 
        BS=True
        appendix="BS"
        i3=kwargs["BS"][0]
        if isfloat(i3)==True:
            i3=i3
        else:
            for j in range (0,len(file_numbersBS)):
                if str(filedata_datBS[j].basename[:-4])==str(i3):
                    i3=j
                    break
        i4=kwargs["BS"][1]
        if isfloat(i4)==True:
            i4=i4
        else:
            for j in range (0,len(file_numbersBS)):
                if str(filedata_datBS[j].basename[:-4])==str(i4):
                    i4=j
                    break
        
    lr=filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i1].signals['Wavelength (nm)'][-1]
    ncol=len(filedata_dat[i1].signals['Wavelength (nm)'])-1
    print(i1)
    if abs(float((filedata_dat[i1].header["GWL"]))- float((filedata_dat[i1+1].header["GWL"])))< 1: # if we did two measurements with different ceter wl
        for i in range(i1,i2+1):
            bias.append(filedata_dat[i].header["Bias (V)"])
            Z=(float(filedata_dat[i].header["Z (m)"]))*1E9
            Z2.append(Z)
    
            try:
                cur=abs(float(filedata_dat[i].header["Current avg. (A)"])+float(filedata_dat[i].header["Current avg. (A) end"]))/2*1E12
            except KeyError:
                cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
            cur2.append(cur)
            counts_cor.append(np.subtract(despike_multi_R(i,**kwargs), bg))
         #   print(np.min(despike_multi_R(i,width=15)-despike_multi_R(i)))
        wl_array=filedata_dat[i1].signals['Wavelength (nm)']
    else:
        ind=0
        for i in range(i1,i2+1):
            if (ind % 2)==0:         
                lr1=filedata_dat[i].signals['Wavelength (nm)'][0]
                rr1=filedata_dat[i].signals['Wavelength (nm)'][-1]
                bias.append(filedata_dat[i].header["Bias (V)"])
                Z=(float(filedata_dat[i].header["Z (m)"]))*1E9
                Z2.append(Z)
        
                try:
                    cur=abs(float(filedata_dat[i].header["Current avg. (A)"])+float(filedata_dat[i].header["Current avg. (A) end"]))/2*1E12
                except KeyError:
                    cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
                cur2.append(cur)
                data1=np.subtract(despike_multi_R(i), bg)
                wl1=filedata_dat[i].signals['Wavelength (nm)']
            else:
                lr2=filedata_dat[i].signals['Wavelength (nm)'][0]
                rr2=filedata_dat[i].signals['Wavelength (nm)'][-1]
                data2=np.subtract(despike_multi_R(i), bg)
                wl2=filedata_dat[i].signals['Wavelength (nm)']
                lower_bound=min(wl1[0],wl1[-1],wl2[0],wl2[-1])
                upper_bound=max(wl1[0],wl1[-1],wl2[0],wl2[-1])
                wl_array=np.linspace(lower_bound,upper_bound,2000)
               # index1=np.abs(wl_array-wl1[-1]).argmin()
               # index2=np.abs(wl_array-wl2[0]).argmin()
              #  print(index1,index2, "ind1","ind2")
                f1 = interp1d( wl1, data1, kind='linear',fill_value=(0,0))
               ## print(np.min(wl1),"minr1",np.min(wl_array),"minru")
               ## print(np.max(wl2),"maxr2",np.max(wl_array),"maxru")
                d1_int=f1(wl_array[0:1000])
                d1=np.array(d1_int)
                f2 = interp1d( wl2, data2, kind='linear')
                d2_int=f2(wl_array[1000:2000])
                d2=np.array(d2_int)
               # d.append(d1_int)
              #  d.append(d2_int)
               # print(len(d),"len")
        
                counts_cor.append(np.concatenate((d1,d2)))
            ind+=1
            
    if BS==True:  # Load bias spectroscopy
        V=filedata_datBS[i3].signals["Bias calc (V)"]
        for i in range(i3,i4+1):
            demod_av=1E9*(filedata_datBS[i].signals["LI Demod 1 Y [bwd] (A)"]+filedata_datBS[i].signals["LI Demod 1 Y (A)"])/2 #load demod Y signal 
            cur_av=1E9*(filedata_datBS[i].signals["Current [bwd] (A)"]+filedata_datBS[i].signals["Current (A)"])/2 #load current
            dIdV.append(demod_av)
            dIdV_norm.append(demod_av/np.max(demod_av))
            cur_mod.append(cur_av)
        
    fig = plt.figure()
    # set height ratios for sublots
    if BS==True:
        gs = gridspec.GridSpec(1,4, width_ratios=[0.3,0.3,1,0.5]) 
    else:
        gs = gridspec.GridSpec(1,3, width_ratios=[0.3,0.3,1]) 
    
    ax2 = plt.subplot(gs[0])
    ax2.plot(1E-3*np.array(cur2),np.arange(0,len(cur2)),ls="None",marker=".",markersize=1.5,label="I",color="red")
    ax2.set_xlabel(r'$I_{\mathrm{avg}}$(nA)')
    ax2.set_ylabel('number')
    ax2.set_ylim([0, len(cur2)])
    ax2.set_xscale('log')
    
    ax0 = plt.subplot(gs[1])
    ax0.plot(Z2,np.arange(0,len(Z2)),ls="None",marker=".",markersize=1.5,label="Z",color="black")
    ax0.set_xlabel(r'$Z_{\mathrm{rel}}$(pm)')
  #  ax0.set_ylabel('number')
    ax0.set_yticklabels([])
    ax0.set_ylim([0, len(Z2)])
    ax1 = plt.subplot(gs[2])
    contrast=False
    #plt.imshow(counts_cor[i1:i2+1,nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)], cmap='viridis',interpolation='None',extent=[x1,x2,i1,i2+1],origin="lower")
    if contrast==True:
        data = (counts_cor - np.min(counts_cor, axis=1)[:, np.newaxis]) / np.ptp(counts_cor, axis=1)[:, np.newaxis]
    else:
        data=counts_cor
    if "tol2d" in kwargs:
        tol= float(kwargs["tol2d"])
        data=filter_2ddespike(np.array(data),tol)
    cmap=matplotlib.cm.get_cmap('seismic')
    if "gamma" in kwargs: #Load gamma for plotting default 1
        gamma=float(kwargs["gamma"])
    else:
        gamma=1
    matplotlib.colors.LinearSegmentedColormap.set_gamma(cmap, gamma)
    if unit=="nm":
        x = wl_array
        print(len(data))
        #y = np.arange(0,len(data))
        y=Z2
        X,Y = np.meshgrid(x,y)
        plt.pcolormesh(X,Y,data, cmap=cmap)
        ax1.set_xlabel('Wavelength [nm]')
    elif unit=="cm":
        x = wl_array
        lwl=632.8
        x=1E7/lwl-1E7/x
        print(len(data))
        y = np.arange(0,len(data))
       # y=Z2
        X,Y = np.meshgrid(x,y)
        plt.pcolormesh(X,Y,data, cmap=cmap)
        ax1.set_xlabel(r'Raman shift ($\mathrm{cm}^{-1}$)')
        ax1.set_yticklabels([])
     #   ax1.xaxis.set_ticks_position('bottom')
       # ax1.xaxis.set_label_position('top')
   # ax1.set_ylabel('Diagonal profile (nm)')
    plt.colorbar
    #print((eVtopix(x2,lr,rr,ncol)-eVtopix(x1,lr,rr,ncol))/(i2+1-i1))
    if BS==True:
        ax3 = plt.subplot(gs[3])
        y = np.arange(0,len(dIdV))
        x=V
        X,Y = np.meshgrid(x,y)
        if "norm" in kwargs:    
            if kwargs["norm"]==True:
                plt.pcolormesh(X,Y,dIdV_norm, cmap=cmap)
            else:
                plt.pcolormesh(X,Y,dIdV, cmap=cmap)
        else:
            plt.pcolormesh(X,Y,dIdV, cmap=cmap)
        ax3.set_xlabel('Bias (V)')
        ax3.yaxis.set_label_position("right")
        ax3.set_ylabel(r'$\mathrm{d}I/\mathrm{d}V$ map)')
        ax3.set_yticklabels([])
    fig.set_size_inches(8,4)
    
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+appendix+".png", dpi=400, bbox_inches = 'tight') # nazev souboru

def nbh(i1,j1,d1,d2,nbh):
    lr=filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i1].signals['Wavelength (nm)'][-1]
    ncol=len(filedata_dat[i1].signals['Wavelength (nm)'])-1
    nbh=int(nbh)
    coordx=[]
    coordy=[]
    coordn=[]
    for i in range (0,nbh+1):
        for j in range (0,nbh+1):
            #print(((int(nbh/2))**2+(nbh-int(nbh/2))**2))
            #print(i**2+j**2)
            if i**2+j**2<=((int(nbh/2))**2+(nbh-int(nbh/2))**2)+0.1:
                if i==0 and j==0:
                    coordx.append(i+i1)
                    coordy.append(i+j1)
                elif i==0 or j==0 :
                    coordx.append(i+i1)
                    coordx.append(-i+i1)
                    coordy.append(j+j1)
                    coordy.append(-j+j1)
                else:
                    coordx.append(i+i1)
                    coordx.append(-i+i1)
                    coordy.append(j+j1)
                    coordy.append(j+j1)
                    coordx.append(i+i1)
                    coordx.append(-i+i1)
                    coordy.append(-j+j1)
                    coordy.append(-j+j1)
    for i in range (0,len(coordx)):
        coordn.append(coordy[i]*d1+coordx[i])
    coordn.sort()
    return coordn
          
 #profile_wf_frommap("a-multimap_1.asc","a-multimap_400.asc",-2,640,660,20,20,[3,3],[3,3],[15,15],2,"nm",True,False)   
def profile_wf_frommap(i1,i2,i_bg,x1,x2,d1,d2,start,step,end,nbhv,unit,save,norm):
#def profile_wf_frommap(i1,i2,i_bg,x1,x2,d1,d2,aspect,contrast,unit,save,norm):
    cur2=[]
    Z2=[]
    scounts=[]
    bias=[]
    ncounts=[]
    sum_bg=np.zeros(len(filedata_dat[0].signals['Counts']))
    counts_cor=[]
    if isfloat(i1)==True:
        i1=i1
    else:
        for j in range (0,len(file_numbers)):
            if str(filedata_dat[j].header["Filename"])==str(i1):
                i1=j
                break
    if isfloat(i2)==True:
        i2=i2
    else:
        for j in range (0,len(file_numbers)):
            if str(filedata_dat[j].header["Filename"])==str(i2):
                i2=j
                break
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if str(filedata_dat[j].header["Filename"])==str(i_bg):
                i_bg=j
                break
    lr=filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i1].signals['Wavelength (nm)'][-1]
    ncol=len(filedata_dat[i1].signals['Wavelength (nm)'])-1
    if i_bg<-0.1 and i_bg>-1:
        n_bg=0
        d_bg=0
        for j in range(i1,i1+d1):
            if (abs(float(filedata_dat[j].header["Current avg. (A)"]))*1E12)<7:
                n_bg=float(filedata_dat[j].header["Number of Accumulations"])
                a=filedata_dat[j].signals['Counts']/n_bg
                sum_bg=np.add(a,sum_bg)
                d_bg=d_bg+1
        sum_bg=sum_bg/d_bg
    elif i_bg<-1.1:
        n_bg=0
        d_bg=0
        for j in range(i2-d1,i2):
            if (abs(float(filedata_dat[j].header["Current avg. (A)"]))*1E12)<7:
                n_bg=float(filedata_dat[j].header["Number of Accumulations"])
                a=filedata_dat[j].signals['Counts']/n_bg
                sum_bg=np.add(a,sum_bg)
                d_bg=d_bg+1
        sum_bg=sum_bg/d_bg
    else:
        n_bg=float(filedata_dat[i_bg].header["Number of Accumulations"])
        sum_bg=filedata_dat[i_bg].signals['Counts']/n_bg
        
    for i in range(i1,i2+1):
        bias.append(filedata_dat[i].header["Bias (V)"])
        n=float(filedata_dat[i].header["Number of Accumulations"])
        try:
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"])+float(filedata_dat[i].header["Current avg. (A) end"]))/2*1E12
        except KeyError:
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        cur2.append(cur)
        try:
            Z=(float(filedata_dat[i].header["Z (m)"])+float(filedata_dat[i].header["Z (m) end"]))/2*1E9
        except KeyError:
            Z=(float(filedata_dat[i].header["Z (m)"]))*1E9
        Z2.append(Z)
        scounts.append(np.subtract(filedata_dat[i].signals['Counts'], n*sum_bg))
    scounts=np.array(scounts)
    cur2=np.array(cur2)
    print(np.shape(scounts))    
    dim1,dim2=scounts.shape
    Z_map=np.reshape(Z2, (d2,d1))
    scounts_map=np.reshape(scounts, (d2,d1,dim2))
    cur_map=np.reshape(cur2, (d2,d1))
    
    fig, ax1 = plt.subplots() #osy   
    j=0
    k=0
    increment=350
    if norm==True:
        increment=increment/70
    for i in range(i1,i2+1):
        plotnow=(start[1]*d1+start[0])+k*(step[1]*d1+step[0])
        stopcycle=(end[1]*d1+end[0])
        if i>stopcycle:
            break
        if i==plotnow:
            ycord=(i-i1)//d1
            xcord=(i-i1) % d1
            coordn=nbh(xcord,ycord,d1,d2,nbhv)
            print(coordn)
            s1=np.zeros(len(filedata_dat[0].signals['Counts']))
            if norm==False:
                for i in coordn:
                    s1=np.add(scounts[i],s1)
                
                counts_cor.append(s1)
            else:
                for i in coordn:
                    s1=np.add(scounts[i]/cur2[i],s1)
                
                counts_cor.append(s1)
            ax1.plot(filedata_dat[i].signals['Wavelength (nm)'],counts_cor[-1]+j,"",color=matplotlib.cm.get_cmap('brg')((i-i1)/(i2-i1) ))
            j=j+increment
            k=k+1
    #fig, ax1 = plt.subplots()
    #plt.imshow(ar_np[i1:i2+1,1,nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)], cmap='viridis',interpolation='None',extent=[x1,x2,i1,i2+1])
    #plt.colorbar
    #fig.set_size_inches(4.4,4.4)
    if norm==False:
        ax1.set_ylabel('Photon intensity [counts]')
    else:
        ax1.set_ylabel('Photon intensity [counts/pA]')
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_xlim((x1, x2))
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax2 = ax1.twiny()
    # get the primary axis x tick locations in plot units
    x1nm=math.floor(E(x1)*10)/10
    x2nm=math.ceil(E(x2)*10)/10
    x1nm_min=math.floor(E(x1)*50)/50
    x2nm_min=math.ceil(E(x2)*50)/50             
    xtl=np.linspace(x1nm,x2nm,int(round(abs(x2nm-x1nm)*10))+1)
    xtl_min=np.linspace(x1nm_min,x2nm_min,int(round(abs(x2nm_min-x1nm_min)*50))+1)
    xtickloc=[WL(x) for x in xtl]
    xtickloc_min=[WL(x) for x in xtl_min]
    #xtickloc = ax1.get_xticks() 
    #print(xtl)
    # set the second axis ticks to the same locations
    ax2.set_xticks(xtickloc)
    ax2.set_xticks(xtickloc_min, minor=True)
    # calculate new values for the second axis tick labels, format them, and set them
    x2labels = ['{:.3g}'.format(E(x)) for x in xtickloc]
    ax2.set_xticklabels(x2labels)
    # force the bounds to be the same
    ax2.set_xlim(ax1.get_xlim()) 
    ax2.set_xlabel('Energy [eV]') 
    ax1.grid(True,linestyle=':')
    fig.set_size_inches(6/2.5,8/2.5)
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"wf_fm.png", dpi=400, bbox_inches = 'tight',transparent=True) # nazev souboru 

def profile_wf_fmman(i1,i2,i_bg,x1,x2,d1,d2,listofelem,nbhv,unit,save,norm):
#def profile_wf_frommap(i1,i2,i_bg,x1,x2,d1,d2,aspect,contrast,unit,save,norm):
    cur2=[]
    Z2=[]
    scounts=[]
    bias=[]
    ncounts=[]
    sum_bg=np.zeros(len(filedata_dat[0].signals['Counts']))
    counts_cor=[]
    if isfloat(i1)==True:
        i1=i1
    else:
        for j in range (0,len(file_numbers)):
            if str(filedata_dat[j].header["Filename"])==str(i1):
                i1=j
                break
    if isfloat(i2)==True:
        i2=i2
    else:
        for j in range (0,len(file_numbers)):
            if str(filedata_dat[j].header["Filename"])==str(i2):
                i2=j
                break
    lr=filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i1].signals['Wavelength (nm)'][-1]
    ncol=len(filedata_dat[i1].signals['Wavelength (nm)'])-1
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if str(filedata_dat[j].header["Filename"])==str(i_bg):
                i_bg=j
                break
    if i_bg<-0.1 and i_bg>-1:
        n_bg=0
        d_bg=0
        for j in range(i1,i1+d1):
            if (abs(float(filedata_dat[j].header["Current avg. (A)"]))*1E12)<7:
                n_bg=float(filedata_dat[j].header["Number of Accumulations"])
                a=filedata_dat[j].signals['Counts']/n_bg
                sum_bg=np.add(a,sum_bg)
                d_bg=d_bg+1
        sum_bg=sum_bg/d_bg
    elif i_bg<-1.1:
        n_bg=0
        d_bg=0
        for j in range(i2-d1,i2):
            if (abs(float(filedata_dat[j].header["Current avg. (A)"]))*1E12)<7:
                n_bg=float(filedata_dat[j].header["Number of Accumulations"])
                a=filedata_dat[j].signals['Counts']/n_bg
                sum_bg=np.add(a,sum_bg)
                d_bg=d_bg+1
        sum_bg=sum_bg/d_bg
    else:
        n_bg=float(filedata_dat[i_bg].header["Number of Accumulations"])
        sum_bg=filedata_dat[i_bg].signals['Counts']/n_bg
        
    for i in range(i1,i2+1):
        bias.append(filedata_dat[i].header["Bias (V)"])
        n=float(filedata_dat[i].header["Number of Accumulations"])
        try:
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"])+float(filedata_dat[i].header["Current avg. (A) end"]))/2*1E12
        except KeyError:
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        cur2.append(cur)
        try:
            Z=(float(filedata_dat[i].header["Z (m)"])+float(filedata_dat[i].header["Z (m) end"]))/2*1E9
        except KeyError:
            Z=(float(filedata_dat[i].header["Z (m)"]))*1E9
        Z2.append(Z)
        scounts.append(np.subtract(filedata_dat[i].signals['Counts'], n*sum_bg))
        
    scounts=np.array(scounts)
    cur2=np.array(cur2)
    print(np.shape(scounts))    
    dim1,dim2=scounts.shape
    Z_map=np.reshape(Z2, (d2,d1))
    scounts_map=np.reshape(scounts, (d2,d1,dim2))
    cur_map=np.reshape(cur2, (d2,d1))
    
    fig, ax1 = plt.subplots() #osy   
    j=0
    k=0
    increment=350
    if norm==True:
        increment=increment/70
    plotnow=[]
    for i in range(0,len(listofelem)):
        plotnow.append((listofelem[i][1]*d1+listofelem[i][0])+i1)
    print(plotnow,"plotnow")
    print(i1,i2," i1i2")
    print(filedata_dat[i1].header["Filename"])
    print(filedata_dat[i2].header["Filename"])
    for i in range(i1,i2+1):
        if i in plotnow:
            ycord=(i-i1)//d1
            xcord=(i-i1) % d1
            coordn=nbh(xcord,ycord,d1,d2,nbhv)
            print(coordn)
            s1=np.zeros(len(filedata_dat[0].signals['Counts']))
            if norm==False:
                for m1 in coordn:
                    s1=np.add(scounts[m1],s1)
                
                counts_cor.append(s1)
            else:
                for m1 in coordn:
                    s1=np.add(scounts[m1]/cur2[m1],s1)
                
                counts_cor.append(s1)
            ax1.plot(filedata_dat[i].signals['Wavelength (nm)'],counts_cor[-1]+j,"",color=matplotlib.cm.get_cmap('brg')((k)/(len(plotnow) )))
            j=j+increment
            k=k+1
            print(filedata_dat[i].header["Filename"])
    #fig, ax1 = plt.subplots()
    #plt.imshow(ar_np[i1:i2+1,1,nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)], cmap='viridis',interpolation='None',extent=[x1,x2,i1,i2+1])
    #plt.colorbar
    #fig.set_size_inches(4.4,4.4)
    if norm==False:
        ax1.set_ylabel('Photon intensity [counts]')
    else:
        ax1.set_ylabel('Photon intensity [counts/pA]')
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_xlim((x1, x2))
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax2 = ax1.twiny()
    # get the primary axis x tick locations in plot units
    x1nm=math.floor(E(x1)*10)/10
    x2nm=math.ceil(E(x2)*10)/10
    x1nm_min=math.floor(E(x1)*50)/50
    x2nm_min=math.ceil(E(x2)*50)/50             
    xtl=np.linspace(x1nm,x2nm,int(round(abs(x2nm-x1nm)*10))+1)
    xtl_min=np.linspace(x1nm_min,x2nm_min,int(round(abs(x2nm_min-x1nm_min)*50))+1)
    xtickloc=[WL(x) for x in xtl]
    xtickloc_min=[WL(x) for x in xtl_min]
    #xtickloc = ax1.get_xticks() 
    #print(xtl)
    # set the second axis ticks to the same locations
    ax2.set_xticks(xtickloc)
    ax2.set_xticks(xtickloc_min, minor=True)
    # calculate new values for the second axis tick labels, format them, and set them
    x2labels = ['{:.3g}'.format(E(x)) for x in xtickloc]
    ax2.set_xticklabels(x2labels)
    # force the bounds to be the same
    ax2.set_xlim(ax1.get_xlim()) 
    ax2.set_xlabel('Energy [eV]')    

    ax1.grid(True,linestyle=':')
    fig.set_size_inches(6/2.5,8/2.5)
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"wf"+str(listofelem)+".png", dpi=400, bbox_inches = 'tight',transparent=True) # nazev souboru 
            
def map_hm(i1,i2,i_bg,x1,x2,d1,d2,unit,save,norm,sigma,gamma,exponent):
    cur2=[]
    Z2=[]
    scounts=[]
    bias=[]
    ncounts=[]
    ncounts2=[]
    sum_bg=0
    if isfloat(i1)==True:
        i1=i1
    else:
        for j in range (0,len(file_numbers)):
            if str(filedata_dat[j].header["Filename"])==str(i1):
                i1=j
                break
    if isfloat(i2)==True:
        i2=i2
    else:
        for j in range (0,len(file_numbers)):
            if str(filedata_dat[j].header["Filename"])==str(i2):
                i2=j
                break
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if str(filedata_dat[j].header["Filename"])==str(i_bg):
                i_bg=j
                break
    lr=filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i1].signals['Wavelength (nm)'][-1]
    ncol=len(filedata_dat[i1].signals['Wavelength (nm)'])-1
    if i_bg<-0.1 and i_bg>-1:
        n_bg=0
        d_bg=0
        for j in range(i1,i1+d1):
            if (abs(float(filedata_dat[j].header["Current avg. (A)"]))*1E12)<7:
                n_bg=float(filedata_dat[j].header["Number of Accumulations"])
                sum_bg=sum_bg+np.sum(filedata_dat[j].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])#/n_bg
                d_bg=d_bg+1
        print(d_bg)
        sum_bg=sum_bg/d_bg
    if i_bg<-1.1 and i_bg>-2.01:
        n_bg=0
        d_bg=0
        for j in range(i2-d1,i2):
            if (abs(float(filedata_dat[j].header["Current avg. (A)"]))*1E12)<7:
                n_bg=float(filedata_dat[j].header["Number of Accumulations"])
                sum_bg=sum_bg+np.sum(filedata_dat[j].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])#/n_bg
                d_bg=d_bg+1
        print(d_bg)
        sum_bg=sum_bg/d_bg
    elif i_bg<-2.1:
        n_bg=0
        d_bg=0
        for j in range(i1,i2):
            if (abs(float(filedata_dat[j].header["Current avg. (A)"]))*1E12)<2:
                n_bg=float(filedata_dat[j].header["Number of Accumulations"])
                sum_bg=sum_bg+np.sum(filedata_dat[j].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])#/n_bg
                d_bg=d_bg+1
        print(d_bg)
        sum_bg=sum_bg/d_bg
    else:
        n_bg=float(filedata_dat[i_bg].header["Number of Accumulations"])
        sum_bg=np.sum(filedata_dat[i_bg].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])#/n_bg
        
    for i in range(i1,i2+1):
        bias.append(filedata_dat[i].header["Bias (V)"])
        n=float(filedata_dat[i].header["Number of Accumulations"])
        try:
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"])+float(filedata_dat[i].header["Current avg. (A) end"]))/2*1E12
        except KeyError:
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        cur2.append(cur)
        try:
            Z=(float(filedata_dat[i].header["Z (m)"])+float(filedata_dat[i].header["Z (m) end"]))/2*1E9
        except KeyError:
            Z=(float(filedata_dat[i].header["Z (m)"]))*1E9
        Z2.append(Z)
        if norm==True:
            scounts.append((sum(filedata_dat[i].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])-sum_bg)/cur)
        else:
            scounts.append((sum(filedata_dat[i].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])-sum_bg))
    scounts=np.array(scounts)
    print(np.shape(scounts))    
    Z_map=np.reshape(Z2, (d2,d1))
    scounts_map=np.reshape(scounts, (d2,d1))
    cur_map=np.reshape(cur2, (d2,d1))
    
    scounts_map=filter_image(scounts_map,sigma)
    amin=np.amin(scounts_map)
    amax=np.amax(scounts_map)
    scounts_map=scounts_map-amin
    
    scounts=np.reshape(scounts_map, (len(scounts)))
    print(np.shape(scounts_map))
    
    for i in range(0,len(scounts)):
        if cur2[i]>20:
            ncounts.append(scounts[i]/cur2[i]**exponent)
        else:
            ncounts.append(0)
    norm_map=np.reshape(ncounts, (d2,d1))   
    #norm_map=norm_map**gamma
    #fig, ax1 = plt.subplots()
    #plt.imshow(ar_np[i1:i2+1,1,nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)], cmap='viridis',interpolation='None',extent=[x1,x2,i1,i2+1])
    #plt.colorbar
    #fig.set_size_inches(4.4,4.4)
    
    fig, ax1 = plt.subplots()
    plt.imshow(scounts_map, cmap='afmhot',interpolation='None',extent=[0,d1,0,d2], origin="lower")
    #fig.colorbar(figure, fraction=0.046, pad=0.04)
    fig.subplots_adjust(0,0,1,1)
    ax1.set_axis_off()
    fig.set_size_inches(4.4,4.4*d2/d1)
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+str(x1)+"-"+str(x2)+"-B"+str("{0:.2f}".format(float(bias[0])))+".png", dpi=400) # nazev souboru 
            
    fig, ax1 = plt.subplots()
    plt.imshow(cur_map, cmap=cmap_cold,interpolation='None',extent=[0,d1,0,d2], origin="lower")
    #fig.colorbar(figure, fraction=0.046, pad=0.04)
    fig.subplots_adjust(0,0,1,1)
    ax1.set_axis_off()
    fig.set_size_inches(4.4,4.4*d2/d1)
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"current"+"-B"+str("{0:.2f}".format(float(bias[0])))+".png", dpi=400) # nazev souboru 
    fig, ax1 = plt.subplots()
    cmap_hot=matplotlib.cm.get_cmap('afmhot')
    matplotlib.colors.LinearSegmentedColormap.set_gamma(cmap_hot, gamma)
    figure=plt.imshow(norm_map, cmap=cmap_hot,interpolation='None',extent=[0,d1,0,d2], origin="lower")
    #fig.colorbar(figure, fraction=0.046, pad=0.04)
    fig.subplots_adjust(0,0,1,1)
    ax1.set_axis_off()
    fig.set_size_inches(4.4,4.4*d2/d1)
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+str(x1)+"-"+str(x2)+"norm"+str(exponent)+".png", dpi=400) # nazev souboru
    if abs(np.amax(Z_map)-np.amin(Z_map))>0.1:
        fig, ax1 = plt.subplots()
        figure=plt.imshow(Z_map, cmap=cmap_cold,interpolation='None',extent=[0,d1,0,d2], origin="lower")
        fig.colorbar(figure, fraction=0.046, pad=0.04)
        fig.subplots_adjust(0,0,1,1)
        ax1.set_axis_off()
        fig.set_size_inches(4.4,4.4*d2/d1)
        if save==True:
                plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+str(x1)+"-"+str(x2)+"topo.png", dpi=400) # nazev souboru

def map_hmsum(i1,i2,i_bg,x1,x2,y1,y2,d1,d2,unit,save,norm,sigma,gamma,exponent,ratio):
    cur2=[]
    Z2=[]
    scounts=[]
    scounts2=[]
    bias=[]
    ncounts=[]
    ncounts2=[]
    sum_bg=0
    sum_bg2=0
    if isfloat(i1)==True:
        i1=i1
    else:
        for j in range (0,len(file_numbers)):
            if str(filedata_dat[j].header["Filename"])==str(i1):
                i1=j
                break
    if isfloat(i2)==True:
        i2=i2
    else:
        for j in range (0,len(file_numbers)):
            if str(filedata_dat[j].header["Filename"])==str(i2):
                i2=j
                break
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if str(filedata_dat[j].header["Filename"])==str(i_bg):
                i_bg=j
                break
    lr=filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i1].signals['Wavelength (nm)'][-1]
    ncol=len(filedata_dat[i1].signals['Wavelength (nm)'])-1
    if i_bg<-0.1 and i_bg>-1:
        n_bg=0
        d_bg=0
        for j in range(i1,i1+d1):
            if (abs(float(filedata_dat[j].header["Current avg. (A)"]))*1E12)<7:
                n_bg=float(filedata_dat[j].header["Number of Accumulations"])
                sum_bg=sum_bg+np.sum(filedata_dat[j].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])#/n_bg
                sum_bg2=sum_bg2+np.sum(filedata_dat[j].signals['Counts'][nmtopix(y1,lr,rr,ncol):nmtopix(y2,lr,rr,ncol)])#/n_bg
                d_bg=d_bg+1
        sum_bg=sum_bg/d_bg
        sum_bg2=sum_bg2/d_bg
    elif i_bg<-1.1:
        n_bg=0
        d_bg=0
        for j in range(i2-d1,i2):
            if (abs(float(filedata_dat[j].header["Current avg. (A)"]))*1E12)<7:
                n_bg=float(filedata_dat[j].header["Number of Accumulations"])
                sum_bg=sum_bg+np.sum(filedata_dat[j].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])#/n_bg
                sum_bg2=sum_bg2+np.sum(filedata_dat[j].signals['Counts'][nmtopix(y1,lr,rr,ncol):nmtopix(y2,lr,rr,ncol)])#/n_bg
                d_bg=d_bg+1
        sum_bg=sum_bg/d_bg
        sum_bg2=sum_bg2/d_bg
    else:
        n_bg=float(filedata_dat[i_bg].header["Number of Accumulations"])
        sum_bg=np.sum(filedata_dat[i_bg].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])#/n_bg
        sum_bg2=np.sum(filedata_dat[i_bg].signals['Counts'][nmtopix(y1,lr,rr,ncol):nmtopix(y2,lr,rr,ncol)])#/n_bg
        
    for i in range(i1,i2+1):
        bias.append(filedata_dat[i].header["Bias (V)"])
        n=float(filedata_dat[i].header["Number of Accumulations"])
        try:
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"])+float(filedata_dat[i].header["Current avg. (A) end"]))/2*1E12
        except KeyError:
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        cur2.append(cur)
        try:
            Z=(float(filedata_dat[i].header["Z (m)"])+float(filedata_dat[i].header["Z (m) end"]))/2*1E9
        except KeyError:
            Z=(float(filedata_dat[i].header["Z (m)"]))*1E9
        Z2.append(Z)
        if norm==True:
            scounts.append((sum(filedata_dat[i].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])-sum_bg)/cur)
            scounts2.append((sum(filedata_dat[i].signals['Counts'][nmtopix(y1,lr,rr,ncol):nmtopix(y2,lr,rr,ncol)])-sum_bg2)/cur)
        else:
            scounts.append((sum(filedata_dat[i].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])-sum_bg))
            scounts2.append((sum(filedata_dat[i].signals['Counts'][nmtopix(y1,lr,rr,ncol):nmtopix(y2,lr,rr,ncol)])-sum_bg2))
    scounts=np.array(scounts)
    scounts2=np.array(scounts2)
    print(np.shape(scounts))    
    Z_map=np.reshape(Z2, (d2,d1))
    scounts_map=np.reshape(scounts, (d2,d1))
    scounts_map2=np.reshape(scounts2, (d2,d1))
    cur_map=np.reshape(cur2, (d2,d1))
    
    scounts_map=filter_image(scounts_map,sigma)
    scounts_map2=filter_image(scounts_map2,sigma)
    amin=np.amin(scounts_map)
    amax=np.amax(scounts_map)
    scounts_map=scounts_map-amin
    amin2=np.amin(scounts_map2)
    amax2=np.amax(scounts_map2)
    scounts_map2=scounts_map2-amin
    
    scounts=np.reshape(scounts_map, (len(scounts)))
    scounts2=np.reshape(scounts_map2, (len(scounts2)))
    print(np.shape(scounts_map))
    
    for i in range(0,len(scounts)):
            ncounts2.append(scounts[i]+ratio*scounts2[i])
    scounts_map_pd=np.reshape(ncounts2, (d2,d1))  
    #norm_map=norm_map**gamma
    #fig, ax1 = plt.subplots()
    #plt.imshow(ar_np[i1:i2+1,1,nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)], cmap='viridis',interpolation='None',extent=[x1,x2,i1,i2+1])
    #plt.colorbar
    #fig.set_size_inches(4.4,4.4)
    fig, ax1 = plt.subplots()
    figure=plt.imshow(scounts_map_pd, cmap='afmhot',interpolation='None',extent=[0,d1,0,d2], origin="lower")
    fig.colorbar(figure, fraction=0.046, pad=0.04)
    fig.subplots_adjust(0,0,1,1)
    ax1.set_axis_off()
    fig.set_size_inches(4.4,4.4*d2/d1)
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+str(x1)+"-"+str(x2)+"sum"+str(y1)+"-"+str(y2)+"ratio"+str("{0:.2f}".format(ratio))+".png", dpi=400) # nazev souboru 
     
def map_hmfit(i1,i2,i_bg,x1,x2,d1,d2,unit,save,norm,sigma,gamma,treshold,vmin,vmax):
    #example map_hmfit("LS-CHmap-a00001", "LS-CHmap-a01024", 309, 730, 735, 32, 32, "nm", True, ,False,0, 1, 20,300,1000)
    cur2=[]
    Z2=[]
    scounts=[]
    bias=[]
    ncounts=[]
    ncounts2=[]
    sum_bg=0
    mu=[]
    mu2=[]
    sigma=[]
    exponent=1
    #p0=[1,648,5,0.3,655,5]
    p0=[1,(x1+x2)/2,5,0]
    if isfloat(i1)==True:
        i1=i1
    else:
        for j in range (0,len(file_numbers)):
            if str(filedata_dat[j].header["Filename"])==str(i1):
                i1=j
                break
    if isfloat(i2)==True:
        i2=i2
    else:
        for j in range (0,len(file_numbers)):
            if str(filedata_dat[j].header["Filename"])==str(i2):
                i2=j
                break
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if str(filedata_dat[j].header["Filename"])==str(i_bg):
                i_bg=j
                break
    print(i1)
    lr=filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i1].signals['Wavelength (nm)'][-1]
    ncol=len(filedata_dat[i1].signals['Wavelength (nm)'])-1
    if i_bg<-0.1 and i_bg>-1:
        n_bg=0
        d_bg=0
        for j in range(i1,i1+d1):
            if (abs(float(filedata_dat[j].header["Current avg. (A)"]))*1E12)<7:
                n_bg=float(filedata_dat[j].header["Number of Accumulations"])
                sum_bg=sum_bg+np.sum(filedata_dat[j].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])#/n_bg
                d_bg=d_bg+1
        print(d_bg)
        sum_bg=sum_bg/d_bg
    if i_bg<-1.1 and i_bg>-2.01:
        n_bg=0
        d_bg=0
        for j in range(i2-d1,i2):
            if (abs(float(filedata_dat[j].header["Current avg. (A)"]))*1E12)<7:
                n_bg=float(filedata_dat[j].header["Number of Accumulations"])
                sum_bg=sum_bg+np.sum(filedata_dat[j].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])#/n_bg
                d_bg=d_bg+1
        print(d_bg)
        sum_bg=sum_bg/d_bg
    elif i_bg<-2.1:
        n_bg=0
        d_bg=0
        for j in range(i1,i2):
            if (abs(float(filedata_dat[j].header["Current avg. (A)"]))*1E12)<2:
                n_bg=float(filedata_dat[j].header["Number of Accumulations"])
                sum_bg=sum_bg+np.sum(filedata_dat[j].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])#/n_bg
                d_bg=d_bg+1
        print(d_bg)
        sum_bg=sum_bg/d_bg
    else:
       # n_bg=float(filedata_dat[i_bg].header["Number of Accumulations"])
       # sum_bg=np.sum(filedata_dat[i_bg].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])#/n_bg
       sum_bg=i_bg 
    for i in range(i1,i2+1):
        bias.append(filedata_dat[i].header["Bias (V)"])
        n=float(filedata_dat[i].header["Number of Accumulations"])
        try:
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"])+float(filedata_dat[i].header["Current avg. (A) end"]))/2*1E12
        except KeyError:
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        cur2.append(cur)
        try:
            Z=(float(filedata_dat[i].header["Z (m)"])+float(filedata_dat[i].header["Z (m) end"]))/2*1E9
        except KeyError:
            Z=(float(filedata_dat[i].header["Z (m)"]))*1E9
        Z2.append(Z)
        if norm==True:
            scounts.append((sum(filedata_dat[i].signals['Counts nf'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])-sum_bg)/cur)
        else:
            scounts.append((sum(filedata_dat[i].signals['Counts nf'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])-sum_bg))
        wl=filedata_dat[i].signals['Wavelength (nm)'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)]
        
        counts_cor=(np.subtract(filedata_dat[i].signals['Counts nf'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)], 309))
        if np.max(counts_cor)>treshold:
           # fig, ax1 = plt.subplots()
           # ax1.plot(wl,counts_cor, label='Experiment')
            #coeff, var_matrix = curve_fit(gauss2, wl, counts_cor, p0=p0,bounds=[[0,-np.inf,0],[+np.inf,+np.inf,+np.inf]])
            try:
                coeff, var_matrix = curve_fit(gauss2, wl, counts_cor, p0=p0,bounds=[[-5,-np.inf,-np.inf,-10],[+np.inf,+np.inf,+np.inf,10]])
                mu.append(coeff[1])
                sigma.append(coeff[2])
             #   ax1.plot(wl,gauss(wl, coeff[0],coeff[1],coeff[2],coeff[3]), label='G1 Fit')
            except RuntimeError:
                mu.append(np.NaN)
                sigma.append(np.NaN)
                
                
            mu2.append(np.mean(wl*counts_cor)/np.mean(counts_cor))
            #mu.append(coeff[1])
            #sigma.append(coeff[2])
            #fit_res = gauss2(wl, *coeff)

          #  plt.axvline(x=mu2[-1])
           # ax1.plot(wl,gauss(wl, coeff[0],coeff[1],coeff[2]), label='G1 Fit')
            #ax1.plot(wl,gauss(wl, coeff[3],coeff[4],coeff[5]), label='G2 Fit')
            #fig.set_size_inches(4.4,4.4)
        else:
            mu.append(np.NaN)
            mu2.append(np.NaN)
            sigma.append(np.NaN)
    scounts=np.array(scounts)
    print(np.shape(scounts))    
    Z_map=np.reshape(Z2, (d2,d1))
    mu_map=np.reshape(mu, (d2,d1))
    mu2_map=np.reshape(mu2, (d2,d1))
    sigma_map=np.reshape(sigma, (d2,d1))
    scounts_map=np.reshape(scounts, (d2,d1))
    cur_map=np.reshape(cur2, (d2,d1))
    
    #scounts_map=filter_image(scounts_map,sigma)
    amin=np.amin(scounts_map)
    amax=np.amax(scounts_map)
    scounts_map=scounts_map-amin
    
    scounts=np.reshape(scounts_map, (len(scounts)))
    print(np.shape(scounts_map))
    
    for i in range(0,len(scounts)):
        if cur2[i]>8:
            ncounts.append(scounts[i]/cur2[i]**exponent)
        else:
            ncounts.append(0)
    norm_map=np.reshape(ncounts, (d2,d1))  
    
    if x1<750:
        offset=np.array(mu)-0.01889*np.array(cur2)-646.92
    else:
        offset=np.array(mu)-0.0216*np.array(cur2)-814.836
    offset_map=np.reshape(offset, (d2,d1))
    #norm_map=norm_map**gamma
    #fig, ax1 = plt.subplots()
    #plt.imshow(ar_np[i1:i2+1,1,nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)], cmap='viridis',interpolation='None',extent=[x1,x2,i1,i2+1])
    #plt.colorbar
    #fig.set_size_inches(4.4,4.4)
    
    fig, ax1 = plt.subplots()
    plt.imshow(scounts_map, cmap='afmhot',interpolation='None',extent=[0,d1,0,d2], origin="lower")
    #fig.colorbar(figure, fraction=0.046, pad=0.04)
    fig.subplots_adjust(0,0,1,1)
    ax1.set_axis_off()
    fig.set_size_inches(4.4,4.4*d2/d1)
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+str(x1)+"-"+str(x2)+"-B"+str("{0:.2f}".format(float(bias[0])))+".png", dpi=400) # nazev souboru 
            
    fig, ax1 = plt.subplots()
    cmap_hot=matplotlib.cm.get_cmap('coolwarm')
    matplotlib.colors.LinearSegmentedColormap.set_gamma(cmap_hot, gamma)
    figure=plt.imshow(mu_map, cmap=cmap_hot,interpolation='None',extent=[0,d1,0,d2], origin="lower",vmin=np.min(mu),vmax=np.max(mu))
    fig.colorbar(figure, fraction=0.046, pad=0.04,label="peak redshift (nm)")
    fig.subplots_adjust(0,0,1,1)
    #ax1.set_axis_off()
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    fig.set_size_inches(4.4,4.4*d2/d1)
    if save==True:
        plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+str(x1)+"-"+str(x2)+"peak_pos(nm)"+"-B"+str("{0:.2f}".format(float(bias[0])))+".png", dpi=400,bbox_inches='tight',transparent=True) # nazev souboru
    
    fig, ax1 = plt.subplots()
    #cmap_hot=matplotlib.cm.get_cmap('viridis')
    matplotlib.colors.LinearSegmentedColormap.set_gamma(cmap_hot, gamma)
    figure=plt.imshow(mu2_map, cmap=cmap_hot,interpolation='None',extent=[0,d1,0,d2], origin="lower",vmin=vmin,vmax=vmax)
    fig.colorbar(figure, fraction=0.046, pad=0.04)
    fig.subplots_adjust(0,0,1,1)
    #ax1.set_axis_off()
    fig.set_size_inches(4.4,4.4*d2/d1)
    if save==True:
        plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+str(x1)+"-"+str(x2)+"peak_pos2(nm)"+"-B"+str("{0:.2f}".format(float(bias[0])))+".png", dpi=400,bbox_inches='tight') # nazev souboru
    
    fig, ax1 = plt.subplots()
    cmap_hot=matplotlib.cm.get_cmap('Reds')
    matplotlib.colors.LinearSegmentedColormap.set_gamma(cmap_hot, gamma)
    figure=plt.imshow(sigma_map, cmap=cmap_hot,interpolation='None',extent=[0,d1,0,d2], origin="lower")
    fig.colorbar(figure, fraction=0.046, pad=0.04)
    fig.subplots_adjust(0,0,1,1)
    #ax1.set_axis_off()
    fig.set_size_inches(4.4,4.4*d2/d1)
    if save==True:
        plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+str(x1)+"-"+str(x2)+"peak_FWHM(nm)"+"-B"+str("{0:.2f}".format(float(bias[0])))+".png", dpi=400,bbox_inches='tight') # nazev souboru
        
    fig, ax1 = plt.subplots()
    cmap_hot=matplotlib.cm.get_cmap('coolwarm')
    matplotlib.colors.LinearSegmentedColormap.set_gamma(cmap_hot, gamma)
    figure=plt.imshow(offset_map, cmap=cmap_hot,interpolation='None',extent=[0,d1,0,d2], origin="lower")
    fig.colorbar(figure, fraction=0.046, pad=0.04)
    fig.subplots_adjust(0,0,1,1)
    #ax1.set_axis_off()
    fig.set_size_inches(4.4,4.4*d2/d1)
    if save==True:
        plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+str(x1)+"-"+str(x2)+"offset"+"-B"+str("{0:.2f}".format(float(bias[0])))+".png", dpi=400,bbox_inches='tight') # nazev souboru
    
    if abs(np.amax(Z_map)-np.amin(Z_map))>0.1:
        fig, ax1 = plt.subplots()
        figure=plt.imshow(Z_map, cmap=cmap_cold,interpolation='None',extent=[0,d1,0,d2], origin="lower")
        fig.colorbar(figure, fraction=0.046, pad=0.04)
        fig.subplots_adjust(0,0,1,1)
        ax1.set_axis_off()
        fig.set_size_inches(4.4,4.4*d2/d1)
        if save==True:
                plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+str(x1)+"-"+str(x2)+"topo.png", dpi=400)
                # nazev souboru
    fig, ax1 = plt.subplots() #osy 
    ax1.set_xlabel('Current [pA]')
    #ax1.set_ylabel('Peak position [nm]')
    ax1.set_ylabel('Peak position [nm]')
    #plt.xlim(xmin=0)
    ax1.plot(cur2,np.array(mu), label=None,marker='x',linestyle="None")
    #cax=ax1.scatter(cur2,np.array(mu), c=offset, cmap=cm.coolwarm, label="exciton"+str(x1)+"-"+str(x2)+" nm Bias "+str(filedata_dat[i1].header["Bias (V)"]),marker='x',linestyle="None")
    
    #ax1.plot(cur2,ncounts2, label=str(filedata_dat[i1].header["Filename"])+" "+str(filedata_dat[i2].header["Filename"])+" "+"B "+str(filedata_dat[i1].header["Bias (V)"]),marker='x',linestyle="None")
    plt.legend()
    #ax1.grid(True,linestyle=':')
    fig.set_size_inches(2.5,2.5)
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+str(x1)+"-"+str(x2)+"peak-cur.png", dpi=400,bbox_inches='tight',transparent=True) # nazev souboru 
    

def map_hmfit_CO(i1,i2,i_bg,x1,x2,unit,save,norm,sigma,gamma,treshold,vmin,vmax):
    #example map_hmfit("LS-CHmap-a00001", "LS-CHmap-a01024", 309, 730, 735, "nm", True, ,False,0, 1, 20,300,1000)
    cur2=[]
    Z2=[]
    scounts=[]
    bias=[]
    ncounts=[]
    ncounts2=[]
    sum_bg=0
    mu=[]
    mu2=[]
    sigma=[]
    exponent=1
    #p0=[1,648,5,0.3,655,5]
    p0=[5,733.7,0.1,0]
    if isfloat(i1)==True:
        i1=i1
    else:
        for j in range (0,len(file_numbers)):
            if str(filedata_dat[j].header["Filename"])==str(i1):
                i1=j
                break
    if isfloat(i2)==True:
        i2=i2
    else:
        for j in range (0,len(file_numbers)):
            if str(filedata_dat[j].header["Filename"])==str(i2):
                i2=j
                break
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if str(filedata_dat[j].header["Filename"])==str(i_bg):
                i_bg=j
                break
    print(i1)
    lr=filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i1].signals['Wavelength (nm)'][-1]
    ncol=len(filedata_dat[i1].signals['Wavelength (nm)'])-1
    i=0
    while abs(abs(float(filedata_dat[i1+i+1].header["X (m)"])-float(filedata_dat[i1+i].header["X (m)"]))-abs(float(filedata_dat[i1+1].header["X (m)"])-float(filedata_dat[i1].header["X (m)"])))<1E-10:
        i += 1
        if i>(i2-i1):
            break
    else:
        d1=i+1
        d2=int((i2+1-i1)/d1)
        print(d1)
        print(d2)
    
    if i_bg<-0.1 and i_bg>-1:
        n_bg=0
        d_bg=0
        for j in range(i1,i1+d1):
            if (abs(float(filedata_dat[j].header["Current avg. (A)"]))*1E12)<7:
                n_bg=float(filedata_dat[j].header["Number of Accumulations"])
                sum_bg=sum_bg+np.sum(filedata_dat[j].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])#/n_bg
                d_bg=d_bg+1
        print(d_bg)
        sum_bg=sum_bg/d_bg
    if i_bg<-1.1 and i_bg>-2.01:
        n_bg=0
        d_bg=0
        for j in range(i2-d1,i2):
            if (abs(float(filedata_dat[j].header["Current avg. (A)"]))*1E12)<7:
                n_bg=float(filedata_dat[j].header["Number of Accumulations"])
                sum_bg=sum_bg+np.sum(filedata_dat[j].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])#/n_bg
                d_bg=d_bg+1
        print(d_bg)
        sum_bg=sum_bg/d_bg
    elif i_bg<-2.1:
        n_bg=0
        d_bg=0
        for j in range(i1,i2):
            if (abs(float(filedata_dat[j].header["Current avg. (A)"]))*1E12)<2:
                n_bg=float(filedata_dat[j].header["Number of Accumulations"])
                sum_bg=sum_bg+np.sum(filedata_dat[j].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])#/n_bg
                d_bg=d_bg+1
        print(d_bg)
        sum_bg=sum_bg/d_bg
    else:
       # n_bg=float(filedata_dat[i_bg].header["Number of Accumulations"])
       # sum_bg=np.sum(filedata_dat[i_bg].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])#/n_bg
       sum_bg=i_bg 
    for i in range(i1,i2+1):
   # for i in range(i1,i1+10):
        bias.append(filedata_dat[i].header["Bias (V)"])
        n=float(filedata_dat[i].header["Number of Accumulations"])
        try:
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"])+float(filedata_dat[i].header["Current avg. (A) end"]))/2*1E12
        except KeyError:
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        cur2.append(cur)
        try:
            Z=(float(filedata_dat[i].header["Z (m)"])+float(filedata_dat[i].header["Z (m) end"]))/2*1E9
        except KeyError:
            Z=(float(filedata_dat[i].header["Z (m)"]))*1E9
        Z2.append(Z)
        if norm==True:
            scounts.append((sum(filedata_dat[i].signals['Counts nf'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])-sum_bg)/cur)
        else:
            scounts.append((sum(filedata_dat[i].signals['Counts nf'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])-sum_bg))
        wl=filedata_dat[i].signals['Wavelength (nm)'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)]
        
        counts_cor=(np.subtract(filedata_dat[i].signals['Counts nf'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)], 309))
        x1fit=nmtopix(730,lr,rr,ncol)
        x2fit=nmtopix(735,lr,rr,ncol)
        if np.max(counts_cor)>treshold:
           # fig, ax1 = plt.subplots()
           # ax1.plot(wl,counts_cor, label='Experiment')
            #coeff, var_matrix = curve_fit(gauss2, wl, counts_cor, p0=p0,bounds=[[0,-np.inf,0],[+np.inf,+np.inf,+np.inf]])
            try:
                coeff, var_matrix = curve_fit(gauss2, wl, counts_cor, p0=p0)
                mu.append(coeff[1])
               # print(nmtopix(730,lr,rr,ncol),nmtopix(735,lr,rr,ncol))
                print(coeff,"p")
                print(coeff[1],"mu")
                sigma.append(coeff[2])
                print(coeff[2],"sigma")
           #     ax1.plot(wl,gauss(wl, coeff[0],coeff[1],coeff[2],coeff[3]), label='G1 Fit')
            except RuntimeError:
                mu.append(np.NaN)
                sigma.append(np.NaN)
                
                
            mu2.append(np.mean(wl*counts_cor)/np.mean(counts_cor))
            #mu.append(coeff[1])
            #sigma.append(coeff[2])
            #fit_res = gauss2(wl, *coeff)

          #  plt.axvline(x=mu2[-1])
           # ax1.plot(wl,gauss(wl, coeff[0],coeff[1],coeff[2]), label='G1 Fit')
            #ax1.plot(wl,gauss(wl, coeff[3],coeff[4],coeff[5]), label='G2 Fit')
         #  fig.set_size_inches(4.4,4.4)
        else:
            mu.append(np.NaN)
            mu2.append(np.NaN)
            sigma.append(np.NaN)
    scounts=np.array(scounts)
    print(np.shape(scounts))    
    #in cm-1
    lwl=632.8
    Z_map=np.reshape(Z2, (d2,d1))
    mu_map=1E7/lwl-1E7/np.reshape(mu, (d2,d1))
    mu2_map=1E7/lwl-1E7/np.reshape(mu2, (d2,d1))
    sigma_map=np.reshape(sigma, (d2,d1))
    scounts_map=np.reshape(scounts, (d2,d1))
    cur_map=np.reshape(cur2, (d2,d1))
    
    #scounts_map=filter_image(scounts_map,sigma)
    amin=np.amin(scounts_map)
    amax=np.amax(scounts_map)
    scounts_map=scounts_map-amin
    
    scounts=np.reshape(scounts_map, (len(scounts)))
    print(np.shape(scounts_map))
    
    for i in range(0,len(scounts)):
        if cur2[i]>8:
            ncounts.append(scounts[i]/cur2[i]**exponent)
        else:
            ncounts.append(0)
    norm_map=np.reshape(ncounts, (d2,d1))  
    
    if x1<750:
        offset=np.array(mu)-0.01889*np.array(cur2)-646.92
    else:
        offset=np.array(mu)-0.0216*np.array(cur2)-814.836
    offset_map=np.reshape(offset, (d2,d1))
    #norm_map=norm_map**gamma
    #fig, ax1 = plt.subplots()
    #plt.imshow(ar_np[i1:i2+1,1,nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)], cmap='viridis',interpolation='None',extent=[x1,x2,i1,i2+1])
    #plt.colorbar
    #fig.set_size_inches(4.4,4.4)
    
    fig, ax1 = plt.subplots()
    plt.imshow(scounts_map, cmap='afmhot',interpolation='None',extent=[0,d1,0,d2], origin="lower")
    #fig.colorbar(figure, fraction=0.046, pad=0.04)
    fig.subplots_adjust(0,0,1,1)
    ax1.set_axis_off()
    fig.set_size_inches(4.4,4.4*d2/d1)
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+str(x1)+"-"+str(x2)+"-B"+str("{0:.2f}".format(float(bias[0])))+".png", dpi=400) # nazev souboru 
            
    fig, ax1 = plt.subplots() #peak position
    cmap_hot=matplotlib.cm.get_cmap('coolwarm')
    matplotlib.colors.LinearSegmentedColormap.set_gamma(cmap_hot, gamma)
    figure=plt.imshow(mu_map, cmap=cmap_hot,interpolation='None',extent=[0,d1,0,d2], origin="lower")
    fig.colorbar(figure, fraction=0.046, pad=0.04,label=r"peak redshift gauss fit ($\mathrm{cm}^1$)")
    fig.subplots_adjust(0,0,1,1)
    #ax1.set_axis_off()
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    fig.set_size_inches(4.4,4.4*d2/d1)
    if save==True:
        plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+str(x1)+"-"+str(x2)+"peak_pos(nm)"+"-B"+str("{0:.2f}".format(float(bias[0])))+".png", dpi=400,bbox_inches='tight',transparent=True) # nazev souboru
    
    fig, ax1 = plt.subplots() #peak position 2
    #cmap_hot=matplotlib.cm.get_cmap('viridis')
    matplotlib.colors.LinearSegmentedColormap.set_gamma(cmap_hot, gamma)
    figure=plt.imshow(mu2_map, cmap=cmap_hot,interpolation='None',extent=[0,d1,0,d2], origin="lower")
    fig.colorbar(figure, fraction=0.046, pad=0.04,label=r"peak redshift mean ($\mathrm{cm}^1$)")
    fig.subplots_adjust(0,0,1,1)
    #ax1.set_axis_off()
    fig.set_size_inches(4.4,4.4*d2/d1)
    if save==True:
        plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+str(x1)+"-"+str(x2)+"peak_pos2(nm)"+"-B"+str("{0:.2f}".format(float(bias[0])))+".png", dpi=400,bbox_inches='tight') # nazev souboru
    
    fig, ax1 = plt.subplots() #peak FWHM
    cmap_hot=matplotlib.cm.get_cmap('Reds')
    matplotlib.colors.LinearSegmentedColormap.set_gamma(cmap_hot, gamma)
    figure=plt.imshow(sigma_map, cmap=cmap_hot,interpolation='None',extent=[0,d1,0,d2], origin="lower")
    fig.colorbar(figure, fraction=0.046, pad=0.04)
    fig.subplots_adjust(0,0,1,1)
    #ax1.set_axis_off()
    fig.set_size_inches(4.4,4.4*d2/d1)
    if save==True:
        plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+str(x1)+"-"+str(x2)+"peak_FWHM(nm)"+"-B"+str("{0:.2f}".format(float(bias[0])))+".png", dpi=400,bbox_inches='tight') # nazev souboru
        
    fig, ax1 = plt.subplots()
    cmap_hot=matplotlib.cm.get_cmap('coolwarm')
    matplotlib.colors.LinearSegmentedColormap.set_gamma(cmap_hot, gamma)
    figure=plt.imshow(offset_map, cmap=cmap_hot,interpolation='None',extent=[0,d1,0,d2], origin="lower")
    fig.colorbar(figure, fraction=0.046, pad=0.04)
    fig.subplots_adjust(0,0,1,1)
    #ax1.set_axis_off()
    fig.set_size_inches(4.4,4.4*d2/d1)
    if save==True:
        plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+str(x1)+"-"+str(x2)+"offset"+"-B"+str("{0:.2f}".format(float(bias[0])))+".png", dpi=400,bbox_inches='tight') # nazev souboru
    
    if abs(np.amax(Z_map)-np.amin(Z_map))>0.1:
        fig, ax1 = plt.subplots()
        figure=plt.imshow(Z_map, cmap=cmap_cold,interpolation='None',extent=[0,d1,0,d2], origin="lower")
        fig.colorbar(figure, fraction=0.046, pad=0.04)
        fig.subplots_adjust(0,0,1,1)
        ax1.set_axis_off()
        fig.set_size_inches(4.4,4.4*d2/d1)
        if save==True:
                plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+str(x1)+"-"+str(x2)+"topo.png", dpi=400)
                # nazev souboru
    else:
        fig, ax1 = plt.subplots()
        figure=plt.imshow(cur_map, cmap=cmap_cold,interpolation='None',extent=[0,d1,0,d2], origin="lower")
        fig.colorbar(figure, fraction=0.046, pad=0.04,label="current(pA)")
        fig.subplots_adjust(0,0,1,1)
        ax1.set_axis_off()
        fig.set_size_inches(4.4,4.4*d2/d1)
        if save==True:
                plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+str(x1)+"-"+str(x2)+"topo.png", dpi=400)
                # nazev souboru
    fig, ax1 = plt.subplots() #osy 
    ax1.set_xlabel('Current [pA]')
    #ax1.set_ylabel('Peak position [nm]')
    ax1.set_ylabel('Peak position [nm]')
    #plt.xlim(xmin=0)
    ax1.plot(cur2,np.array(mu), label=None,marker='x',linestyle="None")
    #cax=ax1.scatter(cur2,np.array(mu), c=offset, cmap=cm.coolwarm, label="exciton"+str(x1)+"-"+str(x2)+" nm Bias "+str(filedata_dat[i1].header["Bias (V)"]),marker='x',linestyle="None")
    
    #ax1.plot(cur2,ncounts2, label=str(filedata_dat[i1].header["Filename"])+" "+str(filedata_dat[i2].header["Filename"])+" "+"B "+str(filedata_dat[i1].header["Bias (V)"]),marker='x',linestyle="None")
    plt.legend()
    #ax1.grid(True,linestyle=':')
    fig.set_size_inches(2.5,2.5)
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+str(x1)+"-"+str(x2)+"peak-cur.png", dpi=400,bbox_inches='tight',transparent=True) # nazev souboru 
    
    
def map_hm_plasm(i1,i2,i_bg,x1,x2,y1,y2,d1,d2,unit,save,norm,sigma,gamma,exponent,tre):
    cur2=[]
    Z2=[]
    scounts=[]
    scounts2=[]
    bias=[]
    ncounts=[]
    ncounts2=[]
    sum_bg=0
    sum_bg2=0
    if isfloat(i1)==True:
        i1=i1
    else:
        for j in range (0,len(file_numbers)):
            if str(filedata_dat[j].header["Filename"])==str(i1):
                i1=j
                break
    if isfloat(i2)==True:
        i2=i2
    else:
        for j in range (0,len(file_numbers)):
            if str(filedata_dat[j].header["Filename"])==str(i2):
                i2=j
                break
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if str(filedata_dat[j].header["Filename"])==str(i_bg):
                i_bg=j
                break
    lr=filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i1].signals['Wavelength (nm)'][-1]
    ncol=len(filedata_dat[i1].signals['Wavelength (nm)'])-1
    if i_bg<-0.1 and i_bg>-1:
        n_bg=0
        d_bg=0
        for j in range(i1,i1+d1):
            if (abs(float(filedata_dat[j].header["Current avg. (A)"]))*1E12)<7:
                n_bg=float(filedata_dat[j].header["Number of Accumulations"])
                sum_bg=sum_bg+np.sum(filedata_dat[j].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])#/n_bg
                sum_bg2=sum_bg2+np.sum(filedata_dat[j].signals['Counts'][nmtopix(y1,lr,rr,ncol):nmtopix(y2,lr,rr,ncol)])#/n_bg
                d_bg=d_bg+1
        sum_bg=sum_bg/d_bg
        sum_bg2=sum_bg2/d_bg
    elif i_bg<-1.1:
        n_bg=0
        d_bg=0
        for j in range(i2-d1,i2):
            if (abs(float(filedata_dat[j].header["Current avg. (A)"]))*1E12)<7:
                n_bg=float(filedata_dat[j].header["Number of Accumulations"])
                sum_bg=sum_bg+np.sum(filedata_dat[j].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])#/n_bg
                sum_bg2=sum_bg2+np.sum(filedata_dat[j].signals['Counts'][nmtopix(y1,lr,rr,ncol):nmtopix(y2,lr,rr,ncol)])#/n_bg
                d_bg=d_bg+1
        sum_bg=sum_bg/d_bg
        sum_bg2=sum_bg2/d_bg
    else:
        n_bg=float(filedata_dat[i_bg].header["Number of Accumulations"])
        sum_bg=np.sum(filedata_dat[i_bg].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])#/n_bg
        sum_bg2=np.sum(filedata_dat[i_bg].signals['Counts'][nmtopix(y1,lr,rr,ncol):nmtopix(y2,lr,rr,ncol)])#/n_bg
        
    for i in range(i1,i2+1):
        bias.append(filedata_dat[i].header["Bias (V)"])
        n=float(filedata_dat[i].header["Number of Accumulations"])
        try:
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"])+float(filedata_dat[i].header["Current avg. (A) end"]))/2*1E12
        except KeyError:
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        cur2.append(cur)
        try:
            Z=(float(filedata_dat[i].header["Z (m)"])+float(filedata_dat[i].header["Z (m) end"]))/2*1E9
        except KeyError:
            Z=(float(filedata_dat[i].header["Z (m)"]))*1E9
        Z2.append(Z)
        if norm==True:
            scounts.append((sum(filedata_dat[i].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])-sum_bg)/cur)
            scounts2.append((sum(filedata_dat[i].signals['Counts'][nmtopix(y1,lr,rr,ncol):nmtopix(y2,lr,rr,ncol)])-sum_bg2)/cur)
        else:
            scounts.append((sum(filedata_dat[i].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])-sum_bg))
            scounts2.append((sum(filedata_dat[i].signals['Counts'][nmtopix(y1,lr,rr,ncol):nmtopix(y2,lr,rr,ncol)])-sum_bg2))
    scounts=np.array(scounts)
    scounts2=np.array(scounts2)
    print(np.shape(scounts))    
    Z_map=np.reshape(Z2, (d2,d1))
    scounts_map=np.reshape(scounts, (d2,d1))
    scounts_map2=np.reshape(scounts2, (d2,d1))
    cur_map=np.reshape(cur2, (d2,d1))
    
    scounts_map=filter_image(scounts_map,sigma)
    scounts_map2=filter_image(scounts_map2,sigma)
    amin=np.amin(scounts_map)
    amax=np.amax(scounts_map)
    scounts_map=scounts_map-amin
    amin2=np.amin(scounts_map2)
    amax2=np.amax(scounts_map2)
    scounts_map2=scounts_map2-amin
    #where=np.zeros((d1,d2))
    #where_sc2=(sc2>100)
    #where=np.copy(where_sc2)
    #scounts_map_pd=np.zeros((d1,d2))
    #scounts_map_pd=np.divide(scounts_map,scounts_map2, where=where)
    
    scounts=np.reshape(scounts_map, (len(scounts)))
    scounts2=np.reshape(scounts_map2, (len(scounts2)))
    print(np.shape(scounts_map))
    
    for i in range(0,len(scounts)):
        if cur2[i]>8:
            ncounts.append(scounts[i]/cur2[i]**exponent)
        else:
            ncounts.append(0)
    for i in range(0,len(scounts)):
        if scounts2[i]>tre:
            ncounts2.append(scounts[i]/scounts2[i])
        else:
            ncounts2.append(0)
    norm_map=np.reshape(ncounts, (d2,d1))  
    scounts_map_pd=np.reshape(ncounts2, (d2,d1))  
    #norm_map=norm_map**gamma
    #fig, ax1 = plt.subplots()
    #plt.imshow(ar_np[i1:i2+1,1,nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)], cmap='viridis',interpolation='None',extent=[x1,x2,i1,i2+1])
    #plt.colorbar
    #fig.set_size_inches(4.4,4.4)
    
    fig, ax1 = plt.subplots()
    figure=plt.imshow(scounts_map, cmap='afmhot',interpolation='None',extent=[0,d1,0,d2], origin="lower")
    fig.colorbar(figure, fraction=0.046, pad=0.04)
    fig.subplots_adjust(0,0,1,1)
    ax1.set_axis_off()
    fig.set_size_inches(4.4,4.4*d2/d1)
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+str(x1)+"-"+str(x2)+".png", dpi=400) # nazev souboru 
   
    fig, ax1 = plt.subplots()
    figure=plt.imshow(scounts_map2, cmap='afmhot',interpolation='None',extent=[0,d1,0,d2], origin="lower")
    fig.colorbar(figure, fraction=0.046, pad=0.04)
    fig.subplots_adjust(0,0,1,1)
    ax1.set_axis_off()
    fig.set_size_inches(4.4,4.4*d2/d1)
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+str(y1)+"-"+str(y2)+".png", dpi=400) # nazev souboru 
   
    fig, ax1 = plt.subplots()
    figure=plt.imshow(scounts_map_pd, cmap='afmhot',interpolation='None',extent=[0,d1,0,d2], origin="lower")
    fig.colorbar(figure, fraction=0.046, pad=0.04)
    fig.subplots_adjust(0,0,1,1)
    ax1.set_axis_off()
    fig.set_size_inches(4.4,4.4*d2/d1)
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+str(x1)+"-"+str(x2)+"div"+str(y1)+"-"+str(y2)+".png", dpi=400) # nazev souboru 
     

    fig, ax1 = plt.subplots()
    plt.imshow(cur_map, cmap=cmap_cold,interpolation='None',extent=[0,d1,0,d2], origin="lower")
    #fig.colorbar(figure, fraction=0.046, pad=0.04)
    fig.subplots_adjust(0,0,1,1)
    ax1.set_axis_off()
    fig.set_size_inches(4.4,4.4*d2/d1)
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"current.png", dpi=400) # nazev souboru 
    fig, ax1 = plt.subplots()
    cmap_hot=matplotlib.cm.get_cmap('afmhot')
    matplotlib.colors.LinearSegmentedColormap.set_gamma(cmap_hot, gamma)
    figure=plt.imshow(norm_map, cmap=cmap_hot,interpolation='None',extent=[0,d1,0,d2], origin="lower")
    fig.colorbar(figure, fraction=0.046, pad=0.04)
    fig.subplots_adjust(0,0,1,1)
    ax1.set_axis_off()
    fig.set_size_inches(4.4,4.4*d2/d1)
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+str(x1)+"-"+str(x2)+"norm"+str(exponent)+".png", dpi=400) # nazev souboru
    if abs(np.amax(Z_map)-np.amin(Z_map))>0.1:
        fig, ax1 = plt.subplots()
        figure=plt.imshow(Z_map, cmap=cmap_cold,interpolation='None',extent=[0,d1,0,d2], origin="lower")
        fig.colorbar(figure, fraction=0.046, pad=0.04)
        fig.subplots_adjust(0,0,1,1)
        ax1.set_axis_off()
        fig.set_size_inches(4.4,4.4*d2/d1)
        if save==True:
                plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+str(x1)+"-"+str(x2)+"topo.png", dpi=400) # nazev souboru
    fig, ax1 = plt.subplots() #osy 
    ax1.set_xlabel('Current [pA]')
    ax1.set_ylabel('Integrated counts '+str(x1)+"-"+str(x2)+" nm")
    #plt.xlim(xmin=0)
    ax1.plot(cur2,scounts, label="exciton"+str(x1)+"-"+str(x2)+" nm Bias "+str(filedata_dat[i1].header["Bias (V)"]),marker='x',linestyle="None")
    ax1.plot(cur2,scounts2, label="plasmon"+str(y1)+"-"+str(y2)+" nm Bias "+str(filedata_dat[i1].header["Bias (V)"]),marker='x',linestyle="None")
    #ax1.plot(cur2,ncounts2, label=str(filedata_dat[i1].header["Filename"])+" "+str(filedata_dat[i2].header["Filename"])+" "+"B "+str(filedata_dat[i1].header["Bias (V)"]),marker='x',linestyle="None")
    plt.legend()
    ax1.grid(True,linestyle=':')
    fig.set_size_inches(4.4,4.4)
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"mol-plas-idep.png", dpi=400) # nazev souboru 
    fig, ax1 = plt.subplots() #osy 
    ax1.set_xlabel('Current [pA]')
    ax1.set_ylabel('Integrated counts ')
    #plt.xlim(xmin=0)
    #ax1.plot(cur2,scounts, label="exciton"+str(x1)+"-"+str(x2)+" nm Bias "+str(filedata_dat[i1].header["Bias (V)"]),marker='x',linestyle="None")
    #ax1.plot(cur2,scounts2, label="plasmon"+str(y1)+"-"+str(y2)+" nm Bias "+str(filedata_dat[i1].header["Bias (V)"]),marker='x',linestyle="None")
    ax1.plot(cur2,ncounts2,label="mol div plasm"+str(x1)+"-"+str(x2)+ "div" +str(y1)+"-"+str(y2)+" nm Bias "+str(filedata_dat[i1].header["Bias (V)"]),marker='x',linestyle="None")
    plt.legend()
    ax1.grid(True,linestyle=':')
    fig.set_size_inches(4.4,4.4)
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"mol-div-plasm-idep.png", dpi=400) # nazev souboru 
def map_hm_shift(i1,i2,i_bg,x1,x2,d1,d2,unit,save,norm,sigma,gamma,exponent,shift):
    cur2=[]
    Z2=[]
    scounts=[]
    bias=[]
    ncounts=[]
    sum_bg=0
    if isfloat(i1)==True:
        i1=i1
    else:
        for j in range (0,len(file_numbers)):
            if str(filedata_dat[j].header["Filename"])==str(i1):
                i1=j
                break
    if isfloat(i2)==True:
        i2=i2
    else:
        for j in range (0,len(file_numbers)):
            if str(filedata_dat[j].header["Filename"])==str(i2):
                i2=j
                break
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if str(filedata_dat[j].header["Filename"])==str(i_bg):
                i_bg=j
                break
    lr=filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i1].signals['Wavelength (nm)'][-1]
    ncol=len(filedata_dat[i1].signals['Wavelength (nm)'])-1
    if i_bg<-0.1 and i_bg>-1:
        n_bg=0
        d_bg=0
        for j in range(i1,i1+d1):
            if (abs(float(filedata_dat[j].header["Current avg. (A)"]))*1E12)<7:
                n_bg=float(filedata_dat[j].header["Number of Accumulations"])
                sum_bg=sum_bg+np.sum(filedata_dat[j].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])/n_bg
                d_bg=d_bg+1
        sum_bg=sum_bg/d_bg
    elif i_bg<-1.1:
        n_bg=0
        d_bg=0
        for j in range(i2-d1,i2):
            if (abs(float(filedata_dat[j].header["Current avg. (A)"]))*1E12)<7:
                n_bg=float(filedata_dat[j].header["Number of Accumulations"])
                sum_bg=sum_bg+np.sum(filedata_dat[j].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])/n_bg
                d_bg=d_bg+1
        sum_bg=sum_bg/d_bg
    else:
        n_bg=float(filedata_dat[i_bg].header["Number of Accumulations"])
        sum_bg=np.sum(filedata_dat[i_bg].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])/n_bg
        
    for i in range(i1,i1+219):
        bias.append(filedata_dat[i].header["Bias (V)"])
        n=float(filedata_dat[i].header["Number of Accumulations"])
        try:
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"])+float(filedata_dat[i].header["Current avg. (A) end"]))/2*1E12
        except KeyError:
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        cur2.append(cur)
        try:
            Z=(float(filedata_dat[i].header["Z (m)"])+float(filedata_dat[i].header["Z (m) end"]))/2*1E9
        except KeyError:
            Z=(float(filedata_dat[i].header["Z (m)"]))*1E9
        Z2.append(Z)
        if norm==True:
            scounts.append((sum(filedata_dat[i].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])-n*sum_bg)/cur)
        else:
            scounts.append((sum(filedata_dat[i].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])-n*sum_bg))
            
    for i in range(i1+219+shift,i2+1):
        bias.append(filedata_dat[i].header["Bias (V)"])
        n=float(filedata_dat[i].header["Number of Accumulations"])
        try:
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"])+float(filedata_dat[i].header["Current avg. (A) end"]))/2*1E12
        except KeyError:
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        cur2.append(cur)
        try:
            Z=(float(filedata_dat[i].header["Z (m)"])+float(filedata_dat[i].header["Z (m) end"]))/2*1E9
        except KeyError:
            Z=(float(filedata_dat[i].header["Z (m)"]))*1E9
        Z2.append(Z)
        if norm==True:
            scounts.append((sum(filedata_dat[i].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])-n*sum_bg)/cur)
        else:
            scounts.append((sum(filedata_dat[i].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])-n*sum_bg))
            
    for i in range(0,shift):
        bias.append(filedata_dat[i1].header["Bias (V)"])
        n=float(filedata_dat[i1].header["Number of Accumulations"])
        try:
            cur=abs(float(filedata_dat[i1].header["Current avg. (A)"])+float(filedata_dat[i1].header["Current avg. (A) end"]))/2*1E12
        except KeyError:
            cur=abs(float(filedata_dat[i1].header["Current avg. (A)"]))*1E12
        cur2.append(cur)
        try:
            Z=(float(filedata_dat[i1].header["Z (m)"])+float(filedata_dat[i1].header["Z (m) end"]))/2*1E9
        except KeyError:
            Z=(float(filedata_dat[i1].header["Z (m)"]))*1E9
        Z2.append(Z)
        if norm==True:
            scounts.append((sum(filedata_dat[i1].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])-n*sum_bg)/cur)
        else:
            scounts.append((sum(filedata_dat[i1].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])-n*sum_bg))
    scounts=np.array(scounts)
    print(np.shape(scounts))    
    Z_map=np.reshape(Z2, (d2,d1))
    scounts_map=np.reshape(scounts, (d2,d1))
    cur_map=np.reshape(cur2, (d2,d1))
    
    scounts_map=filter_image(scounts_map,sigma)
    amin=np.amin(scounts_map)
    amax=np.amax(scounts_map)
    scounts_map=scounts_map-amin
    
    scounts=np.reshape(scounts_map, (len(scounts)))
    print(np.shape(scounts_map))
    
    for i in range(0,len(scounts)):
        if cur2[i]>8:
            ncounts.append(scounts[i]/cur2[i]**exponent)
        else:
            ncounts.append(0)
    norm_map=np.reshape(ncounts, (d2,d1))   
    #norm_map=norm_map**gamma
    #fig, ax1 = plt.subplots()
    #plt.imshow(ar_np[i1:i2+1,1,nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)], cmap='viridis',interpolation='None',extent=[x1,x2,i1,i2+1])
    #plt.colorbar
    #fig.set_size_inches(4.4,4.4)
    
    fig, ax1 = plt.subplots()
    plt.imshow(scounts_map[1:-1,2:], cmap='afmhot',interpolation='None',extent=[0,d1,0,d2], origin="lower")
    #fig.colorbar(figure, fraction=0.046, pad=0.04)
    fig.subplots_adjust(0,0,1,1)
    ax1.set_axis_off()
    fig.set_size_inches(4.4,4.4*d2/d1)
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+str(x1)+"-"+str(x2)+".png", dpi=400) # nazev souboru 
            
    fig, ax1 = plt.subplots()
    plt.imshow(cur_map, cmap=cmap_cold,interpolation='None',extent=[0,d1,0,d2], origin="lower")
    #fig.colorbar(figure, fraction=0.046, pad=0.04)
    fig.subplots_adjust(0,0,1,1)
    ax1.set_axis_off()
    fig.set_size_inches(4.4,4.4*d2/d1)
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"current.png", dpi=400) # nazev souboru 
    fig, ax1 = plt.subplots()
    cmap_hot=matplotlib.cm.get_cmap('coolwarm')
    matplotlib.colors.LinearSegmentedColormap.set_gamma(cmap_hot, gamma)
    figure=plt.imshow(norm_map, cmap=cmap_hot,interpolation='None',extent=[0,d1,0,d2], origin="lower")
    fig.colorbar(figure, fraction=0.046, pad=0.04)
    fig.subplots_adjust(0,0,1,1)
    ax1.set_axis_off()
    fig.set_size_inches(4.4,4.4*d2/d1)
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+str(x1)+"-"+str(x2)+"norm"+str(exponent)+".png", dpi=400) # nazev souboru
    if abs(np.amax(Z_map)-np.amin(Z_map))>0.1:
        fig, ax1 = plt.subplots()
        figure=plt.imshow(Z_map, cmap=cmap_cold,interpolation='None',extent=[0,d1,0,d2], origin="lower")
        fig.colorbar(figure, fraction=0.046, pad=0.04)
        fig.subplots_adjust(0,0,1,1)
        ax1.set_axis_off()
        fig.set_size_inches(4.4,4.4*d2/d1)
        if save==True:
                plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+str(x1)+"-"+str(x2)+"topo.png", dpi=400) # nazev souboru

def map_hmsxm(i1,i2,i_bg,l1,r1,l2,r2,unit,save,sigma,gamma,exponent,**kwargs):
    cur2=[]
    Z2=[]
    scounts=[]
    scounts2=[]
    bias=[]
    ncounts=[]
    ncounts2=[]
    sum_bg=0
    sum_bgar=[]
    sum_bg2=0
    if isfloat(i1)==True:
        i1=i1
    else:
        for j in range (0,len(file_numbers)):
            if str(filedata_dat[j].header["Filename"])==str(i1):
                i1=j
                break
    if isfloat(i2)==True:
        i2=i2
    else:
        for j in range (0,len(file_numbers)):
            if str(filedata_dat[j].header["Filename"])==str(i2):
                i2=j
                break
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if str(filedata_dat[j].header["Filename"])==str(i_bg):
                i_bg=j
                break
    lr=filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i1].signals['Wavelength (nm)'][-1]
    ncol=len(filedata_dat[i1].signals['Wavelength (nm)'])-1
    i=0
    while abs(abs(float(filedata_dat[i1+i+1].header["X (m)"])-float(filedata_dat[i1+i].header["X (m)"]))-abs(float(filedata_dat[i1+1].header["X (m)"])-float(filedata_dat[i1].header["X (m)"])))<1E-10:
        i += 1
        if i>(i2-i1):
            break
    else:
        d1=i+1
        d2=int((i2+1-i1)/d1)
        print(d1)
        print(d2)
    if i_bg<-0.1 and i_bg>-1:
        n_bg=0
        d_bg=0
        for j in range(i1,i1+d1):
            if (abs(float(filedata_dat[j].header["Current avg. (A)"]))*1E12)<7:
                n_bg=float(filedata_dat[j].header["Number of Accumulations"])
                sum_bg=sum_bg+np.sum(filedata_dat[j].signals['Counts'][nmtopix(l1,lr,rr,ncol):nmtopix(r1,lr,rr,ncol)])#/n_bg
                sum_bg2=sum_bg2+np.sum(filedata_dat[j].signals['Counts'][nmtopix(l2,lr,rr,ncol):nmtopix(r2,lr,rr,ncol)])#/n_bg
                d_bg=d_bg+1
        print(d_bg)
        sum_bg=sum_bg/d_bg
        sum_bg2=sum_bg2/d_bg
    if i_bg<-1.1 and i_bg>-2.01:
        n_bg=0
        d_bg=0
        for j in range(i2-d1,i2):
            if (abs(float(filedata_dat[j].header["Current avg. (A)"]))*1E12)<7:
                n_bg=float(filedata_dat[j].header["Number of Accumulations"])
                sum_bg=sum_bg+np.sum(filedata_dat[j].signals['Counts'][nmtopix(l1,lr,rr,ncol):nmtopix(r1,lr,rr,ncol)])#/n_bg
                sum_bg2=sum_bg2+np.sum(filedata_dat[j].signals['Counts'][nmtopix(l2,lr,rr,ncol):nmtopix(r2,lr,rr,ncol)])#/n_bg
                d_bg=d_bg+1
        print(d_bg)
        sum_bg=sum_bg/d_bg
        sum_bg2=sum_bg2/d_bg
    elif i_bg<-2.1:
        n_bg=0
        d_bg=0
        for j in range(i1,i2):
            if (abs(float(filedata_dat[j].header["Current avg. (A)"]))*1E12)<2:
                n_bg=float(filedata_dat[j].header["Number of Accumulations"])
                sum_bg=sum_bg+np.sum(filedata_dat[j].signals['Counts'][nmtopix(l1,lr,rr,ncol):nmtopix(r1,lr,rr,ncol)])#/n_bq
                sum_bg2=sum_bg2+np.sum(filedata_dat[j].signals['Counts'][nmtopix(l2,lr,rr,ncol):nmtopix(r2,lr,rr,ncol)])#/n_bg
                d_bg=d_bg+1
        print(d_bg)
        sum_bg=sum_bg/d_bg
        sum_bg2=sum_bg2/d_bg
    else:
        n_bg=float(filedata_dat[i_bg].header["Number of Accumulations"])
        sum_bg=np.sum(filedata_dat[i_bg].signals['Counts'][nmtopix(l1,lr,rr,ncol):nmtopix(r1,lr,rr,ncol)])#/n_bg
        sum_bg2=np.sum(filedata_dat[i_bg].signals['Counts'][nmtopix(l2,lr,rr,ncol):nmtopix(r2,lr,rr,ncol)])#/n_bg
        
    acq_time=(i2-i1)*float(filedata_dat[i1].header["GAT"]) #acq time
    scan_pixs="\t"+str(int(d1))+"\t"+str(int(d2))
    filename=str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"][-5:])+"_"+str(l1)+"-"+str(r1)+"and"+str(l2)+"-"+str(r2)+".sxm"
    xlen=(((float(filedata_dat[i1+d1-1].header["X (m)"])-float(filedata_dat[i1+1].header["X (m)"]))**2)+((float(filedata_dat[i1+d1-1].header["Y (m)"])-float(filedata_dat[i1+1].header["Y (m)"]))**2))**0.5
    ylen=(((float(filedata_dat[i2-d1+1+1].header["X (m)"])-float(filedata_dat[i1+1].header["X (m)"]))**2)+((float(filedata_dat[i2-d1+1+1].header["Y (m)"])-float(filedata_dat[i1+1].header["Y (m)"]))**2))**0.5
    print(xlen)
    print(ylen)
    print((float(filedata_dat[i1+d1-1].header["X (m)"])-float(filedata_dat[i1].header["X (m)"])),(float(filedata_dat[i1+d1-1].header["Y (m)"])-float(filedata_dat[i1].header["Y (m)"])))
    print(i1+d1-1,' ',i1 )
    print(i2-d1+1,' ',i1 )
    scan_range="\t"+str(xlen+2*xlen/(d1-2))+"\t"+str(ylen+ylen/(d2-1))
    arccos=np.degrees(np.arccos((float(filedata_dat[i1+d1-1].header["X (m)"])-float(filedata_dat[i1+1].header["X (m)"]))/(xlen)))
    print(arccos,'angle',d1,'d1',d2,'d2')
    if float(filedata_dat[i1+1].header["Y (m)"])<=float(filedata_dat[i1].header["Y (m)"]):
        scan_angle=arccos
    else:
        scan_angle=-arccos
    if (int(d1*d2)%2) == 0:
        scan_offsetx=(float(filedata_dat[int((i2-i1-1)/2)].header["X (m)"])+float(filedata_dat[int((i2-i1+1)/2)].header["X (m)"]))/2
        print((i2-i1-1)/2)
        scan_offsety=(float(filedata_dat[int((i2-i1-1)/2)].header["Y (m)"])+float(filedata_dat[int((i2-i1+1)/2)].header["Y (m)"]))/2
        print((i2-i1-1)/2)
    else:
        scan_offsetx=float(filedata_dat[int((i2-i1)/2)].header["X (m)"])
        scan_offsety=float(filedata_dat[int((i2-i1)/2)].header["Y (m)"])
    scan_offset="\t"+str(scan_offsetx)+"\t"+str(scan_offsety)
    scan_par={"ACQ_TIME":str(acq_time),"SCAN_PIXELS":str(scan_pixs),"SCAN_FILE":path+filename,"SCAN_RANGE":scan_range,"SCAN_OFFSET":scan_offset,"SCAN_ANGLE":str(scan_angle),"SCAN_DIR":"up"} 
    for i in range(i1,i2+1):
        bias.append(filedata_dat[i].header["Bias (V)"])
        n=float(filedata_dat[i].header["Number of Accumulations"])
        try:
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"])+float(filedata_dat[i].header["Current avg. (A) end"]))/2*1E12
        except KeyError:
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        cur2.append(cur)
        try:
            Z=(float(filedata_dat[i].header["Z (m)"])+float(filedata_dat[i].header["Z (m) end"]))/2*1E9
        except KeyError:
            Z=(float(filedata_dat[i].header["Z (m)"]))*1E9
        Z2.append(Z)
        if "fitbg" in kwargs:
            try:
                sum_bgcor=abs(nmtopix(l1,lr,rr,ncol)-nmtopix(r1,lr,rr,ncol))*(filedata_dat[i].signals['Counts'][nmtopix(kwargs["fitbg"][1],lr,rr,ncol)]-filedata_dat[i].signals['Counts'][nmtopix(kwargs["fitbg"][0],lr,rr,ncol)])
                scounts.append((sum(filedata_dat[i].signals['Counts'][nmtopix(l1,lr,rr,ncol):nmtopix(r1,lr,rr,ncol)])-sum_bgcor))
            except:
                scounts.append((sum(filedata_dat[i].signals['Counts'][nmtopix(l1,lr,rr,ncol):nmtopix(r1,lr,rr,ncol)])-sum_bg))
            try:
                sum_bg2cor=abs(nmtopix(l1,lr,rr,ncol)-nmtopix(r1,lr,rr,ncol))**(filedata_dat[i].signals['Counts'][nmtopix(kwargs["fitbg"][3],lr,rr,ncol)]-filedata_dat[i].signals['Counts'][nmtopix(kwargs["fitbg"][2],lr,rr,ncol)])
                scounts2.append((sum(filedata_dat[i].signals['Counts'][nmtopix(l2,lr,rr,ncol):nmtopix(r2,lr,rr,ncol)])-sum_bg2cor))
            except:
                scounts2.append((sum(filedata_dat[i].signals['Counts'][nmtopix(l2,lr,rr,ncol):nmtopix(r2,lr,rr,ncol)])-sum_bg2))
        else:
            scounts.append((sum(filedata_dat[i].signals['Counts'][nmtopix(l1,lr,rr,ncol):nmtopix(r1,lr,rr,ncol)])-sum_bg))
            scounts2.append((sum(filedata_dat[i].signals['Counts'][nmtopix(l2,lr,rr,ncol):nmtopix(r2,lr,rr,ncol)])-sum_bg2))
    scounts=np.array(scounts)
    scounts2=np.array(scounts2)
    print(np.shape(scounts))    
    Z_map=np.reshape(Z2, (d2,d1))
    scounts_map=np.reshape(scounts, (d2,d1))
    scounts2_map=np.reshape(scounts2, (d2,d1))
    cur_map=np.reshape(cur2, (d2,d1))
    
    scounts_map=filter_image(scounts_map,sigma)
    amin=np.amin(scounts_map)
    amax=np.amax(scounts_map)
    scounts_map=scounts_map-amin
    
    scounts2_map=filter_image(scounts2_map,sigma)
    amin2=np.amin(scounts2_map)
    amax2=np.amax(scounts2_map)
    scounts2_map=scounts2_map-amin
    
    scounts=np.reshape(scounts_map, (len(scounts)))
    scounts2=np.reshape(scounts2_map, (len(scounts2)))
    print(np.shape(scounts_map))
    
    for i in range(0,len(scounts)):
        if cur2[i]>2:
            ncounts.append(scounts[i]/cur2[i]**exponent)
            ncounts2.append(scounts2[i]/cur2[i]**exponent)
        else:
            ncounts.append(0)
            ncounts2.append(0)
    norm_map=np.reshape(ncounts, (d2,d1))
    norm2_map=np.reshape(ncounts2, (d2,d1))
    #norm_map=norm_map**gamma
    #fig, ax1 = plt.subplots()
    #plt.imshow(ar_np[i1:i2+1,1,nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)], cmap='viridis',interpolation='None',extent=[x1,x2,i1,i2+1])
    #plt.colorbar
    #fig.set_size_inches(4.4,4.4)
    channels=np.array([['14', 'Z', 'm','both','9.000E-9','0.000E+0'],['0', 'Current', 'A','both','9.000E-9','0.000E+0'],['40', 'Intensity'+str(l1)+"-"+str(r1)+'nm', 'Counts','both','1','0.000E+0'],['41', 'Norm.int'+str(l1)+"-"+str(r1)+'nm', 'Counts/pA','both','1','0.000E+0'],['42', 'Intensity'+str(l2)+"-"+str(r2)+'nm', 'Counts','both','1','0.000E+0'],['43', 'Norm.int'+str(l2)+"-"+str(r2)+'nm', 'Counts/pA','both','1','0.000E+0']])
    #dt = np.dtype('>f3')
    data=np.stack((Z_map,Z_map,cur_map,cur_map,scounts_map,scounts_map,norm_map,norm_map))
    data.astype(">f4")
    header={"Number of Accumulations":filedata_dat[i1].header["Number of Accumulations"],"Bias (V)":filedata_dat[i1].header["Bias (V)"],"Start time":filedata_dat[i1].header["Start time"]}
    writesxm(path,filename,header,scan_par,channels,Z_map,cur_map,scounts_map,norm_map,scounts2_map,norm2_map)
    fig, ax1 = plt.subplots()
    plt.imshow(scounts_map, cmap='afmhot',interpolation='None',extent=[0,d1,0,d2], origin="lower")
    #fig.colorbar(figure, fraction=0.046, pad=0.04)
    fig.subplots_adjust(0,0,1,1)
    ax1.set_axis_off()
    fig.set_size_inches(4.4,4.4*d2/d1)
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+str(l1)+"-"+str(r1)+"-B"+str("{0:.2f}".format(float(bias[0])))+".png", dpi=400) # nazev souboru 
            
    fig, ax1 = plt.subplots()
    plt.imshow(cur_map, cmap=cmap_cold,interpolation='None',extent=[0,d1,0,d2], origin="lower")
    #fig.colorbar(figure, fraction=0.046, pad=0.04)
    fig.subplots_adjust(0,0,1,1)
    ax1.set_axis_off()
    fig.set_size_inches(4.4,4.4*d2/d1)
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"current"+"-B"+str("{0:.2f}".format(float(bias[0])))+".png", dpi=400) # nazev souboru 
    fig, ax1 = plt.subplots()
    cmap_hot=matplotlib.cm.get_cmap('afmhot')
    matplotlib.colors.LinearSegmentedColormap.set_gamma(cmap_hot, gamma)
    figure=plt.imshow(norm_map, cmap=cmap_hot,interpolation='None',extent=[0,d1,0,d2], origin="lower")
    #fig.colorbar(figure, fraction=0.046, pad=0.04)
    fig.subplots_adjust(0,0,1,1)
    ax1.set_axis_off()
    fig.set_size_inches(4.4,4.4*d2/d1)
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+str(l1)+"-"+str(r1)+"norm"+str(exponent)+".png", dpi=400) # nazev souboru
    if abs(np.amax(Z_map)-np.amin(Z_map))>0.1:
        fig, ax1 = plt.subplots()
        figure=plt.imshow(Z_map, cmap=cmap_cold,interpolation='None',extent=[0,d1,0,d2], origin="lower")
        fig.colorbar(figure, fraction=0.046, pad=0.04)
        fig.subplots_adjust(0,0,1,1)
        ax1.set_axis_off()
        fig.set_size_inches(4.4,4.4*d2/d1)
        if save==True:
                plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+str(l1)+"-"+str(r1)+"topo.png", dpi=400) # nazev souboru
    
def map_hmsxmnf(i1,i2,i_bg,l1,r1,l2,r2,unit,save,sigma,gamma,exponent,**kwargs):
    cur2=[]
    Z2=[]
    scounts=[]
    scounts2=[]
    bias=[]
    ncounts=[]
    ncounts2=[]
    sum_bg=0
    sum_bgar=[]
    sum_bg2=0
    if isfloat(i1)==True:
        i1=i1
    else:
        for j in range (0,len(file_numbers)):
            if str(filedata_dat[j].header["Filename"])==str(i1):
                i1=j
                break
    if isfloat(i2)==True:
        i2=i2
    else:
        for j in range (0,len(file_numbers)):
            if str(filedata_dat[j].header["Filename"])==str(i2):
                i2=j
                break
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if str(filedata_dat[j].header["Filename"])==str(i_bg):
                i_bg=j
               # print(i_bg,"ibg")
                break
    lr=filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i1].signals['Wavelength (nm)'][-1]
    ncol=len(filedata_dat[i1].signals['Wavelength (nm)'])-1
    i=0
    while abs(abs(float(filedata_dat[i1+i+1].header["X (m)"])-float(filedata_dat[i1+i].header["X (m)"]))-abs(float(filedata_dat[i1+1].header["X (m)"])-float(filedata_dat[i1].header["X (m)"])))<1E-10:
        i += 1
        if i>(i2-i1):
            break
    else:
        d1=i+1
        d2=int((i2+1-i1)/d1)
        print(d1)
        print(d2)
    if i_bg<-0.1 and i_bg>-1:
        n_bg=0
        d_bg=0
        for j in range(i1,i1+d1):
            if (abs(float(filedata_dat[j].header["Current avg. (A)"]))*1E12)<7:
                n_bg=float(filedata_dat[j].header["Number of Accumulations"])
                sum_bg=sum_bg+np.sum(filedata_dat[j].signals['Counts nf'][nmtopix(l1,lr,rr,ncol):nmtopix(r1,lr,rr,ncol)])#/n_bg
                sum_bg2=sum_bg2+np.sum(filedata_dat[j].signals['Counts nf'][nmtopix(l2,lr,rr,ncol):nmtopix(r2,lr,rr,ncol)])#/n_bg
                d_bg=d_bg+1
        print(d_bg)
        sum_bg=sum_bg/d_bg
        sum_bg2=sum_bg2/d_bg
    if i_bg<-1.1 and i_bg>-2.01:
        n_bg=0
        d_bg=0
        for j in range(i2-d1,i2):
            if (abs(float(filedata_dat[j].header["Current avg. (A)"]))*1E12)<7:
                n_bg=float(filedata_dat[j].header["Number of Accumulations"])
                sum_bg=sum_bg+np.sum(filedata_dat[j].signals['Counts nf'][nmtopix(l1,lr,rr,ncol):nmtopix(r1,lr,rr,ncol)])#/n_bg
                sum_bg2=sum_bg2+np.sum(filedata_dat[j].signals['Counts nf'][nmtopix(l2,lr,rr,ncol):nmtopix(r2,lr,rr,ncol)])#/n_bg
                d_bg=d_bg+1
        print(d_bg)
        sum_bg=sum_bg/d_bg
        sum_bg2=sum_bg2/d_bg
    elif i_bg<-2.1:
        n_bg=0
        d_bg=0
        for j in range(i1,i2):
            if (abs(float(filedata_dat[j].header["Current avg. (A)"]))*1E12)<2:
                n_bg=float(filedata_dat[j].header["Number of Accumulations"])
                sum_bg=sum_bg+np.sum(filedata_dat[j].signals['Counts nf'][nmtopix(l1,lr,rr,ncol):nmtopix(r1,lr,rr,ncol)])#/n_bq
                sum_bg2=sum_bg2+np.sum(filedata_dat[j].signals['Counts nf'][nmtopix(l2,lr,rr,ncol):nmtopix(r2,lr,rr,ncol)])#/n_bg
                d_bg=d_bg+1
        print(d_bg)
        sum_bg=sum_bg/d_bg
        sum_bg2=sum_bg2/d_bg
    else:
        n_bg=float(filedata_dat[i_bg].header["Number of Accumulations"])
        sum_bg=np.sum(filedata_dat[i_bg].signals['Counts nf'][nmtopix(l1,lr,rr,ncol):nmtopix(r1,lr,rr,ncol)])#/n_bg
        sum_bg2=np.sum(filedata_dat[i_bg].signals['Counts nf'][nmtopix(l2,lr,rr,ncol):nmtopix(r2,lr,rr,ncol)])#/n_bg
        
    acq_time=(i2-i1)*float(filedata_dat[i1].header["GAT"]) #acq time
    scan_pixs="\t"+str(int(d1))+"\t"+str(int(d2))
    filename=str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"][-5:])+"_"+str(l1)+"-"+str(r1)+"and"+str(l2)+"-"+str(r2)+".sxm"
    xlen=(((float(filedata_dat[i1+d1-1].header["X (m)"])-float(filedata_dat[i1+1].header["X (m)"]))**2)+((float(filedata_dat[i1+d1-1].header["Y (m)"])-float(filedata_dat[i1+1].header["Y (m)"]))**2))**0.5
    ylen=(((float(filedata_dat[i2-d1+1+1].header["X (m)"])-float(filedata_dat[i1+1].header["X (m)"]))**2)+((float(filedata_dat[i2-d1+1+1].header["Y (m)"])-float(filedata_dat[i1+1].header["Y (m)"]))**2))**0.5
    print(xlen)
    print(ylen)
    print((float(filedata_dat[i1+d1-1].header["X (m)"])-float(filedata_dat[i1].header["X (m)"])),(float(filedata_dat[i1+d1-1].header["Y (m)"])-float(filedata_dat[i1].header["Y (m)"])))
    print(i1+d1-1,' ',i1 )
    print(i2-d1+1,' ',i1 )
    scan_range="\t"+str(xlen+2*xlen/(d1-2))+"\t"+str(ylen+ylen/(d2-1))
    arccos=np.degrees(np.arccos((float(filedata_dat[i1+d1-1].header["X (m)"])-float(filedata_dat[i1+1].header["X (m)"]))/(xlen)))
    print(arccos,'angle',d1,'d1',d2,'d2')
    if float(filedata_dat[i1+1].header["Y (m)"])<=float(filedata_dat[i1].header["Y (m)"]):
        scan_angle=arccos
    else:
        scan_angle=-arccos
    if (int(d1*d2)%2) == 0:
        scan_offsetx=(float(filedata_dat[int((i2-i1-1)/2)].header["X (m)"])+float(filedata_dat[int((i2-i1+1)/2)].header["X (m)"]))/2
        print((i2-i1-1)/2)
        scan_offsety=(float(filedata_dat[int((i2-i1-1)/2)].header["Y (m)"])+float(filedata_dat[int((i2-i1+1)/2)].header["Y (m)"]))/2
        print((i2-i1-1)/2)
    else:
        scan_offsetx=float(filedata_dat[int((i2-i1)/2)].header["X (m)"])
        scan_offsety=float(filedata_dat[int((i2-i1)/2)].header["Y (m)"])
    scan_offset="\t"+str(scan_offsetx)+"\t"+str(scan_offsety)
    scan_par={"ACQ_TIME":str(acq_time),"SCAN_PIXELS":str(scan_pixs),"SCAN_FILE":path+filename,"SCAN_RANGE":scan_range,"SCAN_OFFSET":scan_offset,"SCAN_ANGLE":str(scan_angle),"SCAN_DIR":"up"} 
    for i in range(i1,i2+1):
        bias.append(filedata_dat[i].header["Bias (V)"])
        n=float(filedata_dat[i].header["Number of Accumulations"])
        try:
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"])+float(filedata_dat[i].header["Current avg. (A) end"]))/2*1E12
        except KeyError:
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        cur2.append(cur)
        try:
            Z=(float(filedata_dat[i].header["Z (m)"])+float(filedata_dat[i].header["Z (m) end"]))/2*1E9
        except KeyError:
            Z=(float(filedata_dat[i].header["Z (m)"]))*1E9
        Z2.append(Z)
        if "fitbg" in kwargs:
            try:
                sum_bgcor=abs(nmtopix(l1,lr,rr,ncol)-nmtopix(r1,lr,rr,ncol))*(filedata_dat[i].signals['Counts nf'][nmtopix(kwargs["fitbg"][1],lr,rr,ncol)]-filedata_dat[i].signals['Counts nf'][nmtopix(kwargs["fitbg"][0],lr,rr,ncol)])
                scounts.append((sum(filedata_dat[i].signals['Counts nf'][nmtopix(l1,lr,rr,ncol):nmtopix(r1,lr,rr,ncol)])-sum_bgcor))
            except:
                scounts.append((sum(filedata_dat[i].signals['Counts nf'][nmtopix(l1,lr,rr,ncol):nmtopix(r1,lr,rr,ncol)])-sum_bg))
            try:
                sum_bg2cor=abs(nmtopix(l1,lr,rr,ncol)-nmtopix(r1,lr,rr,ncol))**(filedata_dat[i].signals['Counts nf'][nmtopix(kwargs["fitbg"][3],lr,rr,ncol)]-filedata_dat[i].signals['Counts nf'][nmtopix(kwargs["fitbg"][2],lr,rr,ncol)])
                scounts2.append((sum(filedata_dat[i].signals['Counts nf'][nmtopix(l2,lr,rr,ncol):nmtopix(r2,lr,rr,ncol)])-sum_bg2cor))
            except:
                scounts2.append((sum(filedata_dat[i].signals['Counts nf'][nmtopix(l2,lr,rr,ncol):nmtopix(r2,lr,rr,ncol)])-sum_bg2))
        else:
            scounts.append((sum(filedata_dat[i].signals['Counts nf'][nmtopix(l1,lr,rr,ncol):nmtopix(r1,lr,rr,ncol)])-sum_bg))
            scounts2.append((sum(filedata_dat[i].signals['Counts nf'][nmtopix(l2,lr,rr,ncol):nmtopix(r2,lr,rr,ncol)])-sum_bg2))
    if "tol" in kwargs:
        tol= float(kwargs["tol"])
    else:
        tol=0.3
    scounts=np.array(scounts)
    scounts2=np.array(scounts2)
    print(np.shape(scounts))    
    Z_map=np.reshape(Z2, (d2,d1))
    scounts_map=np.reshape(scounts, (d2,d1))
    scounts2_map=np.reshape(scounts2, (d2,d1))
    cur_map=np.reshape(cur2, (d2,d1))
    
    scounts_map=filter_2ddespike(scounts_map,tol)
  #  print(np.shape(scounts_map))
    scounts_map=filter_image(scounts_map,sigma)
   # print(np.shape(scounts))
  #  print(scounts_map,"amin")
    amin=np.amin(scounts_map)
   # print(amin,"amin")
    amax=np.amax(scounts_map)
    scounts_map=scounts_map-amin
    
    scounts2_map=filter_2ddespike(scounts2_map,tol)
    scounts2_map=filter_image(scounts2_map,sigma)
    amin2=np.amin(scounts2_map)
    amax2=np.amax(scounts2_map)
    scounts2_map=scounts2_map-amin
    
    scounts=np.reshape(scounts_map, (len(scounts)))
    scounts2=np.reshape(scounts2_map, (len(scounts2)))
    print(np.shape(scounts_map))
    
    for i in range(0,len(scounts)):
        if cur2[i]>2:
            ncounts.append(scounts[i]/cur2[i]**exponent)
            ncounts2.append(scounts2[i]/cur2[i]**exponent)
        else:
            ncounts.append(0)
            ncounts2.append(0)
    norm_map=np.reshape(ncounts, (d2,d1))
    norm2_map=np.reshape(ncounts2, (d2,d1))
    #norm_map=norm_map**gamma
    #fig, ax1 = plt.subplots()
    #plt.imshow(ar_np[i1:i2+1,1,nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)], cmap='viridis',interpolation='None',extent=[x1,x2,i1,i2+1])
    #plt.colorbar
    #fig.set_size_inches(4.4,4.4)
    channels=np.array([['14', 'Z', 'm','both','9.000E-9','0.000E+0'],['0', 'Current', 'A','both','9.000E-9','0.000E+0'],['40', 'Intensity'+str(l1)+"-"+str(r1)+'nm', 'Counts','both','1','0.000E+0'],['41', 'Norm.int'+str(l1)+"-"+str(r1)+'nm', 'Counts/pA','both','1','0.000E+0'],['42', 'Intensity'+str(l2)+"-"+str(r2)+'nm', 'Counts','both','1','0.000E+0'],['43', 'Norm.int'+str(l2)+"-"+str(r2)+'nm', 'Counts/pA','both','1','0.000E+0']])
    #dt = np.dtype('>f3')
    data=np.stack((Z_map,Z_map,cur_map,cur_map,scounts_map,scounts_map,norm_map,norm_map))
    data.astype(">f4")
    header={"Number of Accumulations":filedata_dat[i1].header["Number of Accumulations"],"Bias (V)":filedata_dat[i1].header["Bias (V)"],"Start time":filedata_dat[i1].header["Start time"]}
    writesxm(path,filename,header,scan_par,channels,Z_map,cur_map,scounts_map,norm_map,scounts2_map,norm2_map)
    fig, ax1 = plt.subplots()
    plt.imshow(scounts_map, cmap='afmhot',interpolation='None',extent=[0,d1,0,d2], origin="lower")
    #fig.colorbar(figure, fraction=0.046, pad=0.04)
    fig.subplots_adjust(0,0,1,1)
    ax1.set_axis_off()
    fig.set_size_inches(4.4,4.4*d2/d1)
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+str(l1)+"-"+str(r1)+"-B"+str("{0:.2f}".format(float(bias[0])))+".png", dpi=400) # nazev souboru 
            
    fig, ax1 = plt.subplots()
    plt.imshow(cur_map, cmap=cmap_cold,interpolation='None',extent=[0,d1,0,d2], origin="lower")
    #fig.colorbar(figure, fraction=0.046, pad=0.04)
    fig.subplots_adjust(0,0,1,1)
    ax1.set_axis_off()
    fig.set_size_inches(4.4,4.4*d2/d1)
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"current"+"-B"+str("{0:.2f}".format(float(bias[0])))+".png", dpi=400) # nazev souboru 
    fig, ax1 = plt.subplots()
    cmap_hot=matplotlib.cm.get_cmap('afmhot')
    matplotlib.colors.LinearSegmentedColormap.set_gamma(cmap_hot, gamma)
    figure=plt.imshow(norm_map, cmap=cmap_hot,interpolation='None',extent=[0,d1,0,d2], origin="lower")
    #fig.colorbar(figure, fraction=0.046, pad=0.04)
    fig.subplots_adjust(0,0,1,1)
    ax1.set_axis_off()
    fig.set_size_inches(4.4,4.4*d2/d1)
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+str(l1)+"-"+str(r1)+"norm"+str(exponent)+".png", dpi=400) # nazev souboru
    if abs(np.amax(Z_map)-np.amin(Z_map))>0.1:
        fig, ax1 = plt.subplots()
        figure=plt.imshow(Z_map, cmap=cmap_cold,interpolation='None',extent=[0,d1,0,d2], origin="lower")
        fig.colorbar(figure, fraction=0.046, pad=0.04)
        fig.subplots_adjust(0,0,1,1)
        ax1.set_axis_off()
        fig.set_size_inches(4.4,4.4*d2/d1)
        if save==True:
                plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+str(l1)+"-"+str(r1)+"topo.png", dpi=400) # nazev souboru
    
def transfer_tre(i1,i2,i_bg,i_ref,x1,x2,unit,save,norm):
    cur2=[]
    scounts=[]
    bias=[]
    bias2=[]
    freq=[]
    tre=[]
    amp0=[]
    counts_cor=[]
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
    if isfloat(i_ref)==True:
        i_ref=i_ref
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_ref):
                i_ref=j
                break
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bg):
                i_bg=j
    lr=filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i1].signals['Wavelength (nm)'][-1]       
    ncol=len(filedata_dat[i1].signals['Wavelength (nm)'])-1
    for i in range(i1,i2+1):
        bias.append(filedata_dat[i].header["Bias (V)"])
        freq.append(filedata_dat[i].header["GEN FREQ (MHz)"])
        amp0.append(float(filedata_dat[i].header["GEN AMP (mV)"]))
        n=float(filedata_dat[i].header["Number of Accumulations"])
        bg=savgol_filter(filedata_dat[i_bg].signals['Counts'],21,0)
        signal_nf=np.subtract(filedata_dat[i].signals['Counts'], bg)
        signal=savgol_filter(np.subtract(filedata_dat[i].signals['Counts'], bg),11,0)
        counts_cor.append(np.subtract(filedata_dat[i].signals['Counts'], bg))
        cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        
        for ij in range(0,len(bg)):
            if signal[ij]>5 and signal_nf[ij+1]>10 and signal_nf[ij+5]>10 and signal_nf[ij+10]>10 and signal_nf[ij+30]>10:
                tre.append(filedata_dat[i1].signals['Wavelength (nm)'][ij])
                break
            if ij==len(bg-1):
                tre.append(np.NaN)
        fig, ax1 = plt.subplots() #osy 
        ax1.set_xlabel('wl [nm]')
        ax1.set_ylabel('signal [a.u.]')
        plt.xlim(xmin=0)
        ax1.plot(filedata_dat[i1].signals['Wavelength (nm)'],signal, label="frekvence",linestyle="--")
        plt.legend()
        ax1.grid(True,linestyle=':')
        ax1.set_xlim((427, 646)) 
        ax1.set_ylim((-5, 25)) 
        fig.set_size_inches(4.4,4.4)
        
    for i in range(i_ref,i_ref+1):
        bias.append(filedata_dat[i].header["Bias (V)"])
        n=float(filedata_dat[i].header["Number of Accumulations"])
        bg=savgol_filter(filedata_dat[i_bg].signals['Counts'],21,0)
        signal_nf=np.subtract(filedata_dat[i].signals['Counts'], bg)
        signal=savgol_filter(np.subtract(filedata_dat[i].signals['Counts'], bg),11,0)
        counts_cor.append(np.subtract(filedata_dat[i].signals['Counts'], bg))
        cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        
        for ij in range(0,len(bg)):
            if signal[ij]>10 and signal_nf[ij+1]>10 and signal_nf[ij+5]>10 and signal_nf[ij+10]>10:
                tre_ref=(filedata_dat[i1].signals['Wavelength (nm)'][ij])
                break
            if ij==len(bg-1):
                tre_ref=(np.NaN)
    dif=1000*abs((eV/tre_ref)-np.divide(eV,(np.array(tre))))
    print(len(dif))
    print(len(amp0))
    amp=np.divide(dif,np.array(amp0))          
    fig, ax1 = plt.subplots() #osy 
    ax1.set_xlabel('Frequency [MHz]')
    ax1.set_ylabel('Treshold [nm]')
    #plt.xlim(xmin=0)
    ax1.plot(freq,amp, label=str(filedata_dat[i1].header["Filename"])+" "+str(filedata_dat[i2].header["Filename"])+" "+"B "+str(filedata_dat[i1].header["Bias (V)"]),linestyle="-")
    plt.legend()
    #popt, pcov=curve_fit(npower2,cur2,scounts)
    #print(popt[1])
    #ax1.plot(cur2,npower(cur2,popt[0],popt[1],popt[2],popt[3]), label="fit")
    #ax1.plot(cur2,npower2(cur2,popt[0],popt[1]), label="fit")
    ax1.grid(True,linestyle=':')
    fig.set_size_inches(4.4,4.4)
    ax1.set_yscale('log')
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"r"+str(x1)+"-"+str(x2)+"transfer-amp.png", dpi=400, bbox_inches = 'tight') # nazev souboru 
    
def transfer_tre_all(i1,i2,i_bg,i_ref,x1,x2,unit,save,norm):
    cur2=[]
    scounts=[]
    bias=[]
    bias2=[]
    freq=[]
    tre=[]
    amp0=[]
    counts_cor=[]
    dif=[]
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
    if isfloat(i_ref)==True:
        i_ref=i_ref
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_ref):
                i_ref=j
                break
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bg):
                i_bg=j
    lr=filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i1].signals['Wavelength (nm)'][-1]       
    ncol=len(filedata_dat[i1].signals['Wavelength (nm)'])-1
    #fig, ax1 = plt.subplots() #osy 
    #ax1.set_xlabel('wl [nm]')
    #ax1.set_ylabel('signal [a.u.]')
    #plt.xlim(xmin=0)
    for i in range(0, len(file_numbers)):
        #  if "sweep" and "LS2" in file_names[i] and float(filedata_dat[i].header["Bias (V)"])>1.8:# version 1
        if "fr-calibration" in file_names[i] and float(filedata_dat[i].header["Bias (V)"])==1.4:
            bias.append(filedata_dat[i].header["Bias (V)"])
            #if float(filedata_dat[i].header["Bias (V)"])>1.9:
            if float(filedata_dat[i].header["Bias (V)"])>1.3:
                bias2.append(True)
            else:
                bias2.append(False)
            #freq.append(filedata_dat[i].header["GEN FREQ (MHz)"])
            freq.append(filedata_dat[i].header["genarb frequency (MHz)"])
            if float(filedata_dat[i].header["GEN AMP (mV)"])>3500:
                amp0.append(float(1000))
            else:
                amp0.append(float(filedata_dat[i].header["GEN AMP (mV)"]))
            n=float(filedata_dat[i].header["Number of Accumulations"])
            bg=savgol_filter(filedata_dat[i_bg].signals['Counts'],21,0)
            signal_nf=np.subtract(filedata_dat[i].signals['Counts'], bg)
            signal=savgol_filter(np.subtract(filedata_dat[i].signals['Counts'], bg),11,0)
            counts_cor.append(np.subtract(filedata_dat[i].signals['Counts'], bg))
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        
            for ij in range(0,len(bg)):
                if signal[ij]>5 and signal_nf[ij+1]>10 and signal_nf[ij+5]>10 and signal_nf[ij+10]>10:
                    tre.append(filedata_dat[i1].signals['Wavelength (nm)'][ij])
                    break
                if ij==len(bg-1):
                    tre.append(np.NaN)
                
     #           ax1.plot(filedata_dat[i1].signals['Wavelength (nm)'],signal, label=str(i),linestyle="--")
                
    #plt.legend()
    #ax1.grid(True,linestyle=':')
    #ax1.set_xlim((427, 646)) 
    #ax1.set_ylim((-5, 25)) 
    #fig.set_size_inches(4.4,4.4)    
    for i in range(i_ref,i_ref+1):
        bias.append(filedata_dat[i].header["Bias (V)"])
        n=float(filedata_dat[i].header["Number of Accumulations"])
        bg=savgol_filter(filedata_dat[i_bg].signals['Counts'],21,0)
        signal_nf=np.subtract(filedata_dat[i].signals['Counts'], bg)
        signal=savgol_filter(np.subtract(filedata_dat[i].signals['Counts'], bg),11,0)
        counts_cor.append(np.subtract(filedata_dat[i].signals['Counts'], bg))
        cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        
        for ij in range(0,len(bg)):
            if signal[ij]>10 and signal_nf[ij+1]>10 and signal_nf[ij+5]>10 and signal_nf[ij+10]>10:
                tre_ref=(filedata_dat[i1].signals['Wavelength (nm)'][ij])
                break
            if ij==len(bg-1):
                tre_ref=(np.NaN)
                
    for i in range(i2,i2+1):
        bias.append(filedata_dat[i].header["Bias (V)"])
        n=float(filedata_dat[i].header["Number of Accumulations"])
        bg=savgol_filter(filedata_dat[i_bg].signals['Counts'],21,0)
        signal_nf=np.subtract(filedata_dat[i].signals['Counts'], bg)
        signal=savgol_filter(np.subtract(filedata_dat[i].signals['Counts'], bg),11,0)
        counts_cor.append(np.subtract(filedata_dat[i].signals['Counts'], bg))
        cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        
        for ij in range(0,len(bg)):
            if signal[ij]>10 and signal_nf[ij+1]>10 and signal_nf[ij+5]>10 and signal_nf[ij+10]>10:
                tre_ref2=(filedata_dat[i1].signals['Wavelength (nm)'][ij])
                break
            if ij==len(bg-1):
                tre_ref2=(np.NaN)
    for i in range(0,len(tre)):
        if bias2[i]==True:
            dif.append(1000*abs((eV/tre_ref)-(eV/tre[i])))
        else:
            dif.append(1000*abs((eV/tre_ref2)-(eV/tre[i])))
    print(len(dif))
    print(len(amp0))
    amp=np.divide(np.array(dif),np.array(amp0))          
    fig, ax1 = plt.subplots() #osy 
    ax1.set_xlabel('Frequency [MHz]')
    ax1.set_ylabel('Treshold [nm]')
    ax1.set_yscale('log')
    #plt.xlim(xmin=0)
    ax1.plot(freq,amp, label=str(filedata_dat[i1].header["Filename"])+" "+str(filedata_dat[i2].header["Filename"])+" "+"B "+str(filedata_dat[i1].header["Bias (V)"]),marker=".",linestyle="None")
    plt.legend()
    #popt, pcov=curve_fit(npower2,cur2,scounts)
    #print(popt[1])
    #ax1.plot(cur2,npower(cur2,popt[0],popt[1],popt[2],popt[3]), label="fit")
    #ax1.plot(cur2,npower2(cur2,popt[0],popt[1]), label="fit")
    ax1.grid(True,linestyle=':')
    fig.set_size_inches(4.4,4.4)
    
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"r"+str(x1)+"-"+str(x2)+"transfer-amp.png", dpi=400, bbox_inches = 'tight') # nazev souboru 
    
    
    
def transfer_tre_new(i1,i2,i_bg,i_ref,x1,x2,unit,save,biasonly):
    cur2=[]
    scounts=[]
    bias=[]
    bias2=[]
    freq=[]
    tre=[]
    amp0=[]
    counts_cor=[]
    dif=[]
    amp_cond=[]
    tre_refar=[]
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
    if isfloat(i_ref)==True:
        i_ref=i_ref
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_ref):
                i_ref=j
                break
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bg):
                i_bg=j
    lr=filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i1].signals['Wavelength (nm)'][-1]       
    ncol=len(filedata_dat[i1].signals['Wavelength (nm)'])-1

    #for i in range(0, len(file_numbers)):
    for i in range(i1, i2+1):
        #  if "sweep" and "LS2" in file_names[i] and float(filedata_dat[i].header["Bias (V)"])>1.8:# version 1
        #if "fr-calibration" in file_names[i] and float(filedata_dat[i].header["Bias (V)"])==1.5:
       # if "freqsweep" in file_names[i] and float(filedata_dat[i].header["genarb power (mV)"])>10 and float(filedata_dat[i].header["Bias (V)"])==biasonly:
        if "freqsweep" in file_names[i]  and float(filedata_dat[i].header["Bias (V)"])==biasonly:
            bias.append(filedata_dat[i].header["Bias (V)"])
            #if float(filedata_dat[i].header["Bias (V)"])>1.9:
            if float(filedata_dat[i].header["Bias (V)"])>1.3:
                bias2.append(True)
            else:
                bias2.append(False)
            #freq.append(filedata_dat[i].header["GEN FREQ (MHz)"])
            freq.append(filedata_dat[i].header["genarb frequency (MHz)"])
            amp0.append(float(filedata_dat[i].header["genarb power (mV)"]))
            n=float(filedata_dat[i].header["Number of Accumulations"])
            bg=savgol_filter(filedata_dat[i_bg].signals['Counts'],11,0)
            signal_nf=np.subtract(filedata_dat[i].signals['Counts'], bg)
            signal=savgol_filter(np.subtract(filedata_dat[i].signals['Counts'], bg),11,0)
            counts_cor.append(np.subtract(filedata_dat[i].signals['Counts'], bg))
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        
            for ij in range(0,len(bg)):
                if signal[ij]>3 and signal_nf[ij+1]>3 and signal_nf[ij+10]>5 and signal_nf[ij+20]>10 and signal_nf[ij+30]>20 and signal_nf[ij+40]>30:
                    tre.append(filedata_dat[i1].signals['Wavelength (nm)'][ij])
                    break
                if ij==len(bg-1):
                    tre.append(np.NaN)
                
            fig, ax1 = plt.subplots() #osy 
            ax1.set_xlabel('wl [nm]')
            ax1.set_ylabel('signal [a.u.]')
            plt.xlim(xmin=0)
            ax1.plot(filedata_dat[i1].signals['Wavelength (nm)'],signal, label="frekvence",linestyle="--")
            plt.axvline(x=tre[-1], label=str(tre[-1]))
            plt.legend()
            ax1.grid(True,linestyle=':')
            ax1.set_xlim((tre[-1]-30, tre[-1]+60)) 
            ax1.set_ylim((-5, 25)) 
            fig.set_size_inches(4.4,4.4)
                
   
    for i in range(i_ref,len(file_names)):
        #print(filedata_dat[i].header["Bias (V)"])
        if float(filedata_dat[i].header["Bias (V)"])==biasonly and "OFF" in filedata_dat[i].header["genarb output"]:
            bias.append(filedata_dat[i].header["Bias (V)"])
            n=float(filedata_dat[i].header["Number of Accumulations"])
            bg=savgol_filter(filedata_dat[i_bg].signals['Counts'],21,0)
            signal_nf=np.subtract(filedata_dat[i].signals['Counts'], bg)
            signal=savgol_filter(np.subtract(filedata_dat[i].signals['Counts'], bg),11,0)
            counts_cor.append(np.subtract(filedata_dat[i].signals['Counts'], bg))
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
            
            for ij in range(0,len(bg)):
                if signal[ij]>3 and signal_nf[ij+1]>5 and signal_nf[ij+10]>5 and signal_nf[ij+20]>10 and signal_nf[ij+40]>30:
                    tre_refar.append(filedata_dat[i1].signals['Wavelength (nm)'][ij])
                    print(tre_refar[-1])
                    print(eV/tre_refar[-1])
                    break
                if ij==len(bg-1):
                    tre_refar.append(np.NaN)
    
            fig, ax1 = plt.subplots() #osy 
            ax1.set_xlabel('wl [nm]')
            ax1.set_ylabel('signal [a.u.]')
            plt.xlim(xmin=0)
            ax1.plot(filedata_dat[i1].signals['Wavelength (nm)'],signal, label="frekvence_ref"+bias[-1],linestyle="--",color="red")
            plt.axvline(x=tre_refar[-1],label=str(tre_refar[-1]))
            plt.legend()
            ax1.grid(True,linestyle=':')
            ax1.set_xlim((tre_refar[-1]-30, tre_refar[-1]+60)) 
            ax1.set_ylim((-5, 25)) 
            fig.set_size_inches(4.4,4.4)   
    tre_ref=np.nanmean(np.array(tre_refar))
    #print(tre_refar,"reference array")        
    for i in range(i2,i2+1):
        bias.append(filedata_dat[i].header["Bias (V)"])
        n=float(filedata_dat[i].header["Number of Accumulations"])
        bg=savgol_filter(filedata_dat[i_bg].signals['Counts'],21,0)
        signal_nf=np.subtract(filedata_dat[i].signals['Counts'], bg)
        signal=savgol_filter(np.subtract(filedata_dat[i].signals['Counts'], bg),11,0)
        counts_cor.append(np.subtract(filedata_dat[i].signals['Counts'], bg))
        cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        
        for ij in range(0,len(bg)):
            if signal[ij]>3 and signal_nf[ij+1]>3 and signal_nf[ij+5]>5 and signal_nf[ij+10]>10:
                tre_ref2=(filedata_dat[i1].signals['Wavelength (nm)'][ij])
                print(tre_ref2)
                print(eV/tre_ref2)
                break
            if ij==len(bg-1):
                tre_ref2=(np.NaN)
    for i in range(0,len(tre)):
        if (tre_ref-tre[i])>14:
            dif.append(1000*abs((eV/tre_ref)-(eV/tre[i])))
            print(tre_ref-tre[i],"difference nm")
            amp_cond.append(dif[i]/amp0[i])
        else:
            dif.append(1000*abs((eV/tre_ref)-(eV/tre[i])))
            amp_cond.append(np.NaN)
    print(len(dif))
    print(len(amp0))
    amp=np.divide(np.array(dif),np.array(amp0))   
    f_amp=scipy.signal.medfilt(amp,5)       
    fig, ax1 = plt.subplots() #osy 
    ax1.set_xlabel('Frequency [MHz]')
    ax1.set_ylabel('Transfer function')
    ax1.set_ylim((0.001,1))
    ax1.set_yscale('log')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Amplitude [meV]')  
    #plt.xlim(xmin=0)
    ax1.plot(freq,amp, label=str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"][-5:])+" "+"B "+str("{0:.2f}".format(float(bias[0])))+"V",marker=".",linestyle="None")
 #   ax1.plot(freq,amp_cond, label=">15 nm shift",marker=".",linestyle="None")
    ax2.plot(freq,np.array(dif), label="compensated",linestyle="-",color="red")
   # plt.legend()
    #popt, pcov=curve_fit(npower2,cur2,scounts)
    #print(popt[1])
    #ax1.plot(cur2,npower(cur2,popt[0],popt[1],popt[2],popt[3]), label="fit")
    #ax1.plot(cur2,npower2(cur2,popt[0],popt[1]), label="fit")
   # ax1.grid(True,linestyle=':')
    fig.set_size_inches(4.4,4.4)
    f= open(path+str(filedata_dat[i1].header["Filename"])+"cal_function.txt","w+")
    f.write("Frequency [MHz]"+'\t'+"Transfer function [fraction]"+'\t'+"Transfer function filtered [fraction]"+"\n")
    for i in range (0,len(amp)):
        f.write(str("{0:.1f}".format(float(freq[i])))+'\t'+str("{0:.4f}".format(float(amp[i])))+'\t'+str("{0:.4f}".format(float(f_amp[i])))+"\n")
    f.close()
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"r"+str(x1)+"-"+str(x2)+"transfer-amp.png", dpi=400, bbox_inches = 'tight') # nazev souboru 
    
def transfer_tre_new2(i1,i2,i_bg,i_ref,i_ref2,save,biasonly):
    cur2=[]
    scounts=[]
    bias=[]
    bias2=[]
    freq=[]
    tre=[]
    amp0=[]
    counts_cor=[]
    dif=[]
    amp_cond=[]
    tre_refar=[]
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
    if isfloat(i_ref)==True:
        i_ref=i_ref
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_ref):
                i_ref=j
                break
    if isfloat(i_ref2)==True:
        i_ref2=i_ref2
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_ref2):
                i_ref2=j
                break
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bg):
                i_bg=j
    lr=filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i1].signals['Wavelength (nm)'][-1]       
    ncol=len(filedata_dat[i1].signals['Wavelength (nm)'])-1
    ppnm=ncol/(rr-lr)
    print(ppnm,"ppnm")
    #for i in range(0, len(file_numbers)):
    for i in range(i1, i2+1):
        #  if "sweep" and "LS2" in file_names[i] and float(filedata_dat[i].header["Bias (V)"])>1.8:# version 1
        #if "fr-calibration" in file_names[i] and float(filedata_dat[i].header["Bias (V)"])==1.5:
       # if "freqsweep" in file_names[i] and float(filedata_dat[i].header["genarb power (mV)"])>10 and float(filedata_dat[i].header["Bias (V)"])==biasonly:
        if float(filedata_dat[i].header["Bias (V)"])==biasonly:
            bias.append(filedata_dat[i].header["Bias (V)"])
            #if float(filedata_dat[i].header["Bias (V)"])>1.9:
            if float(filedata_dat[i].header["Bias (V)"])>1.3:
                bias2.append(True)
            else:
                bias2.append(False)
            #freq.append(filedata_dat[i].header["GEN FREQ (MHz)"])
            freq.append(float(filedata_dat[i].header["genarb frequency (MHz)"]))
            amp0.append(float(filedata_dat[i].header["genarb power (mV)"]))
            n=float(filedata_dat[i].header["Number of Accumulations"])
            bg=savgol_filter(filedata_dat[i_bg].signals['Counts'],11,0)
            signal_nf=np.subtract(filedata_dat[i].signals['Counts'], bg)
            signal=savgol_filter(np.subtract(filedata_dat[i].signals['Counts'], bg),11,0)
            counts_cor.append(np.subtract(filedata_dat[i].signals['Counts'], bg))
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        
            for ij in range(0,len(bg)):
                if signal[ij]>3 and signal_nf[ij+int(ppnm*1)]>3 and signal_nf[ij+int(ppnm*5)]>5 and signal_nf[ij+int(ppnm*10)]>10 and signal_nf[ij+int(ppnm*10)]>20 and signal_nf[ij+40]>10:
                    tre.append(filedata_dat[i1].signals['Wavelength (nm)'][ij])
                    break
                if ij==len(bg-1):
                    tre.append(np.NaN)
                
            fig, ax1 = plt.subplots() #osy 
            ax1.set_xlabel('wl [nm]')
            ax1.set_ylabel('signal [a.u.]')
            plt.xlim(xmin=0)
            ax1.plot(filedata_dat[i1].signals['Wavelength (nm)'],signal_nf, label="frekvence",linestyle="--")
            ax1.plot(filedata_dat[i1].signals['Wavelength (nm)'],signal, label="frekvence",linestyle="-")
            plt.axvline(x=tre[-1], label=str(tre[-1]))
            plt.legend()
            ax1.grid(True,linestyle=':')
            ax1.set_xlim((tre[-1]-30, tre[-1]+60)) 
            ax1.set_ylim((-5, 25)) 
            fig.set_size_inches(4.4,4.4)
                
   
    for i in range(i_ref,i_ref2+1):
        #print(filedata_dat[i].header["Bias (V)"])
        if float(filedata_dat[i].header["Bias (V)"])==biasonly and "OFF" in filedata_dat[i].header["genarb output"]:
            bias.append(filedata_dat[i].header["Bias (V)"])
            n=float(filedata_dat[i].header["Number of Accumulations"])
            bg=savgol_filter(filedata_dat[i_bg].signals['Counts'],21,0)
            signal_nf=np.subtract(filedata_dat[i].signals['Counts'], bg)
            signal=savgol_filter(np.subtract(filedata_dat[i].signals['Counts'], bg),11,0)
            counts_cor.append(np.subtract(filedata_dat[i].signals['Counts'], bg))
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
            
            for ij in range(0,len(bg)):
                if signal[ij]>3 and signal_nf[ij+int(ppnm*1)]>3 and signal_nf[ij+int(ppnm*5)]>5 and signal_nf[ij+int(ppnm*10)]>10 and signal_nf[ij+int(ppnm*10)]>20 and signal_nf[ij+60]>10:
                    tre_refar.append(filedata_dat[i1].signals['Wavelength (nm)'][ij])
                    print(tre_refar[-1])
                    print(eV/tre_refar[-1])
                    break
                if ij==len(bg-1):
                    tre_refar.append(np.NaN)
    
            fig, ax1 = plt.subplots() #osy 
            ax1.set_xlabel('wl [nm]')
            ax1.set_ylabel('signal [a.u.]')
            plt.xlim(xmin=0)
            ax1.plot(filedata_dat[i1].signals['Wavelength (nm)'],signal, label="frekvence_ref"+bias[-1],linestyle="--",color="red")
            plt.axvline(x=tre_refar[-1],label=str(tre_refar[-1]))
            plt.legend()
            ax1.grid(True,linestyle=':')
            ax1.set_xlim((tre_refar[-1]-30, tre_refar[-1]+60)) 
            ax1.set_ylim((-5, 25)) 
            fig.set_size_inches(4.4,4.4)   
    tre_ref=np.nanmean(np.array(tre_refar))
    #print(tre_refar,"reference array")        
    for i in range(i2,i2+1):
        bias.append(filedata_dat[i].header["Bias (V)"])
        n=float(filedata_dat[i].header["Number of Accumulations"])
        bg=savgol_filter(filedata_dat[i_bg].signals['Counts'],21,0)
        signal_nf=np.subtract(filedata_dat[i].signals['Counts'], bg)
        signal=savgol_filter(np.subtract(filedata_dat[i].signals['Counts'], bg),11,0)
        counts_cor.append(np.subtract(filedata_dat[i].signals['Counts'], bg))
        cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        
        for ij in range(0,len(bg)):
            if signal[ij]>3 and signal_nf[ij+1]>3 and signal_nf[ij+5]>5 and signal_nf[ij+10]>10:
                tre_ref2=(filedata_dat[i1].signals['Wavelength (nm)'][ij])
                print(tre_ref2)
                print(eV/tre_ref2)
                break
            if ij==len(bg-1):
                tre_ref2=(np.NaN)
    for i in range(0,len(tre)):
        if (tre_ref-tre[i])>14:
            dif.append(1000*abs((eV/tre_ref)-(eV/tre[i])))
            print(tre_ref-tre[i],"difference nm")
            amp_cond.append(dif[i]/amp0[i])
        else:
            dif.append(1000*abs((eV/tre_ref)-(eV/tre[i])))
            amp_cond.append(np.NaN)
    print(len(dif))
    print(len(amp0))
    amp=np.divide(np.array(dif),np.array(amp0))   
    f_amp=scipy.signal.medfilt(amp,5)       
    fig, ax1 = plt.subplots() #osy 
    ax1.set_xlabel('Frequency [MHz]')
    ax1.set_ylabel('Transfer function')
    ax1.set_ylim((0.001,1))
    ax1.set_yscale('log')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Amplitude [meV]')  
    #plt.xlim(xmin=0)
    ax1.plot(freq,amp, label=str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"][-5:])+" "+"B "+str("{0:.2f}".format(float(bias[0])))+"V",marker=".",linestyle="None")
 #   ax1.plot(freq,amp_cond, label=">15 nm shift",marker=".",linestyle="None")
    ax2.plot(freq,np.array(dif), label="compensated",linestyle="-",color="red")
   # ax1.set_xticks([freq[0],freq[-1]]) 
    #print(freq)
   # plt.legend()
    #popt, pcov=curve_fit(npower2,cur2,scounts)
    #print(popt[1])
    #ax1.plot(cur2,npower(cur2,popt[0],popt[1],popt[2],popt[3]), label="fit")
    #ax1.plot(cur2,npower2(cur2,popt[0],popt[1]), label="fit")
   # ax1.grid(True,linestyle=':')
    fig.set_size_inches(4.4,4.4)
    f= open(path+str(filedata_dat[i1].header["Filename"])+"cal_function.txt","w+")
    f.write("Frequency [MHz]"+'\t'+"Transfer function [fraction]"+'\t'+"Transfer function filtered [fraction]"+"\n")
    for i in range (0,len(amp)):
        f.write(str("{0:.1f}".format(float(freq[i])))+'\t'+str("{0:.4f}".format(float(amp[i])))+'\t'+str("{0:.4f}".format(float(f_amp[i])))+"\n")
    f.close()
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"r"+"transfer-amp.png", dpi=400, bbox_inches = 'tight') # nazev souboru 

def transfer_tre_new3(i1,i2,i_bg,i_ref,i_ref2,save,biasonly,**kwargs):
    cur2=[]
    scounts=[]
    bias=[]
    bias2=[]
    Z=[]
    freq=[]
    tre=[]
    amp0=[]
    counts_cor=[]
    dif=[]
    amp_cond=[]
    trans_ok=[]
    trans_nk=[]
    tre_refar=[]
    singal_nf_ar=[]
    difcounts_ar=[]
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
    if isfloat(i_ref)==True:
        i_ref=i_ref
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_ref):
                i_ref=j
                break
    if isfloat(i_ref2)==True:
        i_ref2=i_ref2
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_ref2):
                i_ref2=j
                break
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bg):
                i_bg=j
    lr=filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i1].signals['Wavelength (nm)'][-1]       
    ncol=len(filedata_dat[i1].signals['Wavelength (nm)'])-1
    ppnm=ncol/(rr-lr)
    bg0=savgol_filter(filedata_dat[i_bg].signals['Counts'],11,0)
    for i in range(i1, i2+1):
         if float(filedata_dat[i].header["Bias (V)"])==biasonly:
            bias.append(filedata_dat[i].header["Bias (V)"])
            Z.append(float(filedata_dat[i].header["Z (m)"]))
            #if float(filedata_dat[i].header["Bias (V)"])>1.9:
            if float(filedata_dat[i].header["Bias (V)"])>1.3:
                bias2.append(True)
            else:
                bias2.append(False)
            #freq.append(filedata_dat[i].header["GEN FREQ (MHz)"])
            #print(filedata_dat[i].header["genarb frequency (MHz)"],"freq")
            try:
                freq.append(float(filedata_dat[i].header["genarb frequency (MHz)"]))
            except ValueError:
                freq.append(0)
                print(i," ",filedata_dat[i].header["Filename"])
            if "gen_arb" in kwargs:
                if kwargs["gen_arb"]==True:
                    amp0.append(0.5*float(filedata_dat[i].header["genarb power (mV)"]))   
                else:
                    amp0.append(np.sqrt(2)*float(filedata_dat[i].header["genarb power (mV)"]))
            else:
                amp0.append(np.sqrt(2)*float(filedata_dat[i].header["genarb power (mV)"]))
            n=float(filedata_dat[i].header["Number of Accumulations"])
            if np.min(filedata_dat[i].signals['Counts'])<250:
                bg=np.full((len(filedata_dat[i_bg].signals['Counts'])),np.mean(filedata_dat[i].signals['Counts'][0:10]) )
                print(bg)
            else:
                bg=bg0
            signal_nf=np.subtract(filedata_dat[i].signals['Counts'], bg)
            signal=savgol_filter(np.subtract(filedata_dat[i].signals['Counts'], bg),11,0)
            counts_cor.append(np.subtract(filedata_dat[i].signals['Counts'], bg))
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        
            for ij in range(0,len(bg)):
               # if signal[ij]>2 and signal_nf[ij]>2 and signal[min(ij+int(ppnm*1),1023)]>3 and signal_nf[min(ij+int(ppnm*1),1023)]>3 and signal[min(ij+int(ppnm*5),1023)]>7 and signal[min(ij+int(ppnm*10),1023)]>15 and signal[min(ij+60,1023)]>10: #new lower threshold 
                if signal[ij]>1.5 and signal_nf[ij]>2 and signal[min(ij+int(ppnm*1),1023)]>3 and signal_nf[min(ij+int(ppnm*1),1023)]>3 and signal[min(ij+int(ppnm*5),1023)]>5 and signal[min(ij+int(ppnm*10),1023)]>10 and signal[min(ij+60,1023)]>10: #grating 4
                    tre.append(filedata_dat[i1].signals['Wavelength (nm)'][ij])
                   # print(ij,"index")
                    break
                if ij==len(bg-1):
                    tre.append(np.NaN)
                    print(tre,"tre NaN")
          #  print(i, "i",freq[-1],"freq", amp0[-1],"amp0", tre[-1],"tre",len(amp0),"lenamp0",len(tre),"lentre")
           # print(amp0)
         
    for i in range(i_ref,i_ref2+1):
        #print(filedata_dat[i].header["Bias (V)"])
        if float(filedata_dat[i].header["Bias (V)"])==biasonly and "OFF" in filedata_dat[i].header["genarb output"]:
            bias.append(filedata_dat[i].header["Bias (V)"])
            n=float(filedata_dat[i].header["Number of Accumulations"])
            bg=savgol_filter(filedata_dat[i_bg].signals['Counts'],21,0)
            signal_nf=np.subtract(filedata_dat[i].signals['Counts'], bg)
            signal=savgol_filter(np.subtract(filedata_dat[i].signals['Counts'], bg),11,0)
            counts_cor.append(np.subtract(filedata_dat[i].signals['Counts'], bg))
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
            wl=filedata_dat[i1].signals['Wavelength (nm)']
            diff1=savgol_filter(np.diff(signal)/np.diff(wl),31,0)
            diff2=savgol_filter(np.diff(diff1)/np.diff(wl[:-1]),31,0)
            for ij in range(0,len(bg)):
                if signal[ij]>2 and signal_nf[ij]>2 and signal[ij+int(ppnm*1)]>3 and signal[min(ij+int(ppnm*5),1023)]>10 and signal_nf[min(ij+int(ppnm*5),1023)]>15 and signal_nf[min(ij+60,1023)]>10:
                    tre_refar.append(filedata_dat[i1].signals['Wavelength (nm)'][ij])
                    #print(tre_refar[-1])
                    #print(eV/tre_refar[-1])
                    break
                if ij==len(bg-1):
                    tre_refar.append(np.NaN)
            #print(tre_refar,"trerefar")
    
            fig, ax1 = plt.subplots() #osy 
            ax1.set_xlabel('wl [nm]')
            ax1.set_ylabel('signal [a.u.]')
            plt.xlim(xmin=0)
            ax1.plot(filedata_dat[i1].signals['Wavelength (nm)'],signal, label="frekvence_ref"+bias[-1],linestyle="--",color="red")
            ax1.plot(filedata_dat[i1].signals['Wavelength (nm)'][:-1],diff1, label="diff"+bias[-1],linestyle="--",color="blue")
            ax1.plot(filedata_dat[i1].signals['Wavelength (nm)'][:-2],diff2, label="diff2"+bias[-1],linestyle="--",color="black")
            plt.axvline(x=tre_refar[-1],label=str(tre_refar[-1]))
            plt.legend()
            ax1.grid(True,linestyle=':')
            ax1.set_xlim((tre_refar[-1]-30, tre_refar[-1]+60)) 
            ax1.set_ylim((-5, 25)) 
            fig.set_size_inches(4.4,4.4)
            singal_nf_ar.append(signal_nf)
    tre_ref=np.nanmean(np.array(tre_refar))
    print(eV/tre_ref,"treref")
    signal_refmean=np.mean(np.array(singal_nf_ar),axis=0)
    #write reference plasmon
    #print(len(signal_refmean),"lenref")
    f= open(path+str(filedata_dat[i1].header["Filename"])+"ref_plasmon.txt","w+")
    f.write("Energy (eV)"+'\t'+"Counts"+"\n")
    for i in range (0,len(signal_refmean)):
        f.write(str("{0:.4f}".format(float(eV/filedata_dat[i_ref].signals['Wavelength (nm)'][i])))+'\t'+str("{0:.1f}".format(float(signal_refmean[i])))+"\n")
    f.close()
    
    for i in range(i1, i2+1):
        if float(filedata_dat[i].header["Bias (V)"])==biasonly:
            bg=savgol_filter(filedata_dat[i_bg].signals['Counts'],11,0)
            signal_nf=np.subtract(filedata_dat[i].signals['Counts'], bg)
            signal_dif=np.subtract(signal_nf, signal_refmean)
            difcounts=sum(signal_dif[nmtopix(tre_ref-20,lr,rr,ncol):nmtopix(tre_ref+10,lr,rr,ncol)])
            difcounts_ar.append(difcounts)
    for i in range(i2,i2+1):
        bias.append(filedata_dat[i].header["Bias (V)"])
        n=float(filedata_dat[i].header["Number of Accumulations"])
        bg=savgol_filter(filedata_dat[i_bg].signals['Counts'],21,0)
        signal_nf=np.subtract(filedata_dat[i].signals['Counts'], bg)
        signal=savgol_filter(np.subtract(filedata_dat[i].signals['Counts'], bg),11,0)
        counts_cor.append(np.subtract(filedata_dat[i].signals['Counts'], bg))
        cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        
        for ij in range(0,len(bg)):
            if signal[ij]>3 and signal_nf[ij+1]>3 and signal_nf[ij+5]>5 and signal_nf[ij+10]>10:
                tre_ref2=(filedata_dat[i1].signals['Wavelength (nm)'][ij])
                print(tre_ref2)
                print(eV/tre_ref2)
                break
            if ij==len(bg-1):
                tre_ref2=(np.NaN)
    for i in range(0,len(tre)):
        if (tre_ref-tre[i])>14:
            dif.append(1000*abs((eV/tre_ref)-(eV/tre[i])))
            #print(tre_ref-tre[i],"difference nm")
            amp_cond.append(dif[i]/amp0[i])
        else:
            dif.append(1000*abs((eV/tre_ref)-(eV/tre[i])))
            amp_cond.append(np.NaN)
           # print(tre_ref-tre[i],"difference nm")
        if amp0[i]>=3500*np.sqrt(2):
            trans_nk.append(dif[i]/amp0[i])
            trans_ok.append(np.NaN)
        else:
            trans_ok.append(dif[i]/amp0[i])
            trans_nk.append(np.NaN)           
 #   print(len(dif))
   # print(len(amp0))
    rel_dif=np.max(dif)*np.array(difcounts_ar)/np.max(difcounts_ar)
  #  print(rel_dif)
    amp=np.divide(np.array(dif),np.array(amp0))   
    f_amp=scipy.signal.medfilt(amp,5) 
    fig, ax1 = plt.subplots() #osy 
    ax1.set_xlabel('Frequency (MHz)')
    ax1.set_ylabel('Transfer function')
    ax1.set_ylim((0.0005,0.5))
    ax1.set_yscale('log')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Amplitude (meV)')  
 #   ax1.plot(freq,amp, label=str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"][-5:])+" "+"B "+str("{0:.2f}".format(float(bias[0])))+"V",marker=".",linestyle="None")
  #  ax1.plot(freq,amp_cond, label=">15 nm shift",marker=".",linestyle="None")
    ax1.plot(freq,trans_ok, label=str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"][-5:])+" "+"B "+str("{0:.2f}".format(float(bias[0])))+"V",marker=".",linestyle="None",markersize=1.5,color="black")
    ax1.plot(freq,trans_nk, label="notcompensated",marker=".",linestyle="None",markersize=1.5,color="red")
    ax2.plot(freq,np.array(dif), label="compensated",linestyle="-",color="blue",alpha=0.8)
    if "plot_rel" in kwargs:
        if kwargs["plot_rel"]==True:
            ax2.plot(freq,rel_dif, label="compensated",linestyle="-",color="black")
    else:
        pass
    if "plot_Z" in kwargs:
        if kwargs["plot_Z"]==True:
            ax2.plot(freq,np.array(Z)*1E9, label="Z",linestyle="-",color="black")
    else:
        pass   
   # ax1.set_xticks([freq[0],freq[-1]]) 
    #print(freq)
   # plt.legend()
    #popt, pcov=curve_fit(npower2,cur2,scounts)
    #print(popt[1])
    #ax1.plot(cur2,npower(cur2,popt[0],popt[1],popt[2],popt[3]), label="fit")
    #ax1.plot(cur2,npower2(cur2,popt[0],popt[1]), label="fit")
   # ax1.grid(True,linestyle=':')
    if "x1" in kwargs:
       plt.xlim(xmin=float(kwargs["x1"]))
    else:
       pass
    if "x2" in kwargs:
       plt.xlim(xmax=float(kwargs["x2"]))
    else:
       pass
   
    if "y1" in kwargs:
        ax1.set_ylim(bottom=float(kwargs["y1"]))
    else:
        pass
    if "y2" in kwargs:
        ax1.set_ylim(top=float(kwargs["y2"]))
    else:
        pass 
    if "y21" in kwargs:
        ax2.set_ylim(bottom=float(kwargs["y21"]))
    else:
        pass
    if "y22" in kwargs:
        ax2.set_ylim(top=float(kwargs["y22"]))
    else:
        pass 
    fig.set_size_inches(4.4,2.2)
    f= open(path+str(filedata_dat[i1].header["Filename"])+"cal_function.txt","w+")
    f.write("Frequency [MHz]"+'\t'+"Transfer function [fraction]"+'\t'+"Transfer function filtered [fraction]"+"\n")
    for i in range (0,len(amp)):
        f.write(str("{0:.1f}".format(float(freq[i])))+'\t'+str("{0:.4f}".format(float(amp[i])))+'\t'+str("{0:.4f}".format(float(f_amp[i])))+"\n")
    f.close()
    
    f= open(path+str(filedata_dat[i1].header["Filename"])+"cal_function_all2.txt","w+")
    f.write("Frequency [MHz]"+'\t'+"Amplitude tre [mV]"+'\t'+"Rel amp integral [a.u.]"+'\t'+"Transfer function [fraction]"+'\t'+"Transfer function filtered [fraction]"+'\t'+"Z [m]"+"\n")
    for i in range (0,len(amp)):
        f.write(str("{0:.1f}".format(float(freq[i])))+'\t'+str("{0:.3f}".format(float(dif[i])))+'\t'+str("{0:.2f}".format(float(rel_dif[i])))+'\t'+str("{0:.5f}".format(float(amp[i])))+'\t'+str("{0:.4f}".format(float(f_amp[i])))+'\t'+str("{0:.6E}".format(float(Z[i])))+"\n")
    f.close()
    description="bias"+str("{0:.2f}".format(float(biasonly)))+" "+filedata_dat[i_bg].header["Filename"]+" "+filedata_dat[i_ref].header["Filename"]+" "+filedata_dat[i_ref2].header["Filename"]
    metadata={"description":description}
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"r"+"transfer-amp.png", dpi=400, bbox_inches = 'tight',metadata=metadata) # nazev souboru 

def transfer_tre_new3multi(i1,i2,i_bg,i_ref,i_ref2,save,biasonly,**kwargs):
    cur2=[]
    scounts=[]
    bias=[]
    bias2=[]
    freq=[]
    tre=[]
    amp0=[]
    counts_cor=[]
    dif=[]
    amp_cond=[]
    trans_ok=[]
    trans_nk=[]
    tre_refar=[]
    singal_nf_ar=[]
    difcounts_ar=[]
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
    if isfloat(i_ref)==True:
        i_ref=i_ref
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_ref):
                i_ref=j
                break
    if isfloat(i_ref2)==True:
        i_ref2=i_ref2
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_ref2):
                i_ref2=j
                break
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bg):
                i_bg=j
    lr=filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i1].signals['Wavelength (nm)'][-1]       
    ncol=len(filedata_dat[i1].signals['Wavelength (nm)'])-1
    bg0=savgol_filter(filedata_dat[i_bg].signals['Counts'],11,0)
    for i in range(i1, i2+1):
         if float(filedata_dat[i].header["Bias (V)"])==biasonly:
            bias.append(filedata_dat[i].header["Bias (V)"])
            #if float(filedata_dat[i].header["Bias (V)"])>1.9:
            if float(filedata_dat[i].header["Bias (V)"])>1.3:
                bias2.append(True)
            else:
                bias2.append(False)
            #freq.append(filedata_dat[i].header["GEN FREQ (MHz)"])
            freq.append(float(filedata_dat[i].header["genarb frequency (MHz)"]))
            if "gen_arb" in kwargs:
                if kwargs["gen_arb"]==True:
                    amp0.append(0.5*float(filedata_dat[i].header["genarb power (mV)"]))    
            else:
                amp0.append(np.sqrt(2)*float(filedata_dat[i].header["genarb power (mV)"]))
            n=float(filedata_dat[i].header["Number of Accumulations"])
            if np.min(filedata_dat[i].signals['Counts'])<250:
                bg=np.full((len(filedata_dat[i_bg].signals['Counts'])),np.mean(filedata_dat[i].signals['Counts'][0:10]) )
                print(bg)
            else:
                bg=bg0
            signal_nf=np.subtract(filedata_dat[i].signals['Counts'], bg)
            signal=savgol_filter(np.subtract(filedata_dat[i].signals['Counts'], bg),11,0)
            counts_cor.append(np.subtract(filedata_dat[i].signals['Counts'], bg))
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        
            for ij in range(0,len(bg)):
                if signal[ij]>3 and signal_nf[ij+1]>3 and signal_nf[ij+10]>5 and signal_nf[ij+20]>10 and signal_nf[ij+30]>20 and signal_nf[ij+40]>30:
                    tre.append(filedata_dat[i1].signals['Wavelength (nm)'][ij])
                    break
                if ij==len(bg-1):
                    tre.append(np.NaN)
   
    for i in range(i_ref,i_ref2+1):
        #print(filedata_dat[i].header["Bias (V)"])
        if float(filedata_dat[i].header["Bias (V)"])==biasonly and "OFF" in filedata_dat[i].header["genarb output"]:
            bias.append(filedata_dat[i].header["Bias (V)"])
            n=float(filedata_dat[i].header["Number of Accumulations"])
            bg=savgol_filter(filedata_dat[i_bg].signals['Counts'],21,0)
            signal_nf=np.subtract(filedata_dat[i].signals['Counts'], bg)
            signal=savgol_filter(np.subtract(filedata_dat[i].signals['Counts'], bg),11,0)
            counts_cor.append(np.subtract(filedata_dat[i].signals['Counts'], bg))
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
            
            for ij in range(0,len(bg)):
                if signal[ij]>3 and signal_nf[ij+1]>5 and signal_nf[ij+10]>5 and signal_nf[ij+20]>10 and signal_nf[ij+40]>30:
                    tre_refar.append(filedata_dat[i1].signals['Wavelength (nm)'][ij])
                    print(tre_refar[-1])
                    print(eV/tre_refar[-1])
                    break
                if ij==len(bg-1):
                    tre_refar.append(np.NaN)
    
    tre_ref=np.nanmean(np.array(tre_refar))
    signal_refmean=np.mean(np.array(singal_nf_ar),axis=0)
    bg=savgol_filter(filedata_dat[i_bg].signals['Counts'],11,0)
    for i in range(i1, i2+1):
        if float(filedata_dat[i].header["Bias (V)"])==biasonly:
            signal_nf=np.subtract(filedata_dat[i].signals['Counts'], bg)
            signal_dif=np.subtract(signal_nf, signal_refmean)
    for i in range(i2,i2+1):
        bias.append(filedata_dat[i].header["Bias (V)"])
        n=float(filedata_dat[i].header["Number of Accumulations"])
        bg=savgol_filter(filedata_dat[i_bg].signals['Counts'],21,0)
        signal_nf=np.subtract(filedata_dat[i].signals['Counts'], bg)
        signal=savgol_filter(np.subtract(filedata_dat[i].signals['Counts'], bg),11,0)
        counts_cor.append(np.subtract(filedata_dat[i].signals['Counts'], bg))
        cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        
        for ij in range(0,len(bg)):
            if signal[ij]>3 and signal_nf[ij+1]>3 and signal_nf[ij+5]>5 and signal_nf[ij+10]>10:
                tre_ref2=(filedata_dat[i1].signals['Wavelength (nm)'][ij])
                print(tre_ref2)
                print(eV/tre_ref2)
                break
            if ij==len(bg-1):
                tre_ref2=(np.NaN)
    for i in range(0,len(tre)):
        if (tre_ref-tre[i])>14:
            dif.append(1000*abs((eV/tre_ref)-(eV/tre[i])))
            print(tre_ref-tre[i],"difference nm")
            amp_cond.append(dif[i]/amp0[i])
        else:
            dif.append(1000*abs((eV/tre_ref)-(eV/tre[i])))
            amp_cond.append(np.NaN)
        if amp0[i]>=3500*np.sqrt(2):
            trans_nk.append(dif[i]/amp0[i])
            trans_ok.append(np.NaN)
        else:
            trans_ok.append(dif[i]/amp0[i])
            trans_nk.append(np.NaN)           
    print(len(dif))
    print(len(amp0))
    amp=np.divide(np.array(dif),np.array(amp0))   
    f_amp=scipy.signal.medfilt(amp,5) 
    fig, ax1 = plt.subplots() #osy 
    ax1.set_xlabel('Frequency (MHz)')
    ax1.set_ylabel('Transfer function')
    ax1.set_ylim((0.0005,0.5))
    ax1.set_yscale('log')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Amplitude (meV)')  
 #   ax1.plot(freq,amp, label=str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"][-5:])+" "+"B "+str("{0:.2f}".format(float(bias[0])))+"V",marker=".",linestyle="None")
  #  ax1.plot(freq,amp_cond, label=">15 nm shift",marker=".",linestyle="None")
    ax1.plot(freq,trans_ok, label=str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"][-5:])+" "+"B "+str("{0:.2f}".format(float(bias[0])))+"V",marker=".",linestyle="None",markersize=1.5,color="black")
    ax1.plot(freq,trans_nk, label="notcompensated",marker=".",linestyle="None",markersize=1.5,color="red")
    #ax2.plot(freq,np.array(dif), label="compensated",linestyle="-",color="blue",alpha=0.8)

    if "x1" in kwargs:
       plt.xlim(xmin=float(kwargs["x1"]))
    else:
       pass
    if "x2" in kwargs:
       plt.xlim(xmax=float(kwargs["x2"]))
    else:
       pass
   
    if "y1" in kwargs:
        ax1.set_ylim(bottom=float(kwargs["y1"]))
    else:
        pass
    if "y2" in kwargs:
        ax1.set_ylim(top=float(kwargs["y2"]))
    else:
        pass 
    if "y21" in kwargs:
        ax2.set_ylim(bottom=float(kwargs["y21"]))
    else:
        pass
    if "y22" in kwargs:
        ax2.set_ylim(top=float(kwargs["y22"]))
    else:
        pass 
    fig.set_size_inches(4.4,2.2)
  #  f= open(path+str(filedata_dat[i1].header["Filename"])+"cal_function.txt","w+")
  #  f.write("Frequency [MHz]"+'\t'+"Transfer function [fraction]"+'\t'+"Transfer function filtered [fraction]"+"\n")
  #  for i in range (0,len(amp)):
  #      f.write(str("{0:.1f}".format(float(freq[i])))+'\t'+str("{0:.4f}".format(float(amp[i])))+'\t'+str("{0:.4f}".format(float(f_amp[i])))+"\n")
  #  f.close()
    
 #   f= open(path+str(filedata_dat[i1].header["Filename"])+"cal_function_all.txt","w+")
#    f.write("Frequency [MHz]"+'\t'+"Amplitude tre [mV]"+'\t'+"Rel amp integral [a.u.]"+'\t'+"Transfer function [fraction]"+'\t'+"Transfer function filtered [fraction]"+"\n")
  #  for i in range (0,len(amp)):
 #       f.write(str("{0:.1f}".format(float(freq[i])))+'\t'+str("{0:.2f}".format(float(dif[i])))+'\t'+str("{0:.2f}".format(float(rel_dif[i])))+'\t'+str("{0:.4f}".format(float(amp[i])))+'\t'+str("{0:.4f}".format(float(f_amp[i])))+"\n")
 #   f.close()
    description="bias"+str("{0:.2f}".format(float(biasonly)))+" "+filedata_dat[i_bg].header["Filename"]+" "+filedata_dat[i_ref].header["Filename"]+" "+filedata_dat[i_ref2].header["Filename"]
    metadata={"description":description}
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"r"+"transfer-amp.png", dpi=400, bbox_inches = 'tight',metadata=metadata) # nazev souboru 
 
def cal_plasm(i1,i2,i_bg,i_ref,x1,x2,save):
    cur2=[]
    scounts=[]
    bias=[]
    tre=[]
    tre_diff=[]
    counts_cor=[]
    shift=-2.0 #632.8nm =624.8 wl with this calibration i e shift 8 but 
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
    if isfloat(i_ref)==True:
        i_ref=i_ref
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_ref):
                i_ref=j
                break
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bg):
                i_bg=j
    lr=filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i1].signals['Wavelength (nm)'][-1]       
    ncol=len(filedata_dat[i1].signals['Wavelength (nm)'])-1

    for i in range(0, len(file_numbers)):
        if "vplasmn" in file_names[i] and "mod" not in file_names[i]  and abs(float(filedata_dat[i].header["Bias (V)"]))>1.37 and float(filedata_dat[i].header["Bias (V)"])<2.8:  #and float(filedata_dat[i].header["Bias (V)"])>1.55
            bias.append(float(filedata_dat[i].header["Bias (V)"]))
            n=float(filedata_dat[i].header["Number of Accumulations"])
            bg=savgol_filter(filedata_dat[i_bg].signals['Counts'],5,0)
            signal_nf=np.subtract(filedata_dat[i].signals['Counts'], bg)
            signal=savgol_filter(np.subtract(filedata_dat[i].signals['Counts'], bg),11,0)
            counts_cor.append(np.subtract(filedata_dat[i].signals['Counts'], bg))
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        
            for ij in range(0,len(bg)):
                if signal[ij]>10 and signal_nf[ij+1]>10 and signal_nf[ij+5]>10 and signal_nf[ij+10]>10:
                #if signal[ij]>5 and signal_nf[ij+1]>5 and signal_nf[ij+5]>5 and signal_nf[ij+10]>10  and abs(bias[-1])<1.9:
                    tre.append(filedata_dat[i1].signals['Wavelength (nm)'][ij])
                    treij=ij
                    break
                #if signal[ij]>5 and signal_nf[ij+1]>5 and signal_nf[ij+5]>20 and signal_nf[ij+10]>30 and (signal_nf[ij-2]-signal_nf[ij-7])<5*(signal_nf[ij+7]-signal_nf[ij+4]) and abs(bias[-1])>1.89 and bias[-1]<2.1 and signal_nf[ij+10]>60:
                   # tre.append(filedata_dat[i1].signals['Wavelength (nm)'][ij])
                  #  treij=ij
                 #   break      
                #if signal[ij]>5 and signal_nf[ij+1]>20 and signal_nf[ij+5]>20 and signal_nf[ij+10]>30 and (signal_nf[ij-2]-signal_nf[ij-7])<4*(signal_nf[ij+7]-signal_nf[ij+4]) and abs(bias[-1])>2.09 and signal_nf[ij+10]>120:
               #     tre.append(filedata_dat[i1].signals['Wavelength (nm)'][ij])
                #    treij=ij
               #     break
                if ij==len(bg-1):
                    tre.append(np.NaN)
                    treij=0
                    break   
            #print(np.amax(np.diff(signal[treij-20:treij+20])))
            #print(np.argmax(np.diff(signal[treij-20:treij+20])))
           # tre_diff.append(filedata_dat[i1].signals['Wavelength (nm)'][ij-20+np.argmax(np.diff(signal[treij-20:treij+20]))])
            #print(signal.index(max(np.diff(signal[treij-10,treij+10]))))
            dif_signal=np.diff(signal)
            fig, ax1 = plt.subplots() #osy 
            ax1.set_xlabel('wl [nm]')
            ax1.set_ylabel('signal [a.u.]')
            plt.xlim(xmin=0)
            ax1.plot(filedata_dat[i1].signals['Wavelength (nm)'],signal, label=str(bias[-1]),linestyle="--")
            ax1.plot(filedata_dat[i1].signals['Wavelength (nm)'][0:1022],10*dif_signal, label=str(bias[-1]),linestyle="-")
            plt.legend()
            ax1.grid(True,linestyle=':')
            ax1.set_xlim((abs(eV/bias[-1])-20, abs(eV/bias[-1])+20)) 
            ax1.set_ylim((-5, 100)) 
            fig.set_size_inches(4.4,4.4)
            
            if abs(bias[-1])>1.3:
                for ik in range(treij-12,treij+12):
                   # print(dif_signal[ik], "derivace", ik)
                    if dif_signal[ik]>1.8:
                        #print(tre_diff[-1],"tre_dif last old")
                        #tre[-1]=filedata_dat[i1].signals['Wavelength (nm)'][ik]
                        #print("wl",filedata_dat[i1].signals['Wavelength (nm)'][ik])
                        #print(tre_diff[-1],"tre_dif last new")
                        break
                    if ik==treij+19:
                        break
           # plt.axvline(x=tre_diff[-1],color='r')   
            plt.axvline(x=tre[-1])
    tre=np.array(tre)
    bias_ar=abs(np.array(bias))
    fig, ax1 = plt.subplots() #osy 
    ax1.set_xlabel('Bias')
    ax1.set_ylabel('Treshold [nm]')
    ax1.plot(eV/bias_ar,tre+shift, label=str(filedata_dat[i1].header["Filename"])+" "+str(filedata_dat[i2].header["Filename"])+" "+"B "+str(filedata_dat[i1].header["Bias (V)"]),marker=".",linestyle="None")
    plt.legend()
    #print(popt[1])
    #ax1.plot(cur2,npower(cur2,popt[0],popt[1],popt[2],popt[3]), label="fit")
    #ax1.plot(cur2,npower2(cur2,popt[0],popt[1]), label="fit")
    ax1.grid(True,linestyle=':')
    fig.set_size_inches(4.4,4.4)
    ax1.set_xlim((x1,x2)) 
    popt, pcov=curve_fit(linfunc,eV/bias_ar,tre-shift)
    print(popt[0],"a",popt[1],"b")
    print("old lr:",lr," new lr:",linfunc(lr,popt[0],popt[1]))
    print("old rr:",rr," new rr:",linfunc(rr,popt[0],popt[1]))
    poptc, pcovc=curve_fit(cubfunc,eV/bias_ar,tre-shift)
    poptcr, pcovcr=curve_fit(cubfunc,tre-shift,eV/bias_ar)
    print(poptc[0],"a",poptc[1],"b",poptc[2],"c",poptc[3],"d")
    print("old lr:",lr," new lr:",cubfunc(lr,poptcr[0],poptcr[1],poptcr[2],poptcr[3]))
    print("old rr:",rr," new rr:",cubfunc(rr,poptcr[0],poptcr[1],poptcr[2],poptcr[3]))
    
    print("840nm:",840," new value:",cubfunc(840,poptcr[0],poptcr[1],poptcr[2],poptcr[3]))
    print("830nm:",830," new value:",cubfunc(830,poptcr[0],poptcr[1],poptcr[2],poptcr[3]))
    fig, ax1 = plt.subplots() #osy 
    ax1.set_xlabel('Bias')
    ax1.set_ylabel('Treshold [nm]')
    ax1.plot(eV/bias_ar,eV/bias_ar-tre+shift, label="old calibration",marker=".",linestyle="None")
    ax1.plot(eV/bias_ar,eV/bias_ar-linfunc(eV/bias_ar,popt[0],popt[1]), label="new calibration",marker=".",linestyle="None")
    ax1.plot(eV/bias_ar,eV/bias_ar-cubfunc(eV/bias_ar,poptc[0],poptc[1],poptc[2],poptc[3]), label="new calibration cubic",marker=".",linestyle="None")
    plt.legend()
    #print(popt[1])
    #ax1.plot(cur2,npower(cur2,popt[0],popt[1],popt[2],popt[3]), label="fit")
    #ax1.plot(cur2,npower2(cur2,popt[0],popt[1]), label="fit")
    ax1.grid(True,linestyle=':')
    fig.set_size_inches(4.4,4.4)
    ax1.set_xlim((x1,x2))
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"calibration.png", dpi=400, bbox_inches = 'tight') # nazev souboru 
    
def noise_amplitude(i1):
    if isfloat(i1)==True:
        i1=i1
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i1):
                i1=j
                break
    return(np.std(filedata_dat[i1].signals['Counts']))
def print_apd():
    f= open(path+"apd_parameters.txt","w+")
    f.write("Bias [V]"+'\t'+"Current [pA]"+ '\t'+"Z [nm]"+'\t'+"Modulation [mV]"+'\t'+"Filename"+"\n")
    for i in range(0,len(file_names)):
        if "apd" or "vdep" in file_names[i]:
            bias=float(filedata_dat[i].header["Bias (V)"])
            
            f.write('{:.2f}'.format(bias)+'\t')
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"])*1E12)
            Z=abs(float(filedata_dat[i].header["Z avg. (m)"])*1E9)
            print(cur)
            f.write('{:.0f}'.format(cur)+'\t')
            f.write('{:.4f}'.format(Z)+'\t')
            try:
                f.write(filedata_dat[i].header["genarb power (mV)"]+'\t')
            except:
                KeyError
            f.write(file_names[i][:-4]+'\t')
            f.write("\n")
    f.close()
def plot_plasm(i1,i2,i_bg,i_ref,x1,x2,save,expr,log,unit):
    cur2=[]
    scounts=[]
    bias=[]
    tre=[]
    tre_diff=[]
    counts_cor=[]
    name=[]
    shift=-2.0 #632.8nm =624.8 wl with this calibration i e shift 8 but 
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
    if isfloat(i_ref)==True:
        i_ref=i_ref
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_ref):
                i_ref=j
                break
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bg):
                i_bg=j
    lr=filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i1].signals['Wavelength (nm)'][-1]       
    ncol=len(filedata_dat[i1].signals['Wavelength (nm)'])-1
    fig, ax1 = plt.subplots() #osy 
    ax1.set_xlabel('Plasmon energy [eV]')
    ax1.set_ylabel('Intensity [counts]')

    for i in range(i1, i2):
        if expr in file_names[i] and "mod" not in file_names[i] and abs(float(filedata_dat[i].header["Bias (V)"]))>1.3 and abs(float(filedata_dat[i].header["Bias (V)"]))<2.89:
            name.append(file_names[i])#and float(filedata_dat[i].header["Bias (V)"])>1.55
            bias.append(float(filedata_dat[i].header["Bias (V)"]))
            n=float(filedata_dat[i].header["Number of Accumulations"])
            bg=savgol_filter(filedata_dat[i_bg].signals['Counts'],5,0)
            signal_nf=np.subtract(filedata_dat[i].signals['Counts'], bg)
            signal=savgol_filter(np.subtract(filedata_dat[i].signals['Counts'], bg),11,0)
            counts_cor.append(np.subtract(filedata_dat[i].signals['Counts'], bg))
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
         
            #print(np.amax(np.diff(signal[treij-20:treij+20])))
            #print(np.argmax(np.diff(signal[treij-20:treij+20])))
           # tre_diff.append(filedata_dat[i1].signals['Wavelength (nm)'][ij-20+np.argmax(np.diff(signal[treij-20:treij+20]))])
            #print(signal.index(max(np.diff(signal[treij-10,treij+10]))))
            dif_signal=np.diff(signal)
            plt.xlim(xmin=0)
            if unit=="nm":
                ax1.plot(filedata_dat[i1].signals['Wavelength (nm)'],signal, label=str(bias[-1]),linestyle="--")
            else:
                ax1.plot(eV/filedata_dat[i1].signals['Wavelength (nm)'],signal, label=str(bias[-1]),linestyle="--")
            #ax1.plot(filedata_dat[i1].signals['Wavelength (nm)'][0:1022],10*dif_signal, label=str(bias[-1]),linestyle="-")
 #   unit="eV"
    if unit=="nm":
        ax1.set_xlabel('Wavelength [nm]')
        ax1.set_xlim((x1, x2))
        ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax2 = ax1.twiny()
        # get the primary axis x tick locations in plot units
        x1nm=math.floor(E(x1)*10)/10
        x2nm=math.ceil(E(x2)*10)/10
        x1nm_min=math.floor(E(x1)*50)/50
        x2nm_min=math.ceil(E(x2)*50)/50             
        xtl=np.linspace(x1nm,x2nm,int(round(abs(x2nm-x1nm)*10))+1)
        print(xtl)
        xtl_min=np.linspace(x1nm_min,x2nm_min,int(round(abs(x2nm_min-x1nm_min)*50))+1)
        print(xtl_min)
        xtickloc=[WL(x) for x in xtl]
        xtickloc_min=[WL(x) for x in xtl_min]
        #xtickloc = ax1.get_xticks() 
        #print(xtl)
        # set the second axis ticks to the same locations
        ax2.set_xticks(xtickloc)
        ax2.set_xticks(xtickloc_min, minor=True)
        # calculate new values for the second axis tick labels, format them, and set them
        x2labels = ['{:.3g}'.format(E(x)) for x in xtickloc]
        ax2.set_xticklabels(x2labels)
        # force the bounds to be the same
        ax2.set_xlim(ax1.get_xlim()) 
        ax2.set_xlabel('Energy [eV]') 
    
        #plt.legend()
        ax1.grid(True,linestyle=':')
    #    ax1.set_xlim(eV/rr, eV/lr)
    else:

        ax1.set_xlabel('Energy [eV]')
        ax1.set_xlim((E(x2), E(x1)))
        ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax2 = ax1.twiny()
        # get the primary axis x tick locations in plot units
        maj_int=100
        min_int=20
        x1nm=math.ceil((x1)/maj_int)*maj_int
        x2nm=math.floor((x2)/maj_int)*maj_int
        x1nm_min=math.ceil(x1/min_int)*min_int
        x2nm_min=math.floor(x2/min_int)*min_int 
        print(x1nm)
        print(x2nm)
        print(x1nm_min)
        print(x2nm_min)
        xtl=np.linspace(x1nm,x2nm,int(round(abs(x2nm-x1nm)/maj_int))+1)
        print(xtl)
        xtl_min=np.linspace(x1nm_min,x2nm_min,int(round(abs(x2nm_min-x1nm_min)/min_int))+1)
        print(xtl_min)
        xtickloc=[E(x) for x in xtl]
        xtickloc_min=[E(x) for x in xtl_min]
        #xtickloc = ax1.get_xticks() 
        #print(xtl)
        # set the second axis ticks to the same locations
        ax2.set_xticks(xtickloc)
        ax2.set_xticks(xtickloc_min, minor=True)
        # calculate new values for the second axis tick labels, format them, and set them
        x2labels = ['{:.0f}'.format(WL(x)) for x in xtickloc]
        ax2.set_xticklabels(x2labels)
        # force the bounds to be the same
        ax2.set_xlim(ax1.get_xlim()) 
        ax2.set_xlabel('Wavelength [nm]')       
    if log==True:
        ax1.set_yscale('log')
        plt.ylim(ymin=1)
            #ax1.set_ylim((-5, 100)) 
    fig.set_size_inches(5,5*3/4)
    if save==True:
        if log==True:
            plt.savefig(path+str(name[0])+"-"+str(name[-1])+"plasmons-log.png", dpi=400, bbox_inches = 'tight',transparent=True) # nazev souboru 
        else:
            plt.savefig(path+str(name[0])+"-"+str(name[-1])+"plasmons-lin.png", dpi=400, bbox_inches = 'tight',transparent=True) # nazev souboru 
            
def getcmap():
    r=[]
    g=[]
    b=[]
    cmap = matplotlib.cm.get_cmap('afmhot')
    f= open(pathcopy+"cmap_afmhot.txt","w+")
    f.write("i\tr\tg\tb\ta\t\n")
    for i in np.arange(0, 1.001, 0.001):
        rgba = cmap(i)
        r.append(rgba[0])
        g.append(rgba[1])
        b.append(rgba[2])
        f.write(str(i)+"\t")
        for j in range(0,len(rgba)):
            f.write(str("{0:.5f}".format(rgba[j]))+"\t")
        f.write("\n")
        print(i,rgba) # (0.99807766255210428, 0.99923106502084169, 0.74602077638401709, 1.0)
    fig, ax1 = plt.subplots()
    ax1.plot(np.arange(0, 1.001, 0.001),r,color="red")
    ax1.plot(np.arange(0, 1.001, 0.001),g)
    ax1.plot(np.arange(0, 1.001, 0.001),b)
    ax1.vline()
    f.close()
 
def integrate(i1,x1,x2):
       
    if isfloat(i1)==True:
        i1=i1
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i1):
                i1=j
                break
    lr=filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i1].signals['Wavelength (nm)'][-1]       
    ncol=len(filedata_dat[i1].signals['Wavelength (nm)'])-1

    scounts=(sum(filedata_dat[i1].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)])-filedata_dat[i1].signals['Counts'][0]*len(filedata_dat[i1].signals['Counts'][nmtopix(x1,lr,rr,ncol):nmtopix(x2,lr,rr,ncol)]))
    return scounts



def transfer_tre_new4(i1,i2,i_bg,i_ref,i_ref2,save,biasonly,**kwargs):
    cur2=[]
    scounts=[]
    bias=[]
    bias2=[]
    lockin=[]
    tre=[]
    amp0=[]
    counts_cor=[]
    dif=[]
    amp_cond=[]
    tre_refar=[]
    singal_nf_ar=[]
    difcounts_ar=[]
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
    if isfloat(i_ref)==True:
        i_ref=i_ref
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_ref):
                i_ref=j
                break
    if isfloat(i_ref2)==True:
        i_ref2=i_ref2
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_ref2):
                i_ref2=j
                break
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bg):
                i_bg=j
    lr=filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i1].signals['Wavelength (nm)'][-1]       
    ncol=len(filedata_dat[i1].signals['Wavelength (nm)'])-1
    ppnm=ncol/(rr-lr)
    bg0=savgol_filter(filedata_dat[i_bg].signals['Counts'],11,0)
    bg=np.full((len(filedata_dat[i_bg].signals['Counts'])),np.mean(filedata_dat[i_bg].signals['Counts']) )
    #print(bg0)
    fig, ax1 = plt.subplots() #osy 
    for i in range(i1, i2+1):
         if float(filedata_dat[i].header["Bias (V)"])==biasonly:
           # print(filedata_dat[i].signals['Counts'])
           # print(filedata_dat[i].header['Filename'])
           # print(len(filedata_dat[i].signals['Counts']))
            bias.append(filedata_dat[i].header["Bias (V)"])
            #if float(filedata_dat[i].header["Bias (V)"])>1.9:
            if float(filedata_dat[i].header["Bias (V)"])>1.3:
                bias2.append(True)
            else:
                bias2.append(False)
            #lockin.append(filedata_dat[i].header["GEN lockin (MHz)"])
            #lockin.append(float(filedata_dat[i].header["Lockin amplitude (mV)"]))
            #amp0.append(float(filedata_dat[i].header["Lockin amplitude (mV)"]))
            lockin.append(float(filedata_dat[i].header["Lockin amplitude (V)"])*1000)
            amp0.append(float(filedata_dat[i].header["Lockin amplitude (V)"])*1000)
            n=float(filedata_dat[i].header["Number of Accumulations"])
            if np.min(filedata_dat[i].signals['Counts'])<250:
                bg=np.full((len(filedata_dat[i_bg].signals['Counts'])),np.mean(filedata_dat[i].signals['Counts'][0:10]) )
                #print(bg,"bg")
                #print(len(bg))
            else:
                bg=bg0
            signal_nf=np.subtract(filedata_dat[i].signals['Counts'], bg)
            signal=savgol_filter(np.subtract(filedata_dat[i].signals['Counts'], bg),11,0)
            counts_cor.append(np.subtract(filedata_dat[i].signals['Counts'], bg))
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        
            for ij in range(0,len(bg)):
                if signal[ij]>3 and signal_nf[ij]>3 and signal[ij+int(ppnm*1)]>5 and signal[ij+int(ppnm*5)]>10 and signal_nf[ij+int(ppnm*10)]>20 and signal_nf[ij+60]>10:
                #if signal[ij]>10 and signal_nf[ij+1]>10 and signal_nf[ij+10]>15 and signal_nf[ij+20]>20 and signal_nf[ij+30]>30 and signal_nf[ij+40]>50: #for old camera
                    tre.append(filedata_dat[i1].signals['Wavelength (nm)'][ij])
                    break
                if ij==len(bg-1):
                    tre.append(np.NaN)
            if int(lockin[-1]) in set([0,10,20,50,100,200,300,400,500]):
                ax1.plot(filedata_dat[i1].signals['Wavelength (nm)'],signal_nf,linestyle="-",linewidth=0.7)  
                plt.axvline(x=tre[-1],linewidth=0.7,color="black",linestyle="--")
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Photon intensity [counts]')
    #plt.xlim(xmin=0)


    #plt.legend()
    #ax1.grid(True,linestyle=':')
    if "x1" in kwargs:
       plt.xlim(xmin=float(kwargs["x1"]))
    else:
       plt.xlim(xmin=480)
    if "x2" in kwargs:
       plt.xlim(xmax=float(kwargs["x2"]))
    else:
       plt.xlim(xmax=tre[0]+30) 
    if "log" in kwargs:
        if "True" in kwargs["log"]:
            ax1.set_yscale('log')
            plt.ylim(ymin=1)
        if "y2" in kwargs:
            plt.ylim(ymax=float(kwargs["y2"]))
    else:
        if "y1" in kwargs:
            plt.ylim(ymin=float(kwargs["y1"]))
        else:
            plt.ylim(ymin=0)
        if "y2" in kwargs:
            plt.ylim(ymax=float(kwargs["y2"]))
        else:
            plt.ylim(ymax=50) 
    x1,x2 = ax1.get_xlim()
    ax2 = ax1.twiny()
    # get the primary axis x tick locations in plot units
    x1nm=math.floor(E(x1)*10)/10
    x2nm=math.ceil(E(x2)*10)/10
    x1nm_min=math.floor(E(x1)*50)/50
    x2nm_min=math.ceil(E(x2)*50)/50             
    xtl=np.linspace(x1nm,x2nm,int(round(abs(x2nm-x1nm)*10))+1)
    print(xtl)
    xtl_min=np.linspace(x1nm_min,x2nm_min,int(round(abs(x2nm_min-x1nm_min)*50))+1)
    print(xtl_min)
    xtickloc=[WL(x) for x in xtl]
    xtickloc_min=[WL(x) for x in xtl_min]
    #xtickloc = ax1.get_xticks() 
    #print(xtl)
    # set the second axis ticks to the same locations
    ax2.set_xticks(xtickloc)
    ax2.set_xticks(xtickloc_min, minor=True)
    # calculate new values for the second axis tick labels, format them, and set them
    x2labels = ['{:.3g}'.format(E(x)) for x in xtickloc]
    ax2.set_xticklabels(x2labels)
    # force the bounds to be the same
    ax2.set_xlim(ax1.get_xlim()) 
    ax2.set_xlabel('Photon energy (eV)')  
    fig.set_size_inches(4.4,3.4)
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"r"+"lockin.png", dpi=400, bbox_inches = 'tight',transparent=True) # nazev souboru 
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"r"+"lockin.svg", dpi=400, bbox_inches = 'tight',transparent=True)
    for i in range(i_ref,i_ref2+1):
        #print(filedata_dat[i].header["Bias (V)"])
        if float(filedata_dat[i].header["Bias (V)"])==biasonly:
            bias.append(filedata_dat[i].header["Bias (V)"])
            n=float(filedata_dat[i].header["Number of Accumulations"])
            bg=savgol_filter(filedata_dat[i_bg].signals['Counts'],21,0)
            signal_nf=np.subtract(filedata_dat[i].signals['Counts'], bg)
            signal=savgol_filter(np.subtract(filedata_dat[i].signals['Counts'], bg),11,0)
            counts_cor.append(np.subtract(filedata_dat[i].signals['Counts'], bg))
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
            
            for ij in range(0,len(bg)):
                if signal[ij]>3 and signal_nf[ij]>3 and signal[ij+int(ppnm*1)]>5 and signal[ij+int(ppnm*5)]>10 and signal_nf[ij+int(ppnm*10)]>20 and signal_nf[ij+60]>10:
                    tre_refar.append(filedata_dat[i1].signals['Wavelength (nm)'][ij])
                    print(tre_refar[-1])
                    print(eV/tre_refar[-1])
                    break
                if ij==len(bg-1):
                    tre_refar.append(np.NaN)
    
            fig, ax1 = plt.subplots() #osy 
            ax1.set_xlabel('wl [nm]')
            ax1.set_ylabel('signal [a.u.]')
            plt.xlim(xmin=0)
            ax1.plot(filedata_dat[i1].signals['Wavelength (nm)'],signal, label="frekvence_ref"+bias[-1],linestyle="--",color="red")
            plt.axvline(x=tre_refar[-1],label=str(tre_refar[-1]))
            plt.legend()
            ax1.grid(True,linestyle=':')
            ax1.set_xlim((tre_refar[-1]-30, tre_refar[-1]+60)) 
            ax1.set_ylim((-5, 25)) 
            fig.set_size_inches(4.4,4.4)
            singal_nf_ar.append(signal_nf)
    tre_ref=np.nanmean(np.array(tre_refar))
    signal_refmean=np.mean(np.array(singal_nf_ar),axis=0)
    for i in range(i1, i2+1):
        if float(filedata_dat[i].header["Bias (V)"])==biasonly:
            bg=savgol_filter(filedata_dat[i_bg].signals['Counts'],11,0)
            signal_nf=np.subtract(filedata_dat[i].signals['Counts'], bg)
            signal_dif=np.subtract(signal_nf, signal_refmean)
            print(tre_ref)
            difcounts=sum(signal_dif[nmtopix(tre_ref-20,lr,rr,ncol):nmtopix(tre_ref+10,lr,rr,ncol)])
            difcounts_ar.append(difcounts)
    for i in range(i2,i2+1):
        bias.append(filedata_dat[i].header["Bias (V)"])
        n=float(filedata_dat[i].header["Number of Accumulations"])
        bg=savgol_filter(filedata_dat[i_bg].signals['Counts'],21,0)
        signal_nf=np.subtract(filedata_dat[i].signals['Counts'], bg)
        signal=savgol_filter(np.subtract(filedata_dat[i].signals['Counts'], bg),11,0)
        counts_cor.append(np.subtract(filedata_dat[i].signals['Counts'], bg))
        cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        
        for ij in range(0,len(bg)):
            if signal[ij]>3 and signal_nf[ij]>3 and signal[ij+int(ppnm*1)]>5 and signal[ij+int(ppnm*5)]>10 and signal_nf[ij+int(ppnm*10)]>20 and signal_nf[ij+60]>10:
                tre_ref2=(filedata_dat[i1].signals['Wavelength (nm)'][ij])
                print(tre_ref2,"trefer",i2-i,"index")
                print(eV/tre_ref2)
                break
            if ij==len(bg-1):
                tre_ref2=(np.NaN)
    for i in range(0,len(tre)):
        if (tre_ref-tre[i])>14:
            dif.append(1000*abs((eV/tre_ref)-(eV/tre[i])))
            print(tre_ref-tre[i],"difference nm")
            amp_cond.append(dif[i]/amp0[i])
        else:
            dif.append(1000*abs((eV/tre_ref)-(eV/tre[i])))
            amp_cond.append(np.NaN)
    print(len(dif))
    print(len(amp0))
    rel_dif=np.max(dif)*np.array(difcounts_ar)/np.max(difcounts_ar)
    print(rel_dif)
    amp=np.divide(np.array(dif),np.array(amp0))   #transfer function from edge
    amp_area=np.divide(np.array(rel_dif),np.array(amp0))
    f_amp=scipy.signal.medfilt(amp,5)
    fig, ax1 = plt.subplots() #osy 
    ax1.set_xlabel('Bias modulation (mV)')
    ax1.set_ylabel('Transfer function')
   # ax1.set_ylim((0.001,1))
    #ax1.set_yscale('log')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Amplitude (mV)')  
    #plt.xlim(xmin=0)
    ax1.plot(lockin,amp, label="T. func.",marker=",",linestyle="None")
    #ax1.plot(lockin[10:],amp_area[10:], label="Transfer area",marker=".",linestyle="None")
    ax2.plot(lockin,np.array(dif), label="Amp. tre.",linestyle="-",color="red")
    ax2.plot(lockin,rel_dif, label="Amp. area",linestyle="-",color="black")
    popt, pcov=curve_fit(fcub,lockin,rel_dif)
    print(popt)
    #fit=fcub(lockin,popt[0],popt[1],popt[2],popt[3])
    #rel_dif_fit=np.max(dif)*np.array(fit)/np.max(difcounts_ar)
    #ax2.plot(lockin,fcub(np.array(lockin),popt[0],popt[1],popt[2],popt[3]), label="amp rel fit",linestyle="--",color="black")
    #ax2.plot(lockin,rel_dif, label="amp rel area",linestyle="-",color="black")
   # ax1.set_xticks([lockin[0],lockin[-1]]) 
    #print(lockin)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    #ax1.legend(h1+h2, l1+l2,loc='upper center')
    #popt, pcov=curve_fit(npower2,cur2,scounts)
    #print(popt[1])
    #ax1.plot(cur2,npower(cur2,popt[0],popt[1],popt[2],popt[3]), label="fit")
    #ax1.plot(cur2,npower2(cur2,popt[0],popt[1]), label="fit")
   # ax1.grid(True,linestyle=':')
    fig.set_size_inches(2.2,2.2)
    f= open(path+str(filedata_dat[i1].header["Filename"])+"cal_function.txt","w+")
    f.write("Lockin [mV]"+'\t'+"Transfer function [fraction]"+'\t'+"Transfer function filtered [fraction]"+"\n")
    for i in range (0,len(amp)):
        f.write(str("{0:.1f}".format(float(lockin[i])))+'\t'+str("{0:.4f}".format(float(amp[i])))+'\t'+str("{0:.4f}".format(float(f_amp[i])))+"\n")
    f.close()
    
    f= open(path+str(filedata_dat[i1].header["Filename"])+"cal_function_all.txt","w+")
    f.write("Lockin [mV]"+'\t'+"Amplitude tre [mV]"+'\t'+"Rel amp integral [a.u.]"+'\t'+"Transfer function [fraction]"+'\t'+"Transfer function filtered [fraction]"+"\n")
    for i in range (0,len(amp)):
        f.write(str("{0:.1f}".format(float(lockin[i])))+'\t'+str("{0:.2f}".format(float(dif[i])))+'\t'+str("{0:.2f}".format(float(rel_dif[i])))+'\t'+str("{0:.4f}".format(float(amp[i])))+'\t'+str("{0:.4f}".format(float(f_amp[i])))+"\n")
    f.close()
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"r"+"transfer-amp.png", dpi=400, bbox_inches = 'tight',transparent=True) # nazev souboru 
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"r"+"transfer-amp.svg", dpi=400, bbox_inches = 'tight',transparent=True) # nazev souboru 
 
def transfer_tre_amp(i1,i2,i_bg,i_ref,i_ref2,save,biasonly,**kwargs):
    cur2=[]
    scounts=[]
    bias=[]
    bias2=[]
    Z=[]
    freq=[]
    tre=[]
    amp0=[] #driving amplitude generator
    counts_cor=[]
    dif=[]
    amp_cond=[]
    trans_ok=[]
    trans_nk=[]
    tre_refar=[]
    singal_nf_ar=[]
    difcounts_ar=[]
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
    if isfloat(i_ref)==True:
        i_ref=i_ref
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_ref):
                i_ref=j
                break
    if isfloat(i_ref2)==True:
        i_ref2=i_ref2
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_ref2):
                i_ref2=j
                break
    if isfloat(i_bg)==True:
        i_bg=i_bg
    else:
        for j in range (0,len(file_numbers)):
            if filedata_dat[j].header["Filename"]==str(i_bg):
                i_bg=j
    lr=filedata_dat[i1].signals['Wavelength (nm)'][0]
    rr=filedata_dat[i1].signals['Wavelength (nm)'][-1]       
    ncol=len(filedata_dat[i1].signals['Wavelength (nm)'])-1
    ppnm=ncol/(rr-lr)
    bg0=savgol_filter(filedata_dat[i_bg].signals['Counts'],11,0)
    for i in range(i1, i2+1):
         if float(filedata_dat[i].header["Bias (V)"])==biasonly:
            bias.append(filedata_dat[i].header["Bias (V)"])
            Z.append(float(filedata_dat[i].header["Z (m)"]))
            #if float(filedata_dat[i].header["Bias (V)"])>1.9:
            if float(filedata_dat[i].header["Bias (V)"])>1.3:
                bias2.append(True)
            else:
                bias2.append(False)
            #freq.append(filedata_dat[i].header["GEN FREQ (MHz)"])
            #print(filedata_dat[i].header["genarb frequency (MHz)"],"freq")
            try:
                freq.append(float(filedata_dat[i].header["genarb frequency (MHz)"]))
            except ValueError:
                freq.append(0)
                print(i," ",filedata_dat[i].header["Filename"])
            if "gen_arb" in kwargs:
                if kwargs["gen_arb"]==True:
                    amp0.append(0.5*float(filedata_dat[i].header["genarb power (mV)"]))   
                else:
                    amp0.append(np.sqrt(2)*float(filedata_dat[i].header["genarb power (mV)"]))
            else:
                amp0.append(np.sqrt(2)*float(filedata_dat[i].header["genarb power (mV)"]))
            n=float(filedata_dat[i].header["Number of Accumulations"])
            if np.min(filedata_dat[i].signals['Counts'])<250:
                bg=np.full((len(filedata_dat[i_bg].signals['Counts'])),np.mean(filedata_dat[i].signals['Counts'][0:10]) )
                print(bg)
            else:
                bg=bg0
            signal_nf=np.subtract(filedata_dat[i].signals['Counts'], bg)
            signal=savgol_filter(np.subtract(filedata_dat[i].signals['Counts'], bg),11,0)
            counts_cor.append(np.subtract(filedata_dat[i].signals['Counts'], bg))
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        
            for ij in range(0,len(bg)):
                #print(ppnm,"ppnm")
                if signal[ij]>2 and signal_nf[ij]>2 and signal[ij+int(ppnm*1)]>3 and signal_nf[ij+int(ppnm*1)]>3 and signal[min(ij+int(ppnm*5),1023)]>7 and signal_nf[min(ij+int(ppnm*5),1023)]>10 and signal_nf[min(ij+50,1023)]>10: #new lower threshold 
                    tre.append(filedata_dat[i1].signals['Wavelength (nm)'][ij])
                    break
                if ij==len(bg-1):
                    tre.append(np.NaN)
            #print(tre,"tre")
    for i in range(i_ref,i_ref2+1):
        #print(filedata_dat[i].header["Bias (V)"])
        if float(filedata_dat[i].header["Bias (V)"])==biasonly and "OFF" in filedata_dat[i].header["genarb output"]:
            bias.append(filedata_dat[i].header["Bias (V)"])
            n=float(filedata_dat[i].header["Number of Accumulations"])
            bg=savgol_filter(filedata_dat[i_bg].signals['Counts'],21,0)
            signal_nf=np.subtract(filedata_dat[i].signals['Counts'], bg)
            signal=savgol_filter(np.subtract(filedata_dat[i].signals['Counts'], bg),11,0)
            counts_cor.append(np.subtract(filedata_dat[i].signals['Counts'], bg))
            cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
            
            for ij in range(0,len(bg)):
                if signal[ij]>2 and signal_nf[ij]>2 and signal[ij+int(ppnm*1)]>3 and signal_nf[ij+int(ppnm*1)]>3 and signal[min(ij+int(ppnm*3),1023)]>7 and signal_nf[min(ij+int(ppnm*5),1023)]>20 and signal_nf[min(ij+50,1023)]>10:
                    tre_refar.append(filedata_dat[i1].signals['Wavelength (nm)'][ij])
                    print(tre_refar[-1])
                    print(eV/tre_refar[-1])
                    break
                if ij==len(bg-1):
                    tre_refar.append(np.NaN)
            #print(tre_refar,"trerefar")
    
            fig, ax1 = plt.subplots() #osy 
            ax1.set_xlabel('wl [nm]')
            ax1.set_ylabel('signal [a.u.]')
            plt.xlim(xmin=0)
            ax1.plot(filedata_dat[i1].signals['Wavelength (nm)'],signal, label="frekvence_ref"+bias[-1],linestyle="--",color="red")
            plt.axvline(x=tre_refar[-1],label=str(tre_refar[-1]))
            plt.legend()
            ax1.grid(True,linestyle=':')
            ax1.set_xlim((tre_refar[-1]-30, tre_refar[-1]+60)) 
            ax1.set_ylim((-5, 25)) 
            fig.set_size_inches(4.4,4.4)
            singal_nf_ar.append(signal_nf)
    tre_ref=np.nanmean(np.array(tre_refar))
    signal_refmean=np.mean(np.array(singal_nf_ar),axis=0)
    for i in range(i1, i2+1):
        if float(filedata_dat[i].header["Bias (V)"])==biasonly:
            bg=savgol_filter(filedata_dat[i_bg].signals['Counts'],11,0)
            signal_nf=np.subtract(filedata_dat[i].signals['Counts'], bg)
            signal_dif=np.subtract(signal_nf, signal_refmean)
            difcounts=sum(signal_dif[nmtopix(tre_ref-20,lr,rr,ncol):nmtopix(tre_ref+10,lr,rr,ncol)])
            difcounts_ar.append(difcounts)
    for i in range(i2,i2+1):
        bias.append(filedata_dat[i].header["Bias (V)"])
        n=float(filedata_dat[i].header["Number of Accumulations"])
        bg=savgol_filter(filedata_dat[i_bg].signals['Counts'],21,0)
        signal_nf=np.subtract(filedata_dat[i].signals['Counts'], bg)
        signal=savgol_filter(np.subtract(filedata_dat[i].signals['Counts'], bg),11,0)
        counts_cor.append(np.subtract(filedata_dat[i].signals['Counts'], bg))
        cur=abs(float(filedata_dat[i].header["Current avg. (A)"]))*1E12
        
        for ij in range(0,len(bg)):
            if signal[ij]>3 and signal_nf[ij+1]>3 and signal_nf[ij+5]>5 and signal_nf[ij+10]>10:
                tre_ref2=(filedata_dat[i1].signals['Wavelength (nm)'][ij])
                print(tre_ref2)
                print(eV/tre_ref2)
                break
            if ij==len(bg-1):
                tre_ref2=(np.NaN)
    for i in range(0,len(tre)):
        if (tre_ref-tre[i])>14:
            dif.append(1000*abs((eV/tre_ref)-(eV/tre[i])))
            print(tre_ref-tre[i],"difference nm")
            amp_cond.append(dif[i]/amp0[i])
        else:
            dif.append(1000*abs((eV/tre_ref)-(eV/tre[i])))
            amp_cond.append(np.NaN)
            #print(tre_ref-tre[i],"difference nm")
        if amp0[i]>=3500*np.sqrt(2):
            trans_nk.append(dif[i]/amp0[i])
            trans_ok.append(np.NaN)
        else:
            trans_ok.append(dif[i]/amp0[i])
            trans_nk.append(np.NaN)           
    print(len(dif))
    print(len(amp0))
    rel_dif=np.max(dif)*np.array(difcounts_ar)/np.max(difcounts_ar)
    print(rel_dif)
    amp=np.divide(np.array(dif),np.array(amp0))   
    f_amp=scipy.signal.medfilt(amp,5) 
    fig, ax1 = plt.subplots() #osy 
    ax1.set_xlabel('Driving amplitude (mV)')
    ax1.set_ylabel('Amplitude (mV)')
    ax2 = ax1.twinx()
    ax2.set_ylim((0.01,0.5))
    ax2.set_yscale('log')
    ax2.set_ylabel('Transmisson (fraction)')  
 #   ax1.plot(freq,amp, label=str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"][-5:])+" "+"B "+str("{0:.2f}".format(float(bias[0])))+"V",marker=".",linestyle="None")
  #  ax1.plot(freq,amp_cond, label=">15 nm shift",marker=".",linestyle="None")
    
    ax1.plot(amp0,dif, label=str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"][-5:])+" "+"B "+str("{0:.2f}".format(float(bias[0])))+"V",marker=".",linestyle="None",markersize=1.5,color="blue")
    ax2.plot(amp0,amp, label=str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"][-5:])+" "+"B "+str("{0:.2f}".format(float(bias[0])))+"V",marker="v",linestyle="None",markersize=1.5,color="black")
    #ax1.plot(freq,trans_nk, label="notcompensated",marker=".",linestyle="None",markersize=1.5,color="red")
    #ax2.plot(freq,np.array(dif), label="compensated",linestyle="-",color="blue",alpha=0.8)
    if "plot_rel" in kwargs:
        if kwargs["plot_rel"]==True:
            ax2.plot(freq,rel_dif, label="compensated",linestyle="-",color="black")
    else:
        pass
    if "plot_Z" in kwargs:
        if kwargs["plot_Z"]==True:
            ax2.plot(freq,np.array(Z)*1E9, label="Z",linestyle="-",color="black")
    else:
        pass   
   # ax1.set_xticks([freq[0],freq[-1]]) 
    #print(freq)
   # plt.legend()
    #popt, pcov=curve_fit(npower2,cur2,scounts)
    #print(popt[1])
    #ax1.plot(cur2,npower(cur2,popt[0],popt[1],popt[2],popt[3]), label="fit")
    #ax1.plot(cur2,npower2(cur2,popt[0],popt[1]), label="fit")
   # ax1.grid(True,linestyle=':')
    if "x1" in kwargs:
       plt.xlim(xmin=float(kwargs["x1"]))
    else:
       pass
    if "x2" in kwargs:
       plt.xlim(xmax=float(kwargs["x2"]))
    else:
       pass
   
    if "y1" in kwargs:
        ax1.set_ylim(bottom=float(kwargs["y1"]))
    else:
        pass
    if "y2" in kwargs:
        ax1.set_ylim(top=float(kwargs["y2"]))
    else:
        pass 
    if "y21" in kwargs:
        ax2.set_ylim(bottom=float(kwargs["y21"]))
    else:
        pass
    if "y22" in kwargs:
        ax2.set_ylim(top=float(kwargs["y22"]))
    else:
        pass 
    fig.set_size_inches(4.4,2.2)
    
    f= open(path+str(filedata_dat[i1].header["Filename"])+"cal_amplitude.txt","w+")
    f.write("Frequency [MHz]"+'\t'+"Gen power peak [mV]"+'\t'+"Amplitude tre [mV]"+'\t'+"Transfer function [fraction]"+"\n")
    for i in range (0,len(amp)):
        f.write(str("{0:.1f}".format(float(freq[i])))+'\t'+str("{0:.2f}".format(float(amp0[i])))+'\t'+str("{0:.2f}".format(float(dif[i])))+'\t'+str("{0:.4f}".format(float(amp[i])))+"\n")
    f.close()
    description="bias"+str("{0:.2f}".format(float(biasonly)))+" "+filedata_dat[i_bg].header["Filename"]+" "+filedata_dat[i_ref].header["Filename"]+" "+filedata_dat[i_ref2].header["Filename"]
    metadata={"description":description}
    if save==True:
            plt.savefig(path+str(filedata_dat[i1].header["Filename"])+"-"+str(filedata_dat[i2].header["Filename"])+"r"+"transfer-amp.png", dpi=400, bbox_inches = 'tight',metadata=metadata) # nazev souboru 
            
def findgrating(number):
        for j in range (0,len(file_numbers)):
            try:
                if float(filedata_dat[j].header["GGR"])==number and float(filedata_dat[j].header["GWL"])<700:
                    print((filedata_dat[j].header["Filename"]))
                    fn=(filedata_dat[j].header["Filename"])
                    bgcorplot_div(fn,0,0,0,1240/(1.9+0.02),1240/(1.9-0.02),"eV",True,path,"div",False,maj_int=0.05,min_int=0.01,maj_int2=10,min_int2=2,bg=305)
            except KeyError:
                pass
#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2019/2019-04-05/a-map1/"
#file_list, filedata_dat, file_numbers,file_names = getsorteddat(path)
#map_hm("f-CHmap_1.asc","f-CHmap_484.asc",-2,640,660,22,22,"nm",True,False,0.5,0.8,1)
#map_hm("a-multimap_1.asc","a-multimap_400.asc",-2,640,660,20,20,"nm",True,False,0.5,1.7)
#path = "C:/Users/Jirka/ownCloud/Documents/fzu/data/2019/2019-04-05/a-map2/"
#file_list, filedata_dat, file_numbers,file_names = getsorteddat(path)
#map_hm("a-multimap_401.asc","a-multimap_800.asc",-2,640,660,20,20,"nm",True,False,0.5,1)

#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2019/2019-04-05/c-CHmap1/"
#file_list, filedata_dat, file_numbers,file_names = getsorteddat(path)
#map_hm("c-CHmap_1.asc","c-CHmap_400.asc",-2,640,660,20,20,"nm",True,False,0.5,0.7)
#path = "C:/Users/Jirka/ownCloud/Documents/fzu/data/2019/2019-04-05/c-CHmap2/"
#file_list, filedata_dat, file_numbers,file_names = getsorteddat(path)
#map_hm("c-CHmap_401.asc","c-CHmap_1424.asc",-1,640,660,32,32,"nm",True,False,0.5,0.7)
#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2019/2019-04-03/e-manual-reorder/"
#file_list, filedata_dat, file_numbers,file_names = getsorteddat(path)
#profile_wf("e-manual_1.asc","e-manual_7.asc",'c-bg_1.asc',0,2,600,700,10,False,"nm",True,True) #fig 3a Au Au tip 
#bgcorplot("e-manual_11.asc",'c-bg_1.asc',600,700,"nm",2)
#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2019/2019-04-05/a-map1/"
#file_list, filedata_dat, file_numbers,file_names = getsorteddat(path)
#profile_wf_fmman("a-multimap_1.asc","a-multimap_400.asc",-2,600,700,20,20,[[9,9],[7,11],[9,11],[6,12],[9,13],[4,14],[9,15]],2,"nm",True,True) #fig 3b Au CO tip

#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2019/2019-04-16 [1]/profileCH/"
#file_list, filedata_dat, file_numbers,file_names = getsorteddat(path)
#bgcorplot("d-CHprofile_13.asc",'bgd-CHprofile_16.asc',600,700,"nm",1.5)
#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2019/2019-07-03/"                                                               
#file_list, filedata_dat, file_numbers,file_names = getsorteddat(path)
#path="C:/Users/Jirka/ownCloud/Documents/fzu/data/2019/2019-04-16 [1]/"
#file_list, filedata_dat, file_numbers,file_names = getsorteddat(path)
#profile_wf_fmman("f-CHmap_1.asc","f-CHmap_484.asc",-2,600,700,22,22,[[12,11],[15,13],[18,16],[12,14],[12,18]],2,"nm",True,True) #fig 4b Ag CO tip

#map_hm("LS-CHmap-c00401","LS-CHmap-c00800",-3,645,660,20,20,"nm",True,False,0,1,1)
#map_hm("LS-CHmap-c00401","LS-CHmap-c00800",-3,808,823,20,20,"nm",True,False,0,1,1)
#map_hm("LS-CHmap-c00001","LS-CHmap-c00400",-3,808,823,20,20,"nm",True,False,0,1,1)
#map_hm("LS-CHmap-c00001","LS-CHmap-c00400",-3,645,660,20,20,"nm",True,False,0,1,1)
#map_hm("LS-CHmap-d00001","LS-CHmap-d00400",-3,645,660,20,20,"nm",True,False,0,1,1)
#map_hm("LS-CHmap-d00001","LS-CHmap-d00400",-3,808,823,20,20,"nm",True,False,0,1,1)
#map_hm("LS-CHmap-b00001","LS-CHmap-b00400",-3,808,823,20,20,"nm",True,False,0,1,1)
#map_hm("LS-CHmap-b00001","LS-CHmap-b00400",-3,645,660,20,20,"nm",True,False,0,1,1)
#map_hm("LS-CHmap-b00401","LS-CHmap-b00800",-3,645,660,20,20,"nm",True,False,0,1,1)

#map_hm("LS-CHmap-b00401","LS-CHmap-b00800",-3,808,823,20,20,"nm",True,False,0,1,1)


#map_hm("LS-CHmap-a00401","LS-CHmap-a00800",-3,640,655,20,20,"nm",True,False,0,1,1)
#map_hm("LS-CHmap-a00401","LS-CHmap-a00800",-3,808,823,20,20,"nm",True,False,0,1,1)
#map_hm("LS-CHmap-a00001","LS-CHmap-a00400",-3,640,655,20,20,"nm",True,False,0,1,1)
#map_hm("LS-CHmap-a00001","LS-CHmap-a00400",-3,808,823,20,20,"nm",True,False,0,1,1)
#map_hm("LS-CHmap-e00001","LS-CHmap-e00576",-3,640,655,24,24,"nm",True,False,0,1,1)
#map_hm("LS-CHmap-e00001","LS-CHmap-e00576",-3,808,823,24,24,"nm",True,False,0,1,1)

#for i in range(0,16,2):
  #  map_hm("LS-CHmap-a00401","LS-CHmap-a00800",-3,640+i,642+i,20,20,"nm",True,False,0,2,1)
#for i in range(0,16,2):
   # map_hm("LS-CHmap-a00401","LS-CHmap-a00800",-3,808+i,810+i,20,20,"nm",True,False,0,2,1)
#transfer_tre_new("LS-fr-calibration00001","LS-fr-calibration00030","LS-bg00001","LS-fr-reference00002",400,900,"nm",True,1.5)
#transfer_tre_new("LS-fr-calibration00001","LS-fr-calibration00030","LS-bg00001","LS-fr-reference00002",400,900,"nm",True,1.5)
#transfer_tre_new("LS-fr-calibration00030","LS-fr-calibration00060","LS-bg00001","LS-fr-reference00003",400,900,"nm",True,1.5)
#transfer_tre_new("LS-fr-calibration00030","LS-fr-calibration00060","LS-bg00001","LS-fr-reference00003",400,900,"nm",True,1.5)
#transfer_tre_new("LS-fr-calibration00030","LS-fr-calibration00060","LS-bg00001","LS-fr-reference00003",400,900,"nm",True,1.5)
#transfer_tre_new("LS-fr-calibration00030","LS-fr-calibration00060","LS-bg00001","LS-fr-reference00003",400,900,"nm",True,1.5)
#transfer_tre_new("LS-fr-calibration00030","LS-fr-calibration00060","LS-bg00001","LS-fr-reference00003",400,900,"nm",True,1.5)
#transfer_tre_new("LS-fr-calibration00030","LS-fr-calibration00060","LS-bg00001","LS-fr-reference00003",400,900,"nm",True,1.5)
#transfer_tre_new("LS-fr-calibration00030","LS-fr-calibration00060","LS-bg00001","LS-fr-reference00003",400,900,"nm",True,1.5)
#transfer_tre_new("LS-fr-calibration00030","LS-fr-calibration00060","LS-bg00001","LS-fr-reference00003",400,900,"nm",True,1.5)
#transfer_tre_new("LS-fr-calibration00030","LS-fr-calibration00060","LS-bg00001","LS-fr-reference00003",400,900,"nm",True,1.5)
#transfer_tre_new("LS-fr-calibration00061","LS-fr-calibration00090","LS-bg00001","LS-fr-reference00004",400,900,"nm",True,1.5)
#transfer_tre_new("LS-fr-calibration00091","LS-fr-calibration00097","LS-bg00001","LS-fr-reference00001",400,900,"nm",True,1.5)
#transfer_tre_new("LS-fr-calibration00098","LS-fr-calibration00104","LS-bg00001","LS-fr-reference00002",400,900,"nm",True,1.5)
#transfer_tre_new("LS-fr-calibration00105","LS-fr-calibration00111","LS-bg00001","LS-fr-reference00003",400,900,"nm",True,1.5)
#transfer_tre_new("LS-fr-calibration00119","LS-fr-calibration00138","LS-bg00001","LS-fr-reference00002",400,900,"nm",True,1.5)
"""map_hmsxm("LS-CHmap-a00001","LS-CHmap-a00400",-3,645,660,808,825,"nm",False,0,1,1)
map_hmsxm("LS-CHmap-a00401","LS-CHmap-a00800",-3,645,660,808,825,"nm",False,0,1,1)
map_hmsxm("LS-CHmap-b00001","LS-CHmap-b00400",-3,645,660,808,825,"nm",False,0,1,1)
map_hmsxm("LS-CHmap-b00401","LS-CHmap-b00800",-3,645,660,808,825,"nm",False,0,1,1)
map_hmsxm("LS-CHmap-c00001","LS-CHmap-c00400",-3,645,660,808,825,"nm",False,0,1,1)
map_hmsxm("LS-CHmap-c00401","LS-CHmap-c00800",-3,645,660,808,825,"nm",False,0,1,1)
map_hmsxm("LS-CHmap-d00001","LS-CHmap-d00400",-3,645,660,808,825,"nm",False,0,1,1)
map_hmsxm("LS-CHmap-d00401","LS-CHmap-d00800",-3,645,660,808,825,"nm",False,0,1,1)
map_hmsxm("LS-CHmap-e00001","LS-CHmap-e00576",-3,645,660,808,825,"nm",False,0,1,1)
"""
#map_hmsxm("LS-CHmap-c00001","LS-CHmap-c00400",-3,635,650,805,820,"nm",False,0,1,1)
'''map_hmsxm("LS-CHmap-e00001","LS-CHmap-e00760",-3,640,649,651,654,"nm",False,0,1,1)
map_hmsxm("LS-CHmap-e00001","LS-CHmap-e00760",-3,807,817,817,825,"nm",False,0,1,1)
map_hmsxm("LS-CHmap-f00001","LS-CHmap-f00760",-3,640,649,651,654,"nm",False,0,1,1)
map_hmsxm("LS-CHmap-f00001","LS-CHmap-f00760",-3,807,817,817,825,"nm",False,0,1,1)
#map_hmsxm("LS-CHmap-a00001","LS-CHmap-f00640",-3,807,817,817,825,"nm",False,0,1,1)
#map_hmsxm("LS-CHmap-a00001","LS-CHmap-a00640",-3,643,654,808,823,"nm",False,0,1,1)'''

"""map_hmsxm("LS-CHmap-a00401","LS-CHmap-a00800",-3,650,652,818,820,"nm",False,0,1,1) #map fig2 subset
map_hmsxm("LS-CHmap-a00401","LS-CHmap-a00800",-3,644,646,812,814,"nm",False,0,1,1)
map_hmsxm("LS-CHmap-a00401","LS-CHmap-a00800",-3,646,648,814,816,"nm",False,0,1,1)
map_hmsxm("LS-CHmap-a00401","LS-CHmap-a00800",-3,648,650,816,818,"nm",False,0,1,1)"""

#for j in range (0,len(file_numbers)):
 #   if "LS-CHmap-d" in filedata_dat[j].header["Filename"]:
#        bgcordiv_txt(j,"LS-CHmap-d00001",0,0,807,827,"eV",True,path,"div",False)  

#plot all spectra
"""
letters = ['a', 'b', 'c', 'd']
numbers = range(1, 70)  # or whatever range you want, inclusive/exclusive

# --- main loop ---
for letter in letters:
    # estimate maximum index (you can adjust)
    for i in range(20):  # enough iterations to cover your data range
        n1 = 1 + i * 8
        n2 = 8 + i * 8

        s1 = f"LS-PL-{letter}-{n1:05d}"
        s2 = f"LS-PL-{letter}-{n2:05d}"

        found1 = any(s1 in f for f in file_names)
        found2 = any(s2 in f for f in file_names)

        if found1 and found2:
            bgcorplot_stitch_2(
                s1, s2,
                "LS-PL-a-00057", "LS-PL-a-00064",
                -50, 6050, "cm", True, path, "div", False,
                maj_int=0.1, min_int=0.02, maj_int2=500, min_int2=10,
                offset=0, despike_w=5, sec_axis=True, laser_nm=532, offset_nm=-0.4
            )
"""