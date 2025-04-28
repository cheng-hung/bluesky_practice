# Cryostream T-dependent XRD and PDF measurements of multiple samples
# Created: 02/04/2022 (MA)
# Tested and working 02/20/2024
import time
import pylab
from pylab import *
import numpy as np
import os
from tifffile import imread, imsave
import matplotlib.pyplot as plt 
import shutil 

#======================================== Variables =======================================
# Users plese change variables as necessary 
SMPI_PDF = [26, 27, 28, 29, 30, 31]#[::-1]                  # Sample indecies for PDF in bt.list()
SMPI_XRD = [32, 33, 34, 35, 36, 37]#[::-1]                  # Sample indecies for XRD in bt.list()
SPI_PDF = [5, 5, 5, 5, 5, 5]                                # for each sample Scan Plan indecies in bt.list() for PDF measurements
SPI_XRD = [4, 4, 4, 4, 4, 4]                                # Scan Plan indecies in bt.list() for XRD measurements
Sample_X = [ -48.13,  -60.04,  -72.16,  -84.16,  -96.3 , -108.2]  #[::-1] # Pre-defined samples positions 
Sample_Y = [45.0, 45.0, 45.0, 45.0, 45.0, 45.0]
FMT = [3, 3, 3, 3, 3, 3]                # Frame acq_times of the samples
Measurement_Option = [0, 0, 0, 0, 0, 0]       # 0 Both PDF and XRD, 1- PDF only, 2 - XRD Only
PDF_Number = [1, 1, 1, 1, 1, 1]              # Number of times PDF measurement is repeated at every temperature
XRD_Number = [1, 1, 1, 1, 1, 1]              # Number of times XRD measurement is repeated at every temperature
DARK_W = [1, 1, 1, 1, 1, 1]    #Unit second

Tlist_1 = [300] + list(range(400,90,-5)) + [300]      # Measurement Temperatures for SP[1]   #large temperatrue step : need give more time 
Tlist_2 = [300] + list(range(400,90,-5)) + [300]      # Measurement Temperatures for SP[2]
Tlist_3 = [300] + list(range(400,90,-5)) + [300]      # Measurement Temperatures for SP[3]
Tlist_4 = [300] + list(range(400,90,-5)) + [300]       # Measurement Temperatures for SP[4]
Tlist_5 = [300] + list(range(400,90,-5)) + [300]       # Measurement Temperatures for SP[5]
Tlist_6 = [300] + list(range(400,90,-5)) + [300]      # Measurement Temperatures for SP[1]


TLIST = [Tlist_1, Tlist_2, Tlist_3, Tlist_4, Tlist_5, Tlist_6]

st = 60      # sleep time for T stabiliy   #large temperatrue step : need give more time 50K step may give more 120 -180 seco at leat
st2 = 30     # sleep time before XRD measuement
st3 = 0     # sleep after changing detector FMT

D1 = 2956   # PDF detector position
D2 = 3956   # XRD detector position

# Do not change anyhing below ---------------
#======================================= Definitions of functions =======================================
Time = []
config_dir = "/nsls2/data3/pdf/pdfhack/legacy/processed/xpdacq_data/user_data/config_base/"

def figure():
    plt.figure()
    plt.xlabel("Time from start of the measurement (h)")
    plt.ylabel("Temperature (K)")

def measurement_data(): # .......Captures metadata   
    info_dict = {}
    info_dict['OT_stage_2_X'] = OT_stage_2_X.read()
    info_dict['OT_stage_2_Y'] = OT_stage_2_Y.read()
    info_dict['Det_1_X'] = Det_1_X.read()
    info_dict['Det_1_Y'] = Det_1_Y.read()
    info_dict['Det_1_Z'] = Det_1_Z.read()
    info_dict['ring_current'] = ring_current.read()
    info_dict['frame_acq_time'] = glbl['frame_acq_time']
    info_dict['dk_window'] = glbl['dk_window']
    info_dict['Temperature'] = cryostream.T.get()
    info_dict['Measurement_time'] = time.time()
    return info_dict

def move_PDF(): # Copy PDF mask & calibration files to config_base
    Det_1_Z.move(D1)
    try:
        os.remove(config_dir + "xpdAcq_calib_info.poni")
    except Exception:
        pass      
    shutil.copy(config_dir + "/PDF/" + "xpdAcq_calib_info.poni" , config_dir)
    try:
        os.remove(config_dir + "Mask.npy")
    except Excemption:
        pass
    shutil.copy(config_dir + "/PDF/" + "Mask.npy" , config_dir)

def move_XRD(): # Copy XRD mask & calibration files to config_base
    Det_1_Z.move(D2)
    try:
        os.remove(config_dir + "xpdAcq_calib_info.poni")
    except Exception:
        pass     
    shutil.copy(config_dir + "/XRD/" + "xpdAcq_calib_info.poni" , config_dir)
    try:
        os.remove(config_dir + "Mask.npy")
    except Excemption:
        pass
    shutil.copy(config_dir + "/XRD/" + "Mask.npy" , config_dir)

def measurement(): # measurement
    t1 = cryostream.T.get()
    time1 = round(time.time())
    Time.append(time1)
    xrun(Sample_Index, spi, more_info = measurement_data()) # Modify SI and Scan_Plan
    plt.plot((time1-Time[0])/3600,t1,'bo')
    plt.ion()  
    plt.pause(0.05)

#=================================== Measurement Logic  ====================================
     
for jj in range(len(Sample_X)):
    OT_stage_2_X.move(Sample_X[jj])
    OT_stage_2_Y.move(Sample_Y[jj])
    glbl['frame_acq_time'] = FMT[jj]
    time.sleep(st3)
    glbl['dk_window'] = DARK_W[jj]         # dark current acquasition window
    NNNN = PDF_Number[jj]
    NNNX = XRD_Number[jj]
    spi_PDF = SPI_PDF[jj]
    spi_XRD = SPI_XRD[jj]
    smpi_PDF = SMPI_PDF[jj]
    smpi_XRD = SMPI_XRD[jj]
    Tlist = TLIST[jj]
    Option_1 = Measurement_Option[jj]
    if jj == 0:
        figure()
    
    if Option_1 == 0:   # PDF and XRD
        for i in range(len(Tlist)):
            RE(mv(cryostream, Tlist[i]))
            time.sleep(st)
            move_PDF()
            Sample_Index = smpi_PDF
            spi = spi_PDF
            for k in range(NNNN):
                measurement()
            move_XRD()
            Sample_Index = smpi_XRD
            spi = spi_XRD
            time.sleep(st2)
            for l in range(NNNX):
                measurement()

    if Option_1 == 1:   #PDF 
        move_PDF()
        Sample_Index = smpi_PDF
        spi = spi_PDF
        for i in range(len(Tlist)):
            RE(mv(cryostream, Tlist[i]))
            time.sleep(st)
            for l in range(NNNN):
                measurement()

    if Option_1 == 2:   #XRD
        move_XRD()
        Sample_Index = smpi_XRD
        spi = spi_XRD
        for i in range(len(Tlist)):
            RE(mv(cryostream, Tlist[i]))
            time.sleep(st)
            for l in range(NNNX):
                measurement()
    meta_data(smpi_PDF)
    meta_data(smpi_XRD)


'''
# -------Below part of the script insert metadata in the headers of I-Q, I-tth, and Gr files in a way that pdfgui and GSAS-II can recognize and import-------
# meta_data function should be pre-defined by running "metadata_insert.py"

    for kk in range(len(SMPI_PDF)):
        try:
            meta_data(SMPI_PDF[kk])
            print(SMPI_PDF[kk])
        except Exception:
            pass
        
    for ll in range(len(SMPI_XRD)):
        try:
            meta_data(SMPI_XRD[ll])
            print(SMPI_XRD[jj])
        except Exception:
            pass  

'''
