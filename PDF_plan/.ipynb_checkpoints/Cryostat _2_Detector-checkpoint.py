# Cryostat T-dependent XRD and PDF measurements
# Date Created: 02/09/2024(MA) 
# Last modified: 03/30/2025(MA)
import time
import numpy as np
import shutil
import matplotlib.pyplot as plt 
N = np.nan


#------------------------------------------------- Definition of variables -----------------------------------------------
''' Change below variables as necessay  🤔🤔🤔 '''
A = list(range(10, 500, 5))
B = [300]#np.arange(25, 305, 10)  #Uper limit excluded
#Tlist_2 = Tlist_1[::-1]

Tlist =  [4.5] + A + B

st = 300                 # sleep time for Temperature stabiliy
st2 = 2                 # sleep time before each measuement. this time will also be used to stabilize detector after changing frame acquisition time

S_X =  np.array([-674.38, -678.52, -680.55, -682.53, -684.55])# - 0.15 # Scan_shifter sample position list 
SMPI_PDF = [1,11,12,13,14]       # bt.list() Sample indices for PDF
SMPI_XRD = [2,23,24,25,26]       # bt.list() Sample indices for XRD
SPI_PDF = [0, 6, 6, 6, 6]        # bt.list() Scan plan indices for PDF
SPI_XRD = [0, 6, 6, 6, 6]        # bt.list() Scan plan indices for PDF
FMT = [0.1, 0.1, 0.1, 0.1, 0.1]  # Frame acquisition times for the samples. Set this value based on PDF requirement
OPTION = [0, 0, 0, 0, 0]         # Measurement option for each sample 0: PDF & XRD, 1: PDF only, 2:XRD Only 

my_config = {'auto_mask': False, 'qmaxinst':24.9, 'qmax':24.9, 'rpoly':0.7,
    'user_mask': '/nsls2/data3/pdf/pdfhack/legacy/processed/xpdacq_data/user_data/config_base/Mask.npy',    
    'method': 'splitpixel'}

#------------------------------------------= ''' Do not change anything  below !!!'''-------------------------------------
config_dir = "/nsls2/data/pdf/pdfhack/legacy/processed/xpdacq_data/user_data/config_base/" 
Time, T1, T2, Temperature, DetZ, SampleX, DI1, DI2, TCF, M_current = [],[],[],[],[],[],[],[],[],[] #TCF before measurement
#Tseries = lst(bt.samples.keys())[smpi] 

def Figure():
    plt.figure()
    plt.xlabel("Time (h)")
    plt.ylabel("Temperature (K)") 

def set_PDF(): # Move PDF poni and mask files to config directory
    xpd_configuration['area_det']=pilatus1
    try:
        os.remove(config_dir + "xpdAcq_calib_info.poni")
    except Exception:
        pass      
    shutil.copy(config_dir + "/PDF/" + "xpdAcq_calib_info.poni" , config_dir)
    try:
        os.remove(config_dir + "Mask.npy")
    except Exception:
        pass
    shutil.copy(config_dir + "/PDF/" + "Mask.npy" , config_dir)

def set_XRD(): # Move XRD poni and mask files to config directory
    xpd_configuration['area_det']=pe1c
    try:
        os.remove(config_dir + "xpdAcq_calib_info.poni")
    except Exception:
        pass     
    shutil.copy(config_dir + "/XRD/" + "xpdAcq_calib_info.poni" , config_dir)
    try:
        os.remove(config_dir + "Mask.npy")
    except Exception:
        pass
    shutil.copy(config_dir + "/XRD/" + "Mask.npy" , config_dir)


def change_T(t): # Cryostat chanel A setpoint loop
    caput('XF:28ID1-ES{LS336:1-Out:1}T-SP',t)
    while True:
        tt =  lakeshore336.read()['lakeshore336_temp_A_T']['value']
        if t-1 <= tt <= t+1: # changed from +/- 0.3
            break
        else:
            print(tt)
        time.sleep(5)
    
def measurement_data(): # .......Captures metadata   
    info_dict = {}
    info_dict['OT_stage_1_X'] = OT_stage_1_X.read()
    info_dict['OT_stage_1_Y'] = OT_stage_1_Y.read()
    info_dict['Det_1_X'] = Det_1_X.read()
    info_dict['Det_1_Y'] = Det_1_Y.read()
    info_dict['Det_1_Z'] = Det_1_Z.read()
    info_dict['Grid_X'] = Grid_X.read()
    info_dict['Grid_Y'] = Grid_Y.read()
    info_dict['Grid_Z'] = Grid_Z.read()
    info_dict['ring_current'] = ring_current.read()
    info_dict['frame_acq_time'] = glbl['frame_acq_time']
    info_dict['dk_window'] = glbl['dk_window']
    info_dict['cryostat_A'] = lakeshore336.read()['lakeshore336_temp_A_T']['value']
    info_dict['cryostat_A_V'] = caget('XF:28ID1-ES{LS336:1-Chan:A}Val:Sens-I')
    info_dict['cryostat_B'] = lakeshore336.read()['lakeshore336_temp_B_T']['value']
    info_dict['cryostat_B_V'] = caget('XF:28ID1-ES{LS336:1-Chan:B}Val:Sens-I')
    info_dict['cryostat_C'] = lakeshore336.read()['lakeshore336_temp_C_T']['value']
    info_dict['cryostat_C_V'] = caget('XF:28ID1-ES{LS336:1-Chan:C}Val:Sens-I')
    info_dict['cryostat_D'] = lakeshore336.read()['lakeshore336_temp_D_T']['value']
    info_dict['cryostat_D_V'] = caget('XF:28ID1-ES{LS336:1-Chan:D}Val:Sens-I')
    info_dict['Measurement_time'] = time.time()
    return info_dict

def Det_scan(SI, SP, repeat): # Pilatus three position scan
    for j in range(repeat):
        det_x = [40.644, 31.356, 36]
        det_y = [-3.356, -12.644, -8]
        for i in range(len(det_x)):
            Grid_X.move(det_x[i])
            Grid_Y.move(det_y[i])
            #xrun(SI, jog([pilatus1], SP, OT_stage_2_Y, use_ypos-.5, use_ypos+.5), dark_strategy=no_dark,  more_info = useful_info())
            xrun(SI, SP, dark_strategy= no_dark, more_info = measurement_data(), user_config = my_config)


def measurement_PDF(): # PDF measurement 
    time_ = time.time()
    Time.append(time_)
    temperature = lakeshore336.read()['lakeshore336_temp_A_T']['value']
    temperature1 = lakeshore336.read()['lakeshore336_temp_B_T']['value']
    temperature2 = lakeshore336.read()['lakeshore336_temp_C_T']['value']
    temperature3 = lakeshore336.read()['lakeshore336_temp_D_T']['value']
    Det_scan(smpi,spi,1)  # Disable this in enable the previous line when 3 positions are not necessary
    plt.plot((time_-Time[0])/3600,temperature, 'ro', markersize=.8)
    plt.plot((time_-Time[0])/3600,temperature1, 'go', markersize=.8)
    plt.plot((time_-Time[0])/3600,temperature2, 'bo', markersize=.8)
    plt.plot((time_-Time[0])/3600,temperature3, 'ko', markersize=.8)
    plt.ion()   
    plt.pause(0.05)

def measurement_XRD(): # XRD measurement
    time_ = time.time()
    Time.append(time_)
    temperature = lakeshore336.read()['lakeshore336_temp_C_T']['value']
    xrun(smpi, spi, more_info = measurement_data(), user_config = my_config) 


        
# ------------------------------------------- ''' Main T dependent Measurement loop '''' -----------------------------------------
for i in range(len(Tlist)):
    change_T(Tlist[i])
    time.sleep(st)
    if i == 0:
        Figure()
        Time = []
    
    set_PDF()    
    for jj in range (len(S_X)):
        glbl['frame_acq_time']= FMT[jj]
        time.sleep(st2)
        OT_stage_1_X.move(S_X[jj])
        spi_PDF = SPI_PDF[jj]
        spi_XRD = SPI_XRD[jj]
        smpi_PDF = SMPI_PDF[jj]
        smpi_XRD = SMPI_XRD[jj]
        option_1 = OPTION[jj]

        if option_1 == 0:   # PDF+XRD       
            spi = spi_PDF
            smpi = smpi_PDF
            measurement_PDF()
        elif option_1 == 1:
            spi = spi_PDF
            smpi = smpi_PDF
            measurement_PDF()
        else:
            pass

    set_XRD()    
    for kk in range (len(S_X)):
        glbl['frame_acq_time']= FMT[kk]
        time.sleep(st2)
        OT_stage_1_X.move(S_X[kk])
        spi_PDF = SPI_PDF[kk]
        spi_XRD = SPI_XRD[kk]
        smpi_PDF = SMPI_PDF[kk]
        smpi_XRD = SMPI_XRD[kk]
        option_1 = OPTION[kk]

        if option_1 == 0:   # PDF+XRD       
            spi = spi_XRD
            smpi = smpi_XRD
            measurement_XRD()
        elif option_1 == 2:
            spi = spi_XRD
            smpi = smpi_XRD
            measurement_XRD()
        else:
            pass
            
'''
# -------Below part of the script insert metada in the headers of IQ, Itth, and Gr files in a way that pdfgui and GSAS-II can recognize and import-------
# meta_data function should be pre-defined by running "metadata_insert.py"
time.sleep(30)
for kk in range(len(SMPI_PDF)):
    try:
        meta_data(SMPI_PDF[kk])
        print(SMPI_PDF[kk])
    except Exception:
        pass
    
for ll in range(len(SMPI_XRD)):
    try:
        meta_data(SMPI_XRD[ll])
        print(SMPI_XRD[ll])
    except Exception:
        pass
        
'''