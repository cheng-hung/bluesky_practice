import pyFAI
import pyFAI.calibrant
import pyFAI.detectors
import os, glob, re
# from ipywidgets import interact, interactive, fixed, interact_manual, HBox, VBox
# import ipywidgets as widgets
from tifffile import imread, imshow, imsave
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
# import matplotlib.colors as colors
# from matplotlib import gridspec
# from matplotlib.widgets import Slider, Button
import yaml, tifffile
# %matplotlib widget

from tiled.client import from_profile
tiled_client = from_profile('pdf')

import importlib
auto_bkg = importlib.import_module("auto_bkg").auto_bkg


def sum_everything(uid, stream_name, masks_path, masks_pos_flist, osetx = 27, osety = 27):
    """ Assuming im2 offset by -osetx, -osety, and im3 offset by +osetx, +osety """
    
    run = tiled_client[uid]
    my_im3 = getattr(run, stream_name[0]).read()['pilatus1_image'].to_numpy()[0][0]
    my_im2 = getattr(run, stream_name[1]).read()['pilatus1_image'].to_numpy()[0][0]
    my_im1 = getattr(run, stream_name[2]).read()['pilatus1_image'].to_numpy()[0][0]


    # masks_pos_fn = ['Mask_pos1_ext_BS.npy', 'Mask_pos2_ext_BS.npy', 'Mask_pos3_ext_BS.npy']
    use_mask_1= np.load(os.path.join(masks_path, masks_pos_flist[0])) # This is mask we are applying befor mergin images
    use_mask_2 = np.load(os.path.join(masks_path, masks_pos_flist[1])) # This is mask we are applying befor mergin images
    use_mask_3 = np.load(os.path.join(masks_path, masks_pos_flist[2])) # This is mask we are applying befor mergin images

    my_imsum = np.ones((my_im1.shape[0]+int(2*osetx), my_im2.shape[1]+int(2*osety),3))*np.nan

    my_imsum[osetx:-osetx,osety:-osety,0] = my_im1
    my_imsum[osetx:-osetx,osety:-osety,0][use_mask_1==1] = np.nan

    my_imsum[:-int(2*osetx),:-int(2*osety):,1] = my_im2
    my_imsum[:-int(2*osetx),:-int(2*osety):,1][use_mask_2==1] = np.nan

    my_imsum[int(2*osetx):,int(2*osety):,2] = my_im3
    my_imsum[int(2*osetx):,int(2*osety):,2][use_mask_3==1] = np.nan
    return np.nanmean(my_imsum,axis=2)


def save_image_sum_T(uid, stream_name, sample_name, osetx = 27, osety = 27, 
                     masks_pos_flist = ['Mask_pos1_ext_BS.npy', 'Mask_pos2_ext_BS.npy', 'Mask_pos3_ext_BS.npy'], 
                     masks_path = '/nsls2/data3/pdf/pdfhack/legacy/processed/xpdacq_data/user_data/config_base/pilatus_mask',  
                     tiff_base_path = '/nsls2/data3/pdf/pdfhack/legacy/processed/xpdacq_data/user_data/tiff_base',
                     ):
    
    data_dir = os.path.join(tiff_base_path, sample_name, "dark_sub")
    meta_dir = os.path.join(tiff_base_path, sample_name, "meta")
    Meta = glob.glob(os.path.join(meta_dir, f'**{uid[:6]}.yaml'))

    File_Name_Prefix = os.path.basename(Meta[0]).split('.')[0]
    sum_dir = os.path.join(data_dir, 'sum')
    os.makedirs(sum_dir, exist_ok=True)  # Create 'sum' directory if it doesn't exist
        
    try:
        with open(os.path.join(meta_dir, Meta[0]), 'r') as f:
            data = yaml.unsafe_load(f)
            T = data.get('more_info')['cryostat_A']
    except (KeyError, IndexError, TypeError):
        T = 'None'
    
    full_imsum = sum_everything(uid, stream_name, masks_path, masks_pos_flist, osety=osety, osetx=osetx)
    
    # Save output file to the new 'sum' directory
    sum_tiff_fn = f"{sum_dir}/{File_Name_Prefix}_Temperature_{T}_K_sum.tiff"
    tifffile.imsave(sum_tiff_fn, full_imsum)
    print(f'\n*** {os.path.basename(sum_tiff_fn)} saved!! ***\n')

    saved_fn_prefix = os.path.basename(sum_tiff_fn).split('.')[0]

    return full_imsum, sum_dir, saved_fn_prefix


def pct_integration(img, fn_prefix, binning=5000, polarization=0.99, UNIT = "q_A^-1", ll=1, ul=99, 
                    poni_fn = '/nsls2/data3/pdf/pdfhack/legacy/processed/xpdacq_data/user_data/config_base/merged_PDF/merged.poni', 
                    mask_fn = '/nsls2/data3/pdf/pdfhack/legacy/processed/xpdacq_data/user_data/config_base/pilatus_mask/stitched_2.npy',
                    directory = '/nsls2/data3/pdf/pdfhack/legacy/processed/xpdacq_data/user_data/tiff_base',
                    ):
    
    ai = pyFAI.load(poni_fn)
    mask0 = np.load(mask_fn)
    
    i2d, q2d, chi2d = ai.integrate2d(img, binning, unit=UNIT, npt_azim=3600, polarization_factor=polarization, mask=mask0) # perform azimuthalintegration on one image to retain 2D information
    intrinsic_mask_unrolled,_,_ = ai.integrate2d(mask0, binning, unit=UNIT, npt_azim=3600, polarization_factor=polarization, mask=mask0)  #trasnform mask0 (base mask) to the same coordinate space and cast it as type bool
    #intrinsic_mask_unrolled = intrinsic_mask_unrolled.astype(bool) 
    outlier_mask_2d = np.zeros_like(i2d)     # Create an array to hold outlier mask

    i2d, q2d, chi2d = ai.integrate2d(img, binning, unit=UNIT, npt_azim=3600, polarization_factor=polarization, mask=mask0)
    mask1 = np.array(i2d<1)*1
    
    for ii, dd in enumerate(i2d.T):
        low_limit, high_limit = np.percentile(dd, (ll,ul))
        outlier_mask_2d[:,ii] = np.any([dd<low_limit, dd>high_limit, intrinsic_mask_unrolled[:,ii]], axis=0)
    mask=outlier_mask_2d + mask1    
    outlier_mask_2d_masked =  ma.masked_array(i2d, mask=outlier_mask_2d + mask1)     
    data = np.column_stack((q2d, ma.mean(outlier_mask_2d_masked,axis=0)))
    #plt.plot(q2d, ma.mean(outlier_mask_2d_masked,axis=0))
    # np.savetxt(directory + "/dark_sub/sum/" + str(a) + "_L" + str(ll)+ "_U" + str(ul) + "_percentile_masked_"+ str(UNIT) + ".dat", data) # Uncomment when the data is ready to be saved

    data_fn = os.path.join(directory, fn_prefix+"_L"+str(ll)+"_U"+str(ul)+"_percentile_masked_"+str(UNIT)+".dat")
    np.savetxt(data_fn, data) # Uncomment when the data is ready to be saved

    print(f'\n*** {os.path.basename(data_fn)} saved!! ***\n')

    return data, data_fn





from pdfstream.io import load_array
from pdfstream.transformation.io import load_pdfconfig, write_pdfgetter
from pdfstream.transformation.main import get_pdf
from diffpy.pdfgetx import PDFConfig
from diffpy.pdfgetx.pdfgetter import PDFConfigError
import typing


## Copy from https://github.com/NSLS2/xpd-profile-collection-ldrd20-31/blob/main/scripts/_get_pdf.py
def transform_bkg(
    cfg_file,
    data_file: str,
    output_dir: str = ".",
    plot_setting: typing.Union[str, dict] = None,
    test: bool = False,
    gr_fn: str = '/home/xf28id2/Documents/test.gr', 
    ) -> typing.Dict[str, str]:
    
    """Transform the data."""
    if isinstance(cfg_file,str):
        pdfconfig = load_pdfconfig(cfg_file)
    else:
        pdfconfig = cfg_file
    
    if type(data_file) is str:
        chi = load_array(data_file)
    elif type(data_file) is np.ndarray:
        chi = data_file
    
    try:
        pdfgetter = get_pdf(pdfconfig, chi, plot_setting=plot_setting)
    except PDFConfigError:
        pdfgetter = get_pdf(pdfconfig, chi, plot_setting='OFF')
    
    # filename = Path(data_file).stem
    # dct = write_pdfgetter(output_dir, filename, pdfgetter)
    
    dct = write_pdfgetter(output_dir, gr_fn, pdfgetter)
    if not test:
        plt.show()
    
    return dct, pdfconfig


## Add by CHLin on 2025/06/13 to turn sample_composition from dict to string
def composition_maker(scan_comp):
    com = ''
    for i in scan_comp.keys():
        com += f'{i} {scan_comp[i] }'

    return com



def get_gr(uid, iq_data, cfg_fn, bkg_fn, output_dir, gr_fn_prefix, is_autobkg=True):
    run = tiled_client[uid]
    
    ## run.start['sample_composition'] is a dict but pdfconfig takes a string for composition 
    # scan_com = run.start['sample_composition']

    pdfconfig = PDFConfig()
    pdfconfig.readConfig(cfg_fn)

    # pdfconfig.composition = composition_maker(scan_com)

    ## There is also a string for composition in run.start. Updated by CHLin on 2025/06/16
    try:
        pdfconfig.composition = run.start['composition_string']
        print(f'\n\nFound composition as {run.start["composition_string"] = }')

    except (KeyError):
        pdfconfig.composition = 'Ni1.0'
        print(f'\n\nCan not find sample composition in run.start. Use "Ni1.0" instead.')

    
    ## Use auto_bkg to repalce the bkg in pdfconfig
    if is_autobkg:
        a_bkg = auto_bkg(iq_data, bkg_fn)
        a_bkg.pdload_data(skiprows=1, sep=' ', names=['Q', 'I'])
        a_bkg.pdload_bkg(skiprows=1, sep=' ', names=['Q', 'I'])
        res = a_bkg.min_integral()
        pdfconfig.bgscales[0] = res.x
        print(f'\nUpdate {pdfconfig.bgscales[0] = } by auto_bkg\n')
        # a_bkg.plot_sub()
    
    pdfconfig.backgroundfiles = bkg_fn
    sqfqgr_path, pdfconfig = transform_bkg(
                pdfconfig, iq_data, 
                output_dir = output_dir, 
                plot_setting={'marker':'.','color':'green'}, test=True, 
                gr_fn=gr_fn_prefix)
    
    print(f'\n*** {os.path.basename(sqfqgr_path["gr"])} saved!! ***\n')

    return sqfqgr_path, pdfconfig




def is_pdf_xrd(uid, distance_limit=0.3):
    ## unit of distance is meter

    run = tiled_client[uid]
    distance = run.start['calibration_md']['Distance']
    acq_mode = ''

    if distance < distance_limit:
        acq_mode = 'PDF'

    elif distance > distance_limit:
        acq_mode = 'XRD'

    return acq_mode

