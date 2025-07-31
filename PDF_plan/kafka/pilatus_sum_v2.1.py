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
from configparser import ConfigParser
import pandas as pd

from tiled.client import from_profile
tiled_client = from_profile('pdf')



def _readable_time(unix_time):
    from datetime import datetime
    dt = datetime.fromtimestamp(unix_time)
    # print(f'{dt.year}{dt.month:02d}{dt.day:02d},{dt.hour:02d}{dt.minute:02d}{dt.second:02d}')
    return (f'{dt.year}{dt.month:02d}{dt.day:02d}'), (f'{dt.hour:02d}{dt.minute:02d}{dt.second:02d}')


def file_prefix_producer(uid, tiled_client):
    run = tiled_client[uid]
    full_uid = run.start['uid']
    sample_name = run.start['sample_name']
    readable_time = _readable_time(run.start['time'])

    file_prefix = f'{sample_name}_{readable_time[0]}-{readable_time[1]}_{full_uid:6.6}'

    return file_prefix



def iq_saver(fn, df, md, header=['q_A^-1', 'I(q)']):
    with open(fn, mode='w', encoding='utf-8') as f:
        f.write('pyFai_poni_information_28ID1_NSLS2_BNL\n')
        for key, value in md.items():
            f.write(f'{key} {value}\n')
    # Now append the dataframe without a header
    df.to_csv(fn, encoding='utf-8', mode='a', header=header, index=False, float_format='{:.8e}'.format, sep=' ')



class Pilatus_config(ConfigParser):
    """The configuration for the server."""

    def __init__(self, config_fn, **kwargs):
        # self.uid = uid
        self.config_fn = config_fn
        super().__init__(**kwargs)


    def read(self, **kwargs):
        return super().read(self.config_fn, **kwargs)
        



class Pilatus_sum(Pilatus_config):

    def __init__(self, uid, tiled_client, config_fn, **kwargs):
        self.uid = uid
        super().__init__(config_fn, **kwargs)
        self.read(**kwargs)

        self.tiled_client = tiled_client
        self.run = tiled_client[uid]
        
        self.raw_db = self.get('topics', 'raw_db', fallback='pdf')
        self.an_db = self.get('topics', 'an_db', fallback='pdf-analysis')

        self.full_uid = self.run.start['uid']

        self.sample_name = None  ## update at start doc
        self.stream_name = None  ## update at stop doc 

    @property
    def user_data(self):
        fallback = '/nsls2/data/pdf/pdfhack/legacy/processed/xpdacq_data/user_data'
        return self.get('PATH', 'user_data', fallback=fallback)
    
    @property
    def tiff_base(self):
        fallback = 'tiff_base'
        n = self.get('PATH', 'tiff_base', fallback=fallback)
        return os.path.join(self.user_data, n)
    
    @property
    def config_base(self):
        fallback = 'config_base'
        n = self.get('PATH', 'config_base', fallback=fallback)
        return os.path.join(self.user_data, n)     

    @property
    def pilatus_PDF(self):
        fallback = 'pilatus_PDF'
        n = self.get('PATH', 'pilatus_PDF', fallback=fallback)
        return os.path.join(self.config_base, n)
    
    @property
    def pilatus_XRD(self):
        fallback = 'pilatus_XRD'
        n = self.get('PATH', 'pilatus_XRD', fallback=fallback)
        return os.path.join(self.config_base, n)  

    @property
    def masks_pos_flist(self):
        m1 = self.get('PATH', 'mask_01', fallback='Mask_pos1_ext_BS.npy')
        m2 = self.get('PATH', 'mask_02', fallback='Mask_pos2_ext_BS.npy')
        m3 = self.get('PATH', 'mask_03', fallback='Mask_pos3_ext_BS.npy')
        return [m1, m2, m3]
    
    @property
    def osetx(self):
        return self.getint('SUM', 'osetx', fallback=27)
    
    @property
    def osety(self):
        return self.getint('SUM', 'osety', fallback=27)
    
    @property
    def use_flat_field(self):
        return self.getboolean('SUM', 'use_flat_field', fallback=False)
    
    @property
    def flat_field(self):
        n_folder = self.get('PATH', 'flat_filed', fallback='flat_filed')
        n = self.get('PATH', 'flat_field_fn', fallback='flat_filed.tiff')
        return os.path.join(self.config_base, n_folder, n)
    
    @property
    def temperature(self):
        try:
            temp_controller = self.get('TEMPERATURE', 'temp_controller', fallback='cryostat_A')
            T = self.run.start['more_info'][temp_controller]

        except (KeyError, IndexError, TypeError):
            T = 'None'

        return T


    @property
    def readable_time(self):
        t = _readable_time(self.run.start['time'])
        return f'{t[0]}-{t[1]}'
    
    @property
    def file_name_prefix(self):
        T = self.temperature

        if type(T) is float:
            return f'{self.sample_name}_{self.readable_time}_{self.full_uid:6.6}_{T}_K'

        else:
            return f'{self.sample_name}_{self.readable_time}_{self.full_uid:6.6}'
        
    

    def is_pdf_xrd(self, distance_limit=0.3, ):
        ## unit of distance is meter
        # run = tiled_client[uid]

        try:
            distance = self.run.start['calibration_md']['Distance']
            acq_mode = ''

            if distance < distance_limit:
                acq_mode = 'PDF'

            elif distance > distance_limit:
                acq_mode = 'XRD'

            else:
                acq_mode = 'PDF'

        except KeyError:
            print('\nCannot find distance in metadata in start doc')
            print('\nUse PDF as default mode.')
            acq_mode = 'PDF'

        return acq_mode


    def sum_everything(self):
        """ Assuming im2 offset by -osetx, -osety, and im3 offset by +osetx, +osety """
        
        # run = tiled_client[uid]
        my_im3 = getattr(self.run, self.stream_name[0]).read()['pilatus1_image'].to_numpy()[0][0]
        my_im2 = getattr(self.run, self.stream_name[1]).read()['pilatus1_image'].to_numpy()[0][0]
        my_im1 = getattr(self.run, self.stream_name[2]).read()['pilatus1_image'].to_numpy()[0][0]

        if self.use_flat_field:
            flat_field = imread(self.flat_field)
            my_im3 = my_im3 / flat_field
            my_im2 = my_im2 / flat_field
            my_im1 = my_im1 / flat_field

        if self.is_pdf_xrd() == 'PDF':
            mask_dir = self.pilatus_PDF

        else:
            mask_dir = self.pilatus_XRD

        # masks_pos_fn = ['Mask_pos1_ext_BS.npy', 'Mask_pos2_ext_BS.npy', 'Mask_pos3_ext_BS.npy']
        m1_path = os.path.join(mask_dir, self.masks_pos_flist[0])
        m2_path = os.path.join(mask_dir, self.masks_pos_flist[1])
        m3_path = os.path.join(mask_dir, self.masks_pos_flist[2])
        use_mask_1= np.load(m1_path)  # This is mask we are applying befor mergin images
        use_mask_2 = np.load(m2_path) # This is mask we are applying befor mergin images
        use_mask_3 = np.load(m3_path) # This is mask we are applying befor mergin images

        my_imsum = np.ones((my_im1.shape[0]+int(2*self.osetx), my_im2.shape[1]+int(2*self.osety),3))*np.nan

        my_imsum[self.osetx:-self.osetx,self.osety:-self.osety,0] = my_im1
        my_imsum[self.osetx:-self.osetx,self.osety:-self.osety,0][use_mask_1==1] = np.nan

        my_imsum[:-int(2*self.osetx),:-int(2*self.osety):,1] = my_im2
        my_imsum[:-int(2*self.osetx),:-int(2*self.osety):,1][use_mask_2==1] = np.nan

        my_imsum[int(2*self.osetx):,int(2*self.osety):,2] = my_im3
        my_imsum[int(2*self.osetx):,int(2*self.osety):,2][use_mask_3==1] = np.nan
        
        return np.nanmean(my_imsum, axis=2, dtype=np.float32)



    def save_image_sum_T(self):

        data_dir = os.path.join(self.tiff_base, self.sample_name, "dark_sub")

        sum_dir = os.path.join(data_dir, 'sum')
        os.makedirs(sum_dir, exist_ok=True)  # Create 'sum' directory if it doesn't exis
        
        full_imsum = self.sum_everything()
        
        # Save output file to the new 'sum' directory
        sum_tiff_fn = os.path.join(sum_dir, f'{self.file_name_prefix}_sum.tiff')
        tifffile.imsave(sum_tiff_fn, full_imsum)
        print(f'\n*** {os.path.basename(sum_tiff_fn)} saved!! ***\n')

        return full_imsum, sum_dir





class Pilatus_Int(Pilatus_sum):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def merged_poin(self):
        n = self.get('PATH', 'merged_poin', fallback='merged.poni')

        if self.is_pdf_xrd() == 'PDF':
            n_folder = self.pilatus_PDF
        else:
            n_folder = self.pilatus_XRD

        return os.path.join(n_folder, n)
    

    @property
    def stitched_mask(self):
        n = self.get('PATH', 'stitched_mask', fallback='stitched_mask.npy')

        if self.is_pdf_xrd() == 'PDF':
            n_folder = self.pilatus_PDF
        else:
            n_folder = self.pilatus_XRD

        return os.path.join(n_folder, n)


    @property
    def binning(self):
        return self.getint('INTEGRATION', 'binning', fallback=4096)
    
    @property
    def polarization(self):
        return self.getfloat('INTEGRATION', 'polarization', fallback=0.99)
    
    @property
    def UNIT(self):
        return self.get('INTEGRATION', 'UNIT', fallback="q_A^-1")
    
    @property
    def ll(self):
        return self.getfloat('INTEGRATION', 'low_limit_pcfilter', fallback=1.0)
    
    @property
    def ul(self):
        return self.getfloat('INTEGRATION', 'up_limit_pcfilter', fallback=99.0) 

    
    def pct_integration(self, full_imsum, sum_dir):

        ai = pyFAI.load(self.merged_poin)
        mask0 = np.load(self.stitched_mask)
        
        ## perform azimuthalintegration on one image to retain 2D information
        ## i2d.shape is (npt_azim, self.binning) which corresponds the intensity of 2D image cake
        ## q2d.shape is (self.binning, )
        i2d, q1d, chi1d = ai.integrate2d(full_imsum, self.binning, 
                                         unit=self.UNIT, npt_azim=3600, 
                                         polarization_factor=self.polarization, 
                                         mask=mask0) 
        
        ## trasnform mask0 (base mask) to the same coordinate space and cast it as type bool
        intrinsic_mask_unrolled, _, _ = ai.integrate2d(mask0, self.binning, 
                                                       unit=self.UNIT, npt_azim=3600, 
                                                       polarization_factor=self.polarization, 
                                                       mask=mask0)
        #intrinsic_mask_unrolled = intrinsic_mask_unrolled.astype(bool) 
        
        ## Create an array to hold outlier mask
        outlier_mask_2d = np.zeros_like(i2d)     
        mask1 = np.array(i2d<1)*1
        
        ## Apply percentile filter along radial direction (axis=0)
        for ii, dd in enumerate(i2d.T):
            low_limit, high_limit = np.percentile(dd, (self.ll, self.ul))
            outlier_mask_2d[:,ii] = np.any([dd<low_limit, dd>high_limit, intrinsic_mask_unrolled[:,ii]], axis=0)
          
        outlier_mask_2d_masked = ma.masked_array(i2d, mask=outlier_mask_2d + mask1)
        
        ## calculate mean values along radial direction (axis=0) to make i1d.shape is (self.binning, )
        i1d = ma.mean(outlier_mask_2d_masked, axis=0)
        
        df = pd.DataFrame()
        df['q'] = q1d
        df['I'] = i1d
        iq_fn = os.path.join(sum_dir, f'{self.file_name_prefix}_sum.iq')
        md = ai.getPyFAI()
        _md = {'detector': self.run.start['sp_detector'], 
               'uid':self.full_uid, 
               'time': self.run.start['time'], 
               'readable_time': self.readable_time, 
               'percentile_low_limit': self.ll, 
               'percentile_up_limit': self.ul, 
               }
        md.update(_md)

        iq_saver(iq_fn, df, md)
        print(f'\n*** {os.path.basename(iq_fn)} saved!! ***\n')

        return df, iq_fn



'''

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



def get_gr(uid, iq_data, cfg_fn, bkg_fn, output_dir, gr_fn_prefix):
    run = tiled_client[uid]
    
    ## run.start['sample_composition'] is a dict but pdfconfig takes a string for composition 
    # scan_com = run.start['sample_composition']

    pdfconfig = PDFConfig()
    pdfconfig.readConfig(cfg_fn)

    # pdfconfig.composition = composition_maker(scan_com)

    ## There is also a string for composition in run.start. Updated by CHLin on 2025/06/16
    try:
        pdfconfig.composition = run.start['composition_string']
        print(f'\n\nFound composition as {run.start['composition_string'] = }')

    except (KeyError):
        pdfconfig.composition = 'Ni1.0'
        print(f'\n\nCan not find sample composition in run.start. Use "Ni1.0" instead.')

    pdfconfig.backgroundfiles = bkg_fn
    sqfqgr_path, pdfconfig = transform_bkg(
                pdfconfig, iq_data, 
                output_dir = output_dir, 
                plot_setting={'marker':'.','color':'green'}, test=True, 
                gr_fn=gr_fn_prefix)
    
    print(f'\n*** {os.path.basename(sqfqgr_path["gr"])} saved!! ***\n')

    return sqfqgr_path, pdfconfig
    pdfconfig.readConfig(cfg_fn)

    # pdfconfig.composition = composition_maker(scan_com)

    ## There is also a string for composition in run.start. Updated by CHLin on 2025/06/16
    try:
        pdfconfig.composition = run.start['composition_string']
        print(f'\n\nFound composition as {run.start['composition_string'] = }')

    except (KeyError):
        pdfconfig.composition = 'Ni1.0'
        print(f'\n\nCan not find sample composition in run.start. Use "Ni1.0" instead.')

    pdfconfig.backgroundfiles = bkg_fn
    sqfqgr_path, pdfconfig = transform_bkg(
                pdfconfig, iq_data, 
                output_dir = output_dir, 
                plot_setting={'marker':'.','color':'green'}, test=True, 
                gr_fn=gr_fn_prefix)
    
    print(f'\n*** {os.path.basename(sqfqgr_path["gr"])} saved!! ***\n')

    return sqfqgr_path, pdfconfig

'''


