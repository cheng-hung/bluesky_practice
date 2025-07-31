import pyFAI
import os
from tifffile import imread
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import tifffile
from configparser import ConfigParser
import pandas as pd


from pdfstream.transformation.io import write_pdfgetter
from pdfstream.transformation.main import get_pdf
from diffpy.pdfgetx import PDFConfig
# from diffpy.pdfgetx.pdfgetter import PDFConfigError


from tiled.client import from_profile
tiled_client = from_profile('pdf')


import importlib
auto_bkg = importlib.import_module("auto_bkg").auto_bkg


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
        num_row = 1
        for key, value in md.items():
            f.write(f'{key} {value}\n')
            num_row += 1
    
    # Now append the dataframe without a header
    df.to_csv(fn, encoding='utf-8', mode='a', header=header, index=False, float_format='{:.8e}'.format, sep=' ')

    return num_row


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
    def T_controller(self):
        return self.get('TEMPERATURE', 'temp_controller', fallback='No_temp_controller')
    
    @property
    def temperature(self):
        try:
            # temp_controller = self.get('TEMPERATURE', 'temp_controller', fallback='No_temp_controller')
            T = self.run.start['more_info'][self.T_controller]

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

        self.num_rows_header = 1

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
    def npt_rad(self):
        # equivalent to binning
        return self.getint('INTEGRATION', 'npt_rad', fallback=4096)
    
    @property
    def npt_azim(self):
        return self.getint('INTEGRATION', 'npt_azim', fallback=3600)
    
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
        ## i2d.shape is (self.npt_azim, self.npt_rad) which corresponds the intensity of 2D image cake
        ## q1d.shape is (self.npt_rad, )
        i2d, q1d, chi1d = ai.integrate2d(full_imsum, self.npt_rad, 
                                         unit=self.UNIT, npt_azim=self.npt_azim, 
                                         polarization_factor=self.polarization, 
                                         mask=mask0) 
        
        ## trasnform mask0 (base mask) to the same coordinate space and cast it as type bool
        intrinsic_mask_unrolled, _, _ = ai.integrate2d(mask0, self.npt_rad, 
                                                       unit=self.UNIT, npt_azim=self.npt_azim, 
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
        
        ## calculate mean values along radial direction (axis=0) to make i1d.shape is (self.npt_rad, )
        i1d = ma.mean(outlier_mask_2d_masked, axis=0)
        
        iq_df = pd.DataFrame()
        iq_df['q'] = q1d
        iq_df['I'] = i1d
        iq_fn = os.path.join(sum_dir, f'{self.file_name_prefix}_sum.iq')
        md = ai.getPyFAI()
        _md = {'detector': self.run.start['sp_detector'], 
               'uid':self.full_uid, 
               'time': self.run.start['time'], 
               'readable_time': self.readable_time, 
               'percentile_low_limit': self.ll, 
               'percentile_up_limit': self.ul, 
               self.T_controller: self.temperature, 
               }
        md.update(_md)

        ## num_row will be the number of rows of the header in saved iq data file
        self.num_rows_header = iq_saver(iq_fn, iq_df, md)
        print(f'\n*** {os.path.basename(iq_fn)} saved!! ***\n')

        return iq_df






class Pilatus_getpdf(Pilatus_Int):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.auto_bkg = 1.0

    @property
    def use_auto_bkg(self):
        return self.getboolean('pdfgetx3', 'use_auto_bkg', fallback=False)

    @property
    def pdfconfig_dict(self):
        return {
            'dataformat':     self.get('pdfgetx3', 'dataformat', fallback='QA'), 
            'outputtype':     self.get('pdfgetx3', 'outputtype', fallback='gr'), 
            'backgroundfile': self.get('pdfgetx3', 'backgroundfile', fallback=''), 
            'plot':           self.get('pdfgetx3', 'plot', fallback='none'), 
            'bgscale':        self.getfloat('pdfgetx3', 'bgscale', fallback=0.98), 
            'rpoly':          self.getfloat('pdfgetx3', 'rpoly', fallback=1.0), 
            'qmaxinst':       self.getfloat('pdfgetx3', 'qmaxinst', fallback=27.0), 
            'qmin':           self.getfloat('pdfgetx3', 'qmin', fallback=0.6), 
            'qmax':           self.getfloat('pdfgetx3', 'qmax', fallback=25.0), 
            'rmin':           self.getfloat('pdfgetx3', 'rmin', fallback=0.0), 
            'rmax':           self.getfloat('pdfgetx3', 'rmax', fallback=100.0), 
            'rstep':          self.getfloat('pdfgetx3', 'rstep', fallback=0.1), 
            }



    def pdfconfig(self):
        
        p = PDFConfig()
        for key, value in self.pdfconfig_dict:
            setattr(p, key, value)
        
        return p



    ## Modified from https://github.com/NSLS2/xpd-profile-collection-ldrd20-31/blob/main/scripts/_get_pdf.py
    def transform_bkg(self, iq_df, sum_dir, test=False):
        
        # try:
        #     pdfgetter = get_pdf(self.pdfconfig(), iq_df, plot_setting=self.plot)
        
        # except PDFConfigError:
        #     pdfgetter = get_pdf(self.pdfconfig(), iq_df, plot_setting='OFF')
        
        pdfgetter = get_pdf(self.pdfconfig(), iq_df, plot_setting='OFF')

        sqfqgr_path = write_pdfgetter(sum_dir, f'{self.file_name_prefix}_sum', pdfgetter)
        
        # if not test:
        #     plt.show()
        
        return sqfqgr_path




    def get_gr(self, iq_df, sum_dir, num_row):

        try:
            self.pdfconfig().composition = self.run.start['composition_string']
            print(f'\n\nFound composition as {self.run.start['composition_string'] = }')

        except (KeyError):
            self.pdfconfig().composition = 'Ni1.0'
            print(f'\n\nCan not find sample composition in run.start. Use "Ni1.0" instead.')

        
        ## Use auto_bkg to repalce the bkg in pdfconfig
        if self.use_auto_bkg:
            a_bkg = auto_bkg()
            a_bkg.data_df = iq_df
            a_bkg.pdload_bkg(self.pdfconfig_dict['backgroundfile'], 
                             skiprows=self.num_rows_header, 
                             sep=' ', names=['Q', 'I'])
            res = a_bkg.min_integral(bkg_tor=0.01)
            self.pdfconfig().bgscale[0] = res.x
            self.auto_bkg = res.x
            print(f'\nUpdate {self.pdfconfig().bgscales[0] = } by auto_bkg\n')
            # a_bkg.plot_sub()


        sqfqgr_path = self.transform_bkg(iq_df, sum_dir)
        
        print(f'\n*** {os.path.basename(sqfqgr_path["gr"])} saved!! ***\n')

        return sqfqgr_path




'''

    @property
    def backgroundfile(self):
        return self.get('pdfgetx3', 'backgroundfile', fallback='')
    
    @property
    def bgscale(self):
        return self.getfloat('pdfgetx3', 'bgscale', fallback=0.98)
    
    @property
    def rpoly(self):
        return self.getfloat('pdfgetx3', 'rpoly', fallback=1.0)

    @property
    def qmaxinst(self):
        return self.getfloat('pdfgetx3', 'qmaxinst', fallback=27.0)
    
    @property
    def qmin(self):
        return self.getfloat('pdfgetx3', 'qmin', fallback=0.6)
    
    @property
    def qmax(self):
        return self.getfloat('pdfgetx3', 'qmax', fallback=25.0)

    @property
    def rmin(self):
        return self.getfloat('pdfgetx3', 'rmin', fallback=0.0)
    
    @property
    def rmax(self):
        return self.getfloat('pdfgetx3', 'rmax', fallback=100.0)
    
    @property
    def rstep(self):
        return self.getfloat('pdfgetx3', 'rstep', fallback=0.1)
    
    @property
    def dataformat(self):
        return self.get('pdfgetx3', 'dataformat', fallback='QA')
    
    @property
    def outputtype(self):
        return self.get('pdfgetx3', 'outputtype', fallback='gr')     
    
    
    
'''


