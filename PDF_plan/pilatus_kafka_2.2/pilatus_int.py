import pyFAI
import os
import numpy as np
import numpy.ma as ma
import pandas as pd

import importlib
Pilatus_sum = importlib.import_module("pilatus_sum_v2.2").Pilatus_sum


def iq_saver(fn, df, md, header=['q_A^-1', 'I(q)']):
    
    with open(fn, mode='w', encoding='utf-8') as f:
        f.write('pyFai_poni_information_28ID1_NSLS2_BNL\n')
        num_row = 1
        for key, value in md.items():
            f.write(f'{key} {value}\n')
            num_row += 1
    
    ## Now append the dataframe
    df.to_csv(fn, encoding='utf-8', mode='a', header=header, index=False, float_format='{:.8e}'.format, sep=' ')

    ## return the number of rows of the header
    return num_row


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
    def pe1c_PDF(self):
        n = self.get('PATH', 'pe1c_PDF', fallback='pe1c_PDF')
        return os.path.join(self.config_base, n)
    
    @property
    def pe1c_XRD(self):
        n = self.get('PATH', 'pe1c_XRD', fallback='pe1c_XRD')
        return os.path.join(self.config_base, n)


    @property
    def poni_pe1c(self):
        n = self.get('PATH', 'poni_pe1c', fallback='xpdAcq_calib_info.poni')

        if self.is_pdf_xrd() == 'PDF':
            n_folder = self.pe1c_PDF
        else:
            n_folder = self.pe1c_XRD

        return os.path.join(n_folder, n)
    

    @property
    def mask_pe1c(self):
        n = self.get('PATH', 'mask_pe1c', fallback='Mask.npy')

        if self.is_pdf_xrd() == 'PDF':
            n_folder = self.pe1c_PDF
        else:
            n_folder = self.pe1c_XRD

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

    
    def pct_integration(self, full_imsum, process_dir):

        if self.run.start['detectors'][0] == 'pilatus1':
            ai = pyFAI.load(self.merged_poin)
            mask0 = np.load(self.stitched_mask)

        elif self.run.start['detectors'][0] == 'pe1c':
            ai = pyFAI.load(self.poni_pe1c)
            mask0 = np.load(self.mask_pe1c)
        
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
        iq_fn = os.path.join(process_dir, f'{self.file_name_prefix}_sum.iq')
        md = ai.getPyFAI()
        _md = {'detector': self.run.start['detectors'][0], 
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

        return iq_df, iq_fn