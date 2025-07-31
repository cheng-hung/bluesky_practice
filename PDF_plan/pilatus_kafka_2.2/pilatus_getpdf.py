import os
from pdfstream.transformation.io import write_pdfgetter
from pdfstream.transformation.main import get_pdf
from diffpy.pdfgetx import PDFConfig
# from diffpy.pdfgetx.pdfgetter import PDFConfigError

import importlib
Pilatus_Int = importlib.import_module("pilatus_int_v2.2").Pilatus_Int
auto_bkg = importlib.import_module("auto_bkg").auto_bkg


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