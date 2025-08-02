import os
from pdfstream.transformation.io import write_pdfgetter
from pdfstream.transformation.main import get_pdf
from diffpy.pdfgetx import PDFConfig
# from diffpy.pdfgetx.pdfgetter import PDFConfigError

import importlib
Pilatus_Int = importlib.import_module("pilatus_int").Pilatus_Int
auto_bkg = importlib.import_module("auto_bkg").auto_bkg


class Pilatus_getpdf(Pilatus_Int):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bgscale = self.getfloat('pdfgetx3', 'bgscale', fallback=0.98)
        self.auto_bkg = 1.0

    @property
    def use_auto_bkg(self):
        return self.getboolean('pdfgetx3', 'use_auto_bkg', fallback=False)

    @property
    def do_reduction(self):
        return self.getboolean('pdfgetx3', 'do_reduction', fallback=True)

    @property
    def bgscale(self):
        return self._bgscale
    
    @bgscale.setter
    def bgscale(self, scale):
        self._bgscale = scale

    @property
    def pdfconfig_dict(self):
        return {
            'dataformat':       self.get('pdfgetx3', 'dataformat', fallback='QA'), 
            'outputtype':       self.get('pdfgetx3', 'outputtype', fallback='gr'), 
            'backgroundfile':   self.get('pdfgetx3', 'backgroundfile', fallback='test'), 
            'plot':             self.get('pdfgetx3', 'plot', fallback='none'), 
            # 'bgscale':          self.getfloat('pdfgetx3', 'bgscale', fallback=0.98), 
            'bgscale':          self.bgscale, 
            'rpoly':            self.getfloat('pdfgetx3', 'rpoly', fallback=1.0), 
            'qmaxinst':         self.getfloat('pdfgetx3', 'qmaxinst', fallback=27.0), 
            'qmin':             self.getfloat('pdfgetx3', 'qmin', fallback=0.6), 
            'qmax':             self.getfloat('pdfgetx3', 'qmax', fallback=25.0), 
            'rmin':             self.getfloat('pdfgetx3', 'rmin', fallback=0.0), 
            'rmax':             self.getfloat('pdfgetx3', 'rmax', fallback=100.0), 
            'rstep':            self.getfloat('pdfgetx3', 'rstep', fallback=0.1), 
            'composition':      self.run.start['composition_string'], 
            }

    
    def pdfconfig(self):
        
        p = PDFConfig()
        for key, value in self.pdfconfig_dict.items():
            setattr(p, key, value)
        
        return p



    ## Modified from https://github.com/NSLS2/xpd-profile-collection-ldrd20-31/blob/main/scripts/_get_pdf.py
    def transform_bkg(self, iq_df, process_dir, test=False):
        
        print(self.pdfconfig_dict)
        # print(self.pdfconfig())
        
        pdfgetter = get_pdf(self.pdfconfig(), iq_df, plot_setting='OFF')

        if self.run.start['detectors'][0] == 'pilatus1':
            sqfqgr_path = write_pdfgetter(process_dir, f'{self.file_name_prefix}_sum', pdfgetter)

        elif self.run.start['detectors'][0] == 'pe1c':
            sqfqgr_path = write_pdfgetter(process_dir, f'{self.file_name_prefix}_flat', pdfgetter)

        else:
            sqfqgr_path = write_pdfgetter(process_dir, f'{self.file_name_prefix}_process', pdfgetter)

        # if not test:
        #     plt.show()
        
        return sqfqgr_path




    def get_gr(self, iq_df, process_dir):

        try:
            # self.pdfconfig().composition = self.run.start['composition_string']
            print(f'\nFound composition as {self.run.start["composition_string"] = }\n')

        except (KeyError):
            # self.pdfconfig().composition = 'Ni1.0'
            print(f'\nCan not find sample composition in run.start. Use "Ni1.0" instead.\n')

        
        ## Use auto_bkg to repalce the bkg in pdfconfig
        if self.use_auto_bkg:
            a_bkg = auto_bkg()
            a_bkg.data_df = iq_df
            
            a_bkg.pdload_bkg(self.pdfconfig_dict['backgroundfile'], 
                             skiprows=self.num_rows_header, 
                             sep=' ', names=['Q', 'I'])
            res = a_bkg.min_integral()
            print(f'{res.x = }')
            self.bgscale = float(res.x)
            print(f'{self.pdfconfig().bgscale[0] = }')
            self.auto_bkg = res.x
            print(f'\nUpdate {self.pdfconfig().bgscales[0] = } by auto_bkg\n')
            # a_bkg.plot_sub()

        iq_array = iq_df.to_numpy().T
        sqfqgr_path = self.transform_bkg(iq_array, process_dir)
        
        print(f'\n*** {os.path.basename(sqfqgr_path["gr"])} saved!! ***\n')

        return sqfqgr_path