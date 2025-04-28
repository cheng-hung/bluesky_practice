import numpy as np
import matplotlib.plot as plt
import pandas as pd
import os
import glob

from bluesky.utils import ts_msg_hook
import bluesky.preprocessors as bpp
import bluesky.plan_stubs as bps


def acq_config(
                sample_x=0.0, 
                sample_y=45, 
                sample_name = 'test', 
                det_position=[2956, 3956],   ##[PDF, XRD]
                exposure_sec=[30, 30],       ##[PDF, XRD] 
                dark_win=[1, 1],             ##[PDF, XRD] 
                frame_acq_times=[0.2, 0.2],  ##[PDF, XRD]  
                config_dir='', 
                ):
    
    config = {
                'sample_x': sample_x, 
                'sample_y': sample_y, 
                'sample_name': sample_name, 
                'det_position': det_position, 
                'exposure_sec': exposure_sec, 
                'dark_win': dark_win, 
                'frame_acq_times': frame_acq_times, 
                'config_dir':config_dir, 
                }
    
    return config
    


def measurement_data(): # .......Captures metadata by MA
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


def set_glbl(frame_acq_time, dk_window):
    
    glbl['frame_acq_time']=frame_acq_time
    print(f"{glbl['frame_acq_time'] = }")
    
    glbl['dk_window']=dk_window
    print(f"{glbl['dk_window'] = }")
    
    # glbl['auto_load_calib'] = auto_load_calib
    # print(f"{glbl['auto_load_calib'] = }")
    
    # glbl['shutter_control'] = shutter_control
    # print(f"{glbl['shutter_control'] = }")


def pdf_xrd_one_det(det, det_motor, acq_config, *args, md=None, switch_rest=3, **kwargs):

    ## Measure PDF and XRD by moving det

    def switch_acquisition_type(det_position, config_type, config_dir):
        
        try:
            os.remove(config_dir + "xpdAcq_calib_info.poni")
        except Exception:
            pass      
        
        poni_to_be_copied = os.path.join(config_dir, config_type, "xpdAcq_calib_info.poni")
        # shutil.copy(config_dir + f"/{config_type}/" + "xpdAcq_calib_info.poni" , config_dir)
        shutil.copy(poni_to_be_copied , config_dir)
        
        try:
            os.remove(config_dir + "Mask.npy")
        except Excemption:
            pass
        
        mask_to_be_copied = os.path.join(config_dir, config_type, "Mask.npy" )
        # shutil.copy(config_dir + f"/{config_type}/" + "Mask.npy" , config_dir)
        shutil.copy(mask_to_be_copied , config_dir)
        
        yield from bps.mv(det_motor, det_position)
    
    _md = {}
    _md.update(md or {})
    _md.update(measurement_data())
    
    @bpp.stage_decorator([det])
    @bpp.run_decorator(md=_md)
    def trigger_at_TwoPositions():

        ## Switch to measurment of PDF
        set_glbl(acq_config['frame_acq_times'][0], acq_config['dark_win'][0])
        yield from switch_acquisition_type(acq_config['det_position'][0], 'PDF', acq_config['config_dir'])
        yield from configure_area_det2(det, acq_config['exposure_sec'][0])
        yield from bps.sleep(2)
        yield from bps.trigger(det, wait=True)
        yield from bps.create(name="PDF")
        reading = (yield from bps.read(det))
        # print(f"reading = {reading}")
        # ret.update(reading)
        yield from bps.save()

        ## rest before switch to XRD
        yield from bps.sleep(switch_rest)
        
        ## Switch to measurment of XRD
        set_glbl(acq_config['frame_acq_times'][1], acq_config['dark_win'][1])
        yield from switch_acquisition_type(acq_config['det_position'][1], 'XRD', acq_config['config_dir'])
        yield from configure_area_det2(det, acq_config['exposure_sec'][1])
        yield from bps.sleep(2)
        yield from bps.trigger(det, wait=True)
        yield from bps.create(name="XRD")
        reading = (yield from bps.read(det))
        # print(f"reading = {reading}")
        # ret.update(reading)
        yield from bps.save()


    yield from trigger_at_TwoPositions()




    
    