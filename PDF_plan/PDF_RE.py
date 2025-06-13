import bluesky.preprocessors as bpp
import bluesky.plan_stubs as bps
from bluesky.plans import count
from xpdacq.beamtime import _configure_area_det



def measurement_data(temperature=False): # .......Captures metadata   
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
    
    if temperature:
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




def pdf_RE(det, *args, md=None, **kwargs):
  
    _md = {}
    _md.update(md or {})
    # _md.update(measurement_data())
    
    @bpp.stage_decorator([det])
    @bpp.run_decorator(md=_md)
    def trigger_det():
        ret = {}
        # set_glbl(acq_config['frame_acq_times'][0], acq_config['dark_win'][0])
        # yield from configure_area_det2(det, acq_config['exposure_sec'][0])
        # yield from bps.sleep(2)
        yield from bps.trigger(det, wait=True)
        yield from bps.create(name="PDF_RE")
        reading = (yield from bps.read(det))
        # print(f"reading = {reading}")
        ret.update(reading)
        yield from bps.save()

    yield from trigger_det()
    
    

def pdf_RE2(det, *args, md=None, exposure_time=5.0, **kwargs):
  
    _md = {}
    _md.update(md or {})
    # _md.update(measurement_data())
    
    yield from configure_area_det2(det, exposure_time)
    
    @bpp.stage_decorator([det])
    @bpp.run_decorator(md=_md)
    def trigger_det():
        ret = {}
        # set_glbl(acq_config['frame_acq_times'][0], acq_config['dark_win'][0])
        # yield from configure_area_det2(det, exposure_time)
        # yield from bps.sleep(2)
        yield from bps.trigger(det, wait=True)
        yield from bps.create(name="PDF_RE")
        reading = (yield from bps.read(det))
        # print(f"reading = {reading}")
        ret.update(reading)
        yield from bps.save()

    yield from periodic_dark(trigger_det())

    

def pdf_RE3(dets, exposure):
    """
    Take one reading from area detector with given exposure time

    Parameters
    ----------
    dets : list
        list of 'readable' objects. default to area detector
        linked to xpdAcq.
    exposure : float
        total time of exposrue in seconds

    Notes
    -----
    area detector being triggered will  always be the one configured
    in global state. To find out which these are, please using
    following commands:

        >>> xpd_configuration['area_det']

    to see which device is being linked
    """

    pe1c, = dets
    md = {}
    # setting up area_detector
    (num_frame, acq_time, computed_exposure) = yield from _configure_area_det(
        exposure
    )
    area_det = xpd_configuration["area_det"]
    # update md
    _md = ChainMap(
        md,
        {
            "sp_time_per_frame": acq_time,
            "sp_num_frames": num_frame,
            "sp_requested_exposure": exposure,
            "sp_computed_exposure": computed_exposure,
            "sp_type": "bps.trigger",
            "sp_uid": str(uuid.uuid4()),
            "sp_plan_name": "pdf_RE3",
        },
    )
    
    
    def pdf_RE_inner(det, *args, md=None, **kwargs):
        _md = {'detectors':[detector.name for detector in det]}
        _md.update(md or {})

        @bpp.stage_decorator([det])
        @bpp.run_decorator(md=_md)
        def trigger_det():
            ret = {}
            yield from bps.trigger(det, wait=True)
            yield from bps.create(name="PDF_RE")
            reading = (yield from bps.read(det))
            # print(f"reading = {reading}")
            ret.update(reading)
            yield from bps.save()

        yield from periodic_dark(trigger_det())
    
    # plan = bp.count([area_det], md=_md)
    # yield from plan
    yield from pdf_RE_inner(area_det, md=_md)


    
    
    
def pdf_RE4(dets, exposure, areaDet_name='pe1c', metadata={}, det_x_pos=None, det_y_pos=None, det_z_pos=None):
    
    """
    Take one reading from area detector with given exposure time

    Parameters
    ----------
    dets : list
        list of 'readable' objects. default to area detector
        linked to xpdAcq.
    exposure : float
        total time of exposrue in seconds

    Notes
    -----
    area detector being triggered will  always be the one configured
    in global state. To find out which these are, please using
    following commands:

        >>> xpd_configuration['area_det']

    to see which device is being linked
    """
    
    pe1c, = dets

    # change detector in configuration to the given one
    if xpd_configuration["area_det"].name == areaDet_name:
        pass
    else:
        if areaDet_name == 'pe1c':
            xpd_configuration["area_det"] = pe1c
        elif areaDet_name == 'pe2c':
            xpd_configuration["area_det"] = pe2c
        elif areaDet_name == 'pilatus1':
            areaDet_name["area_det"] = pilatus1
            
    area_det = xpd_configuration["area_det"]       
    

    # TODO: check if _configure_area_det for pilatus
    # setting up area_detector
    (num_frame, acq_time, computed_exposure) = yield from _configure_area_det(exposure)
    
    # update md
    md = metadata
    _md = ChainMap(
        md,
        {
            "sp_time_per_frame": acq_time,
            "sp_num_frames": num_frame,
            "sp_requested_exposure": exposure,
            "sp_computed_exposure": computed_exposure,
            "sp_type": "bps.trigger",
            "sp_uid": str(uuid.uuid4()),
            "sp_plan_name": "pdf_RE4",
            "sp_detector": area_det.name, 
        },
    )
    
    @bpp.stage_decorator([area_det])
    @bpp.run_decorator(md=_md)
    def pdf_RE_inner(area_det):
        # _md = {'detectors':[det.name]}
        # _md.update(md or {})
        
        def trigger_det(stream_name):
            ret = {}
            yield from bps.trigger(area_det, wait=True)
            yield from bps.create(name=stream_name)
            reading = (yield from bps.read(area_det))
            # print(f"reading = {reading}")
            ret.update(reading)
            yield from bps.save()

        if area_det.name == 'pe1c':
            yield from periodic_dark(trigger_det(f"PDF_{area_det.name}"))
                
        elif area_det.name == 'pilatus1':
            if det_x_pos==None and det_y_pos==None:
                det_x_pos = [40.644, 31.356, 36]
                det_y_pos = [-3.356, -12.644, -8]
            else:
                pass
            
            for i in range(len(det_x_pos)):
                # Grid_X.move(det_x[i])
                # Grid_Y.move(det_y[i])
                yield from bps.mv(Grid_X, det_x_pos[i], Grid_Y, det_y_pos[i])
                yield from periodic_dark(trigger_det(f"PDF_{area_det.name}_{i}"))
                    
                
    yield from pdf_RE_inner(area_det)
    