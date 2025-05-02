import bluesky.preprocessors as bpp
import bluesky.plan_stubs as bps
from bluesky.plans import count
from xpdacq.beamtime import _configure_area_det


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
            "sp_type": "pdf_RE",
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

    