import bluesky.preprocessors as bpp
import bluesky.plan_stubs as bps



def pdf_RE(det, *args, md=None, **kwargs):
  
    _md = {}
    _md.update(md or {})
    # _md.update(measurement_data())
    
    @bpp.stage_decorator([det])
    @bpp.run_decorator(md=_md)
    def trigger_det():
        # set_glbl(acq_config['frame_acq_times'][0], acq_config['dark_win'][0])
        # yield from configure_area_det2(det, acq_config['exposure_sec'][0])
        # yield from bps.sleep(2)
        yield from bps.trigger(det, wait=True)
        yield from bps.create(name="PDF_RE")
        # reading = (yield from bps.read(det))
        # print(f"reading = {reading}")
        # ret.update(reading)
        yield from bps.save()

    yield from trigger_det()