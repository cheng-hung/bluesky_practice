import numpy as np
import matplotlib.plot as plt
import pandas as pd
import os
import glob

from bluesky.utils import ts_msg_hook
import bluesky.preprocessors as bpp

def pdf_xrd_PE(det, *args, md=None, **kwargs):

    ## Measure PDF and XRD by moving det

    _md = {}
    _md.update(md or {})
    @bpp.stage_decorator([det])
    @bpp.run_decorator(md=_md)
    def trigger_at_TwoPositions():

        ## Moving to position of PDF
        yield from bps.trigger(det, wait=True)
        yield from bps.create(name="PDF")
        reading = (yield from bps.read(det))
        # print(f"reading = {reading}")
        # ret.update(reading)
        yield from bps.save()


        ## Moving to position of XRD
        yield from bps.trigger(det, wait=True)
        yield from bps.create(name="XRD")
        reading = (yield from bps.read(det))
        # print(f"reading = {reading}")
        # ret.update(reading)
        yield from bps.save()


    yield from trigger_at_TwoPositions()




    
    