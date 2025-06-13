import os
import datetime
import pprint
import uuid
# from bluesky_kafka import RemoteDispatcher
from bluesky_kafka.consume import BasicConsumer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import databroker
from tiled.client import from_profile
tiled_client = from_profile('pdf')

import importlib
## import stitching modules maade by AM
pilasum = importlib.import_module("pilatus_sum_AM")

"--------------------------USER INPUTS------------------------------"
## Defined top layer folders
user_data = '/nsls2/data3/pdf/pdfhack/legacy/processed/xpdacq_data/user_data'
tiff_base_path = os.path.join(user_data, 'tiff_base')
config_base_path = os.path.join(user_data, 'config_base')

## Variables to sum and stitch images
masks_path = os.path.join(config_base_path,'pilatus_mask')
masks_pos_flist = ['Mask_pos1_ext_BS.npy', 'Mask_pos2_ext_BS.npy', 'Mask_pos3_ext_BS.npy']
osetx = 27
osety = 27

## Varibles for pyFai integration
poni_fn = os.path.join(config_base_path, 'merged_PDF', 'merged_dioptas.poni')
stitched_mask_fn = os.path.join(config_base_path, 'pilatus_mask', 'stitched_2.npy')
binning = 5000
polarization = 0.99
UNIT = "q_A^-1"
ll=1
ul=99

## Variables for pdfgetx3
## pdfgetx3 config file (.cfg) and bkg file (.chi) should saved in '/config_base/pdfgetx'
do_reduction = True
cfg_name = 'pdfgetx3.cfg'
bkg_name = 'A_emptycap_0p5mm_20250609-173153_c3cfeb_primary-1_mean_q.chi'
cfg_fn = os.path.join(config_base_path, 'pdfgetx', cfg_name)
bkg_fn = os.path.join(config_base_path, 'pdfgetx', bkg_name)

"--------------DO NOT TOUCH BELOW!! Unless CHLin said OK!-----------"

try:
    from nslsii import _read_bluesky_kafka_config_file  # nslsii <0.7.0
except (ImportError, AttributeError):
    from nslsii.kafka_utils import _read_bluesky_kafka_config_file  # nslsii >=0.7.0

# these two lines allow a stale plot to remain interactive and prevent
# the current plot from stealing focus.  thanks to Tom:
# https://nsls2.slack.com/archives/C02D9V72QH1/p1674589090772499
plt.ion()
plt.rcParams["figure.raise_window"] = False


def print_kafka_messages(beamline_acronym_01, beamline_acronym_02, 
                         tiff_base_path=tiff_base_path, 
                         masks_path=masks_path, 
                         masks_pos_flist=masks_pos_flist, 
                         osetx=osetx, osety=osety, 
                         poni_fn=poni_fn,
                         stitched_mask_fn=stitched_mask_fn, 
                         binning=binning, 
                         polarization=polarization, 
                         UNIT=UNIT, ll=ll, ul=ul, 
                         do_reduction=do_reduction, 
                         cfg_fn=cfg_fn, bkg_fn=bkg_fn, 
                         ):
    
    print(f"Listening to Kafka messages for {beamline_acronym_01}")
    print(f"Listening to Kafka messages for {beamline_acronym_02}")


    def print_message(consumer, doctype, doc):
        name, message = doc
        # print(
        #     f"\n{datetime.datetime.now().isoformat()} document: {name}\n"
        # #     f"\ndocument keys: {list(message.keys())}\n"
        # #     f"\ncontents: {pprint.pformat(message)}\n"
        # )

        if (name == 'start') and ('topic' not in message):
            print(
                f"\n\n{datetime.datetime.now().isoformat()} documents {name}\n"
                # f"document keys: {list(message.keys())}"
                f"\n{message['uid'] = }\n")
            try:
                print(f"\n{message['topic'] = }\n")
            except KeyError:
                print(f"\nThis document has no topic.\n")
            
            global uid, sample_name
            uid = message['uid']
            sample_name = message['sample_name']
                

        # elif name == 'event':
        #     print(
        #         f"\n{datetime.datetime.now().isoformat()} documents {name}\n"
        #         f"\ndocument keys: {list(message.keys())}\n"
        #         )
        #     try:
        #         print(f"\n{message['topic'] = }\n")
        #     except KeyError:
        #         print(f"\nThis document has no topic.\n")
        
        
        elif (name == 'stop') and ('topic' not in message):
            print(
                f"\n{datetime.datetime.now().isoformat()} documents {name}\n"
                f"\ndocument keys: {list(message.keys())}\n"
                f"\ncontents: {pprint.pformat(message['num_events'])}\n"
                )
            try:
                print(f"\n{message['topic'] = }\n")
            except KeyError:
                print(f"\nThis document has no topic.\n")

            global stream_name
            stream_name = list(message['num_events'].keys())
            print(f'\n{stream_name = }\n')
        
        
        elif (name == 'stop') and ('topic' in message) and (message['num_events']['primary']==3):
        #     # print('Kafka test good!!')
            print(
                f"\n{datetime.datetime.now().isoformat()} documents {name}\n"
                f"\ndocument keys: {list(message.keys())}\n"
                f"\ncontents: {pprint.pformat(message['num_events'])}\n"
                )
            try:
                print(f"\n{message['topic'] = }\n")
            except KeyError:
                print(f"\nThis document has no topic.\n")

            print(
                f"\nStart to stitch pilatus1 data uid = {uid}\n"
            )
            
            ## Sum three images at three positions
            full_imsum, sum_dir, saved_fn_prefix = pilasum.save_image_sum_T(
                                                uid, stream_name, sample_name, 
                                                osetx=osetx, osety=osety, 
                                                masks_path=masks_path, 
                                                masks_pos_flist=masks_pos_flist, 
                                                tiff_base_path=tiff_base_path, 
                                                )

            ## pyFai integration: 2D to 1D
            iq_data, iq_fn = pilasum.pct_integration(full_imsum, saved_fn_prefix, 
                                              binning=binning, 
                                              polarization=polarization, 
                                              UNIT=UNIT, ll=ll, ul=ul, 
                                              poni_fn=poni_fn, 
                                              mask_fn=stitched_mask_fn, 
                                              directory=sum_dir, 
                                              )

            ## Data reduction: I(Q) to G(r)
            if do_reduction:
                gr_path = pilasum.get_gr(iq_fn, cfg_fn, bkg_fn, 
                                         sum_dir, saved_fn_prefix)


            print('\n########### Events printing division ############\n')
      
           
        

    kafka_config = _read_bluesky_kafka_config_file(config_file_path="/etc/bluesky/kafka.yml")

    # this consumer should not be in a group with other consumers
    #   so generate a unique consumer group id for it
    unique_group_id = f"echo-{beamline_acronym_01}-{str(uuid.uuid4())[:8]}"

    kafka_consumer = BasicConsumer(
        topics=[f"{beamline_acronym_01}.bluesky.runengine.documents", 
                f"{beamline_acronym_02}.bluesky.runengine.documents"],
        bootstrap_servers=kafka_config["bootstrap_servers"],
        group_id=unique_group_id,
        consumer_config=kafka_config["runengine_producer_config"],
        process_message = print_message,
    )

    try:
        kafka_consumer.start_polling(work_during_wait=lambda : plt.pause(.1))
    except KeyboardInterrupt:
        print('\nExiting Kafka consumer')
        return()


if __name__ == "__main__":
    import sys
    print_kafka_messages(sys.argv[1], sys.argv[2])
