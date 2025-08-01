import datetime
import pprint
import uuid
# from bluesky_kafka import RemoteDispatcher
from bluesky_kafka.consume import BasicConsumer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tiled.client import from_profile


import importlib
Pilatus_sum = importlib.import_module("pilatus_sum").Pilatus_sum
Pilatus_Int = importlib.import_module("pilatus_int").Pilatus_Int
Pilatus_getpdf = importlib.import_module("pilatus_getpdf").Pilatus_Int
pilaplot = importlib.import_module("pilatus_plotter")

"--------------------------USER INPUTS------------------------------"
tiled_client = from_profile('pdf')
ini_config = '/nsls2/users/clin1/Documents/Git_BNL/bluesky_practice/PDF_plan/pilatus_kafka_2.2/pilatus_kafka_config.ini'

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
                         tiled_client, ini_config):
    
    print(f"Listening to Kafka messages for {beamline_acronym_01}")
    print(f"Listening to Kafka messages for {beamline_acronym_02}")

    # print("\n")
    # print(f"{masks_path = }\n")
    # print(f"{poni_fn = }\n")
    # print(f"{stitched_mask_fn = }\n")
    # print(f"{cfg_fn = }\n")
    # print(f"{bkg_fn = }\n")


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
                     
            uid = message['uid']
            pila_analyzer = Pilatus_getpdf(uid, tiled_client, ini_config)
                

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

            stream_name = list(message['num_events'].keys())
            pila_analyzer.stream_name = stream_name
            print(f'\n{stream_name = }\n')
        

        # elif (name == 'stop') and ('topic' in message) and (message['num_events']['primary']==3):
        elif (name == 'stop') and ('topic' in message):
            print(
                f"\n{datetime.datetime.now().isoformat()} documents {name}\n"
                f"\ndocument keys: {list(message.keys())}\n"
                f"\ncontents: {pprint.pformat(message['num_events'])}\n"
                )
            try:
                print(f"\n{message['topic'] = }\n")
            except KeyError:
                print(f"\nThis document has no topic.\n")

            plotter = pilaplot.plot_pilatus(pila_analyzer.sample_name)

            ## Sum three images at three positions
            if message['num_events']['primary']==3:
                print(f"\nStart to stitch {pila_analyzer.run.start['sp_detector']} data: uid = {pila_analyzer.uid}\n")
                full_imsum, process_dir = pila_analyzer.save_image_sum_T()
                mask_img = np.load(pila_analyzer.stitched_mask)
                plotter.plot_tiff(full_imsum, mask_img)

            ## Process pe1c data without stitching 
            elif pila_analyzer.use_flat_field_pe1c:
                print(f"\nStart to process {pila_analyzer.run.start['detectors'][0]} data: uid = {pila_analyzer.uid}\n")
                full_imsum, process_dir = pila_analyzer.flat_filed_pe1c()
                mask_img = np.load(pila_analyzer.mask_pe1c)
                plotter.plot_tiff2(full_imsum, mask_img)

            ## pyFai integration: 2D to 1D
            print(f"\nStart to do 2D integration: uid = {uid}\n")
            iq_df, iq_fn = pila_analyzer.pct_integration(full_imsum, process_dir)
            plotter.plot_iq(iq_fn)

            ## Data reduction: I(Q) to G(r)
            if pila_analyzer.do_reduction:
                print(f"\nStart to reduce sq, fq, gr: uid = {uid}\n")
                sqfqgr_path = pila_analyzer.get_gr(iq_df, process_dir)
                
                bkg_scale = pila_analyzer.pdfconfig().bgscale[0]
                bkg_fn = pila_analyzer.pdfconfig_dict['backgroundfile']
                plotter.plot_sqfqgr(sqfqgr_path, bkg_scale, bkg_fn)


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
