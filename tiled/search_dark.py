from tiled.client import from_uri, from_profile
from tiled.queries import Key
from datetime import datetime

import numpy as np
import os, glob
import pandas as pd


# tiled_client = from_profile('xpd')

def search_dark(uid_incorrect_list:list, uid_candidate_list:list, tiled_catalog_name:str ='xpd'):
    tiled_client = from_profile(tiled_catalog_name)

    ## Remove uid_incorrect_dark from uid_candidate_list
    for uid_incorrect_dark in uid_incorrect_list:
        [uid_candidate_list.remove(uid) for uid in uid_candidate_list if uid_incorrect_dark in uid]


    ## Remove uid which detectors are not consistent from uid_candidate_list
    for uid in uid_candidate_list:
        run = tiled_client[uid]
        scan_ad = run.start['detectors'][0]
        
        dark_uid = run.start['sc_dk_field_uid']
        dark_run = tiled_client[dark_uid]
        dark_ad = dark_run.start['detectors'][0]

        if scan_ad != dark_ad:
            uid_candidate_list.remove(uid)


    ## Make time and detector list for uid_candidate_list
    time_candidate_list = []
    detector_candidate_list = []
    for uid in uid_candidate_list:
        run = tiled_client[uid]
        tt = run.start['time']
        ad = run.start['detectors'][0]  ## ad: area detector
        time_candidate_list.append(tt)
        detector_candidate_list.append(ad)

    time_candidate_array = np.asarray(time_candidate_list)
    detector_candidate_array = np.asarray(detector_candidate_list)


    ## Find minimum time difference for incorrect dark scans
    uid_mintime_list = []
    for uid_incorrect_dark in uid_incorrect_list:
        run_incorrect_dark = tiled_client[uid_incorrect_dark]
        time_incorrect_dark = run_incorrect_dark.start['time']
        time_incorrect_dark = np.asarray(time_incorrect_dark)

        time_differ = np.abs(time_incorrect_dark - time_candidate_array)
        
        # ## Only consider the dark taken before the scan
        # time_differ[time_differ < 0] = 999999999.0

        ## Make sure the detector is consistent
        detector_incorrect_dark = np.asarray(run_incorrect_dark.start['detectors'][0])
        detector_bool = detector_candidate_array != detector_incorrect_dark
        time_differ[detector_bool] = 999999999.0

        min_time_index = np.argmin(time_differ)
        print(f'Find mintime = {time_differ[min_time_index]} for incorrect dark uid = {uid_incorrect_dark}')

        uid_mintime_list.append(uid_candidate_list[min_time_index])

    return uid_mintime_list
    # return uid_candidate_list[min_time_index], time_incorrect_dark, time_candidate_array, time_differ














