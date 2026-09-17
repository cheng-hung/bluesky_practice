import numpy as np
from tiled.client import from_profile
import os
import matplotlib.pyplot as plt


num_positions = 3
beamline_acronym = 'pdf'
# uid = '9f839908-d684-40eb-9c62-b8739eb881a9'
# uid = '48d412f8-b663-4a24-8aaa-a6073765c7b9'
uid = 'e376b3be-affa-4fdf-9cd2-285f8e66e8ca'
detector_motors = ['Grid_X', 'Grid_Y']
pixel_size = 0.172 ## mm




tiled_client = from_profile(beamline_acronym)
run = tiled_client[uid]
stream_name = list(run.stop['num_events'].keys())
data_keys = list(run[stream_name[0]].read().keys())
img_key = [key for key in data_keys if 'image' in key][0]

pos_x = []
pos_y = []

my_im1 = np.float32(getattr(run, stream_name[0]).read()[img_key].to_numpy()[0][0])
x_size = my_im1.shape[0]  ## pixels
y_size = my_im1.shape[1]  ## pixels
my_im  = np.zeros([x_size, y_size, num_positions])

masks_pos_flist = ['Mask_pos1_ext_BS.npy', 'Mask_pos2_ext_BS.npy', 'Mask_pos3_ext_BS.npy']
mask_dir = '/nsls2/data/pdf/pdfhack/legacy/processed/xpdacq_data/user_data_Abeykoon_lambda_317090_af95b56d_2026-04-15-1032/config_base/pilatus_PDF'
user_mask = np.zeros([x_size, y_size, num_positions])

for i in range(num_positions):
    ## Read detector motor positions into pos_x, pos_y
    x = run[stream_name[i]].read()[detector_motors[0]].to_numpy()[0]
    y = run[stream_name[i]].read()[detector_motors[1]].to_numpy()[0]
    
    ## Image xy and motor xy are reversed since python is row first which in image is y.
    pos_y.append(float(x))
    pos_x.append(float(y))
    
    ## Read different position images into zeros array
    img =  np.float32(run[stream_name[i]].read()[img_key].to_numpy()[0][0])
    my_im[:,:,i] = img

    ## Read mask file into zeros array
    m_path = os.path.join(mask_dir, masks_pos_flist[i])
    mask = np.load(m_path)
    user_mask[:,:,i] = mask

pos_x = np.round(pos_x, decimals=3)  ## mm
pos_y = np.round(pos_y, decimals=3)  ## mm


## sort order according to the detector x position
sort_idx = np.argsort(pos_x)
sort_pos_x = pos_x[sort_idx]
sort_pos_y = pos_y[sort_idx]
sort_my_im = my_im[:,:,sort_idx]
sort_user_mask = user_mask[:,:,sort_idx]

osetx_total = abs(round((sort_pos_x[-1]-sort_pos_x[0])/pixel_size))  ## pixels
osety_total = abs(round((sort_pos_y[-1]-sort_pos_y[0])/pixel_size))  ## pixels

center_x = (sort_pos_x[0]+sort_pos_x[-1])/2  ## mm
center_y = (sort_pos_y[0]+sort_pos_y[-1])/2  ## mm

my_imsum = np.ones((x_size+osetx_total, y_size+osety_total, 3))*np.nan
x_sum_size = my_imsum.shape[0] ## pixels
y_sum_size = my_imsum.shape[1] ## pixels

x_sum_center = round((x_sum_size+1)/2)  ## pixel
y_sum_center = round((y_sum_size+1)/2)  ## pixel

for i in range(num_positions):
    x_offset = round((sort_pos_x[i] - center_x)/pixel_size)  ## pixels
    y_offset = round((sort_pos_y[i] - center_y)/pixel_size)  ## pixels

    start_x = int(x_sum_center - x_size/2) + x_offset  ## pixel
    end_x = int(x_sum_center + x_size/2) + x_offset    ## pixel

    start_y = int(y_sum_center - y_size/2) + y_offset  ## pixel
    end_y = int(y_sum_center + y_size/2) + y_offset    ## pixel

    my_imsum[start_x:end_x, start_y:end_y, i] = sort_my_im[:,:,i]
    my_imsum[start_x:end_x, start_y:end_y, i][sort_user_mask[:,:,i]==1] = np.nan

    # plt.figure()
    # plt.imshow(my_imsum[:,:,i], vmin=0, vmax=50, alpha=0.5)


stitch_img = np.nanmean(my_imsum, axis=2, dtype=np.float32)
plt.figure()
plt.imshow(stitch_img, vmin=0, vmax=1000, alpha=1.0)

