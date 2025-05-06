from tiled.client import from_profile
tiled_client = from_profile("pdf")

uid_list = []


def plot_raw_pilatus(uid_list, vmax=1000, vmin=0):
    for uid in uid_list:
        run = tiled_client[uid]
        data = run.primary.read()
        array_data = data['pilatus1_image'][0][0]
        plt.figure()
        plt.imshow(array_data, vmax=vmax, vmin=vmin)
    
    
    
