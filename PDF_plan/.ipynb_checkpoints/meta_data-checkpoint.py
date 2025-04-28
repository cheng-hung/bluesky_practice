import pandas as pd
def meta_data(smpi):
    data_dir = "/nsls2/data3/pdf/pdfhack/legacy/processed/xpdacq_data/user_data/tiff_base/" + list(bt.samples.keys())[smpi] + "/"
    import yaml
    Tiff, IQ, I_2Theta, Gr, Meta = [], [], [], [], []
    
    #-----------'''Load files into lists multiple ''' ------------
    for file in os.listdir(data_dir + "dark_sub/"): 
        if file.endswith(".tiff"):
            Tiff.append(file)
    Tiff.sort()

    for file in os.listdir(data_dir + "integration/"):
        if file.endswith("q.chi"):
            IQ.append(file)
    IQ.sort()

    for file in os.listdir(data_dir + "integration/"):
        if file.endswith("tth.chi"):
            I_2Theta.append(file)
    I_2Theta.sort()

    for file in os.listdir(data_dir + "pdf/"):
        if file.endswith("gr"):
            Gr.append(file)
    Gr.sort()

    for file in os.listdir(data_dir + "meta/"):
        if file.endswith("yaml"):
            Meta.append(file)
    Meta.sort()
    
    # ------- '''Ammend Temperature in file headers of IQ, Itth, and Gr files'''-------------
    Time, Temperature, file_name = [], [], []
    for i in range(len(Meta)):
        with open(data_dir + "meta/" + Meta[i], 'r') as f:
            data = yaml.unsafe_load(f)
        T = data.get('more_info').get('Temperature')
        Temperature.append(T)
        Time.append(data.get('more_info').get('Measurement_time'))
        file_name.append(I_2Theta[i])

        line_index = 0 
        lines = None
        with open(data_dir + "integration/" + IQ[i], 'r') as file_handler: 
            lines = file_handler.readlines()       
        lines.insert(line_index, '# /* '+'\n') 
        lines.insert(line_index+1, '# Temp (K) =              '+str(float(T))+'\n')
        lines.insert(line_index+2, '# */ '+'\n')
        with open(data_dir + "integration/" + IQ[i], 'w') as file_handler: 
            file_handler.writelines(lines) 

        with open(data_dir + "integration/" + I_2Theta[i], 'r') as file_handler: 
            lines = file_handler.readlines()       
        lines.insert(line_index, '# /* '+'\n') 
        lines.insert(line_index+1, '# Temp (K) =              '+str(float(T))+'\n')
        lines.insert(line_index+2, '# */ '+'\n')
        with open(data_dir + "integration/" + I_2Theta[i], 'w') as file_handler: 
            file_handler.writelines(lines) 

        line_index = 17
        lines = None
        with open(data_dir + "pdf/" + Gr[i], 'r') as file_handler: 
            lines = file_handler.readlines()       
        lines.insert(line_index, 'temperature = '+ str(float(T))+ ' K'+'\n') 
        with open(data_dir + "pdf/" + Gr[i], 'w') as file_handler: 
            file_handler.writelines(lines) 

    plt.figure()
    plt.plot((np.round(Time)-np.round(Time[0]))/3600, Temperature, 'r.')
    plt.xlabel('Time (h)')
    plt.ylabel('Temperature (K)')

        # Create a DataFrame
    df = pd.DataFrame({
        'Filename': file_name,
        'Value': Temperature
    })

    # Save to a text file (tab-separated format)
    file_path = data_dir + "File_name_vs_Temperature.csv"
    df.to_csv(file_path, sep='\t', index=False, header=False)

    print(f"Data saved to {file_path}")