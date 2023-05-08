import datasetAlarm as dsa
import pandas as pd
import pickle
import csv


ROOT = './'

DATA_PATH = ROOT + 'processed'

PARAMS = {'windows_input' : [1720],
          'windows_output' : [480],
          'min_counts' : [20],
          'sigmas' : [3],
          'offsets' : [60]}

RAW_DATA_FILE_PATH = f'alarms_log_data/raw/alarms.csv'

# load raw data
data = pd.read_csv(RAW_DATA_FILE_PATH, index_col=0, header=0, parse_dates=True)

 # create dataset params
params_list = dsa.create_params_list(DATA_PATH, PARAMS)

dsa.create_datasets(data, params_list)
#store_path = "../data/MACHINE_TYPE_00_alarms_window_input_1720_window_output_480_offset_60_min_count_20_sigma_3"
store_path = "./processed/MACHINE_TYPE_00_alarms_window_input_1720_window_output_480_offset_60_min_count_20_sigma_3"
filename= "all_alarms.pickle"
dsa.convert_to_json(store_path, filename)
'''
# Load the data from the pickle file
with open('./processed/MACHINE_TYPE_00_alarms_window_input_1720_window_output_480_offset_60_min_count_20_sigma_3/all_alarms.pickle', 'rb') as f:
    data = pickle.load(f)

# Separate the data into four vectors
x_train = data['x_train']
y_train = data['y_train']
x_val = data['x_val'] 
y_val = data['y_val']
'''