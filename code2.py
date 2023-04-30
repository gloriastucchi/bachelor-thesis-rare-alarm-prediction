import datasetTest as dsa
import pandas as pd
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
