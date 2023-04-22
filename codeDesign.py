import datasetAlarm as dsa
import pandas as pd
import csv

dtypes = {
    "timestamp": "category",
    "alarm": "category",
    "serial": "category",
}

if __name__ == "__main__":
    
    my_params = {}
    my_params['windows_input']=[480,960,240]
    my_params['windows_output']=[240]
    my_params['min_counts']=[3]
    my_params['sigmas']=[3]
    my_params['offsets']=[230]
    my_params['verbose']=True
    #filename = "alarms.csv"
    #names = ["timestamp", "alarm", "serial"]
    #data = pd.read_csv(filename, names=names)
    #print(data.shape)
    params = []
    '''
    df = pd.read_csv(
        "groupby-data/legislators-historical.csv",
        dtype=dtypes,
        usecols=list(dtypes) + ["birthday", "last_name"],
        parse_dates=["birthday"]
    )
    '''
    df = pd.read_csv(
        r'alarms.csv',
        dtype= dtypes,
        delimiter= ','
    )
    print(type(df))
    #create_datasets(params_list, start_point=0)
    #several combinations of parameters are created, each combination can be considered as a new dataset
    #params_list = dsa.create_params_list(data_path="", params=my_params, verbose=True)
    dsa.create_datasets(df, my_params, start_point=0, file_tag='all_alarms')