from dask import dataframe as dd
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

data_prep_start = time.time()
print("Data preparation start time: ", datetime.now())

df1 = pd.read_csv('/work/data/csv_dir/sample_dataset_1.csv')
df1['event_time1'] = pd.to_datetime(df1['event_time1'])
df1=df1.sort_values(by=['event_time1'])

df2 = pd.read_csv('/work/data/csv_dir/sample_dataset_2.csv')
df2['event_time2'] = pd.to_datetime(df2['event_time2'])
df2=df2.sort_values(by=['event_time2'])

#convert to parquet format with 16 & 8 partitions
ddf1 = dd.from_pandas(df1, npartitions=16)
ddf1.to_parquet("/work/data/parquet_partitioned_dir_1/", write_index=False, engine='fastparquet', times='int96', overwrite=True)
print("Listing the contents of the parquet file directory 1:")
print(os.listdir('/work/data/parquet_partitioned_dir_1'))

ddf2 = dd.from_pandas(df2, npartitions=8)
ddf2.to_parquet("/work/data/parquet_partitioned_dir_2/", write_index=False, engine='fastparquet', times='int96', overwrite=True)
print("Listing the contents of the parquet file directory 2:")
print(os.listdir('/work/data/parquet_partitioned_dir_2'))

data_prep_end = time.time()
print("Data preparation end time: ", datetime.now())
print("Time taken for data preparation: ",(data_prep_end - data_prep_start),"sec")
