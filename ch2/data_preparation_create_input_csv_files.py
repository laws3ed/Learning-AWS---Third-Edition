import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

data_prep_start = time.time()
print("CSV Data preparation start time: ", datetime.now())

start_date1 = '2020-04-01'
end_date1 = '2023-04-01'

start_date2 = '2022-09-01'
end_date2 = '2023-04-01'

dates1 = pd.date_range(start=start_date1, end=end_date1, freq='S')
dates2 = pd.date_range(start=start_date2, end=end_date2, freq='S')

df1 = pd.DataFrame(data=np.random.randint(9999, 999999, size=(len(dates1), 10)),columns=['C1', 'C2', 'C3', 'C4', 'C5',
	'C6', 'C7', 'C8', 'C9', 'C10'])

df2 = pd.DataFrame(data=np.random.randint(9999, 999999, size=(len(dates2), 5)),columns=['C21', 'C22', 'C23', 'C24', 'C25'])  

df1["event_time1"] = dates1
df2["event_time2"] = dates2

df1 = df1.sample(frac = 1)
df2 = df2.sample(frac = 1)

print("df1 shape", df1.shape)
print("df2 shape", df2.shape)

df1.to_csv('/work/data/csv_dir/sample_dataset_1.csv', index=False)
df2.to_csv('/work/data/csv_dir/sample_dataset_2.csv', index=False)

print("Listing the contents of the csv file directory:")
print(os.listdir('/work/data/csv_dir'))

data_prep_end = time.time()
print("Time taken for csv data preparation: ",(data_prep_end - data_prep_start),"sec")
print("Data preparation end time: ", datetime.now())
