import pandas as pd
import time
from datetime import datetime
import math
from sklearn.preprocessing import StandardScaler

print("Pandas csv processing start time: ", datetime.now())

#Feature computation functions
def compute_feature12 (row):
   if (row['C2_scaled'] > 0.65) or (row['C4_scaled'] > 0.65):
      return row['C2_scaled'] + row['C4_scaled']
   elif (row['C2_scaled'] < 0.35) or (row['C4_scaled'] > 0.35):
   	  return row['C2_scaled'] - row['C4_scaled']
   else:
   	  return (row['C2_scaled'] + row['C4_scaled'])/2

def compute_feature13 (row):
   if (row['C6_scaled'] > 0.5):
      return abs(row['C8_scaled'] - row['C10_scaled'])
   elif (row['C6_scaled'] < 0.25):
   	  return row['C8_scaled'] * row['C10_scaled']
   else:
   	  return math.sqrt(abs(row['C6_scaled'] + row['C10_scaled']))

def compute_feature22 (row):
   if (row['C21_scaled'] < 0.25):
      return row['C21_scaled'] + row['C22_scaled']
   elif (row['C21_scaled'] >= 0.25) and (row['C21_scaled'] < 0.5):
   	  return row['C21_scaled'] * row['C21_scaled']
   elif (row['C21_scaled'] >= 0.5) and (row['C21_scaled'] < 0.75):
   	  return row['C21_scaled'] * row['C22_scaled']
   else:
   	  return (row['C21_scaled'] + row['C22_scaled'])/2

def compute_feature23 (row):
   if (row['C23_scaled'] > 0.5):
      return abs(row['C24_scaled'] + row['C25_scaled'])
   elif (row['C23_scaled'] < 0.25):
   	  return row['C23_scaled'] * row['C25_scaled']
   else:
   	  return math.sqrt(abs(row['C23_scaled'] + row['C25_scaled']))

start = time.time()

# read the csv file 1
df1 = pd.read_csv('./csv_dir/sample_dataset_1.csv')
df1['event_time1'] = pd.to_datetime(df1['event_time1'])

# sort the values in dataset 1
df1=df1.sort_values(by=['event_time1'])
read_end_1 = time.time()
print("Read csv file 1 and sort : {:.3f}".format(read_end_1-start)," sec")
# print memory usage
print(df1.memory_usage())

# read the csv file 2
df2 = pd.read_csv('./csv_dir/sample_dataset_2.csv')
df2['event_time2'] = pd.to_datetime(df2['event_time2'])

# sort the values in dataset 2
df2=df2.sort_values(by=['event_time2'])
read_end_2 = time.time()
print("Read csv file 2 and sort : {:.3f}".format(read_end_2-read_end_1)," sec")
# print memory usage
print(df2.memory_usage())
#print(df1.head())
#print(df2.head())
# scale the values using standardscaler
scaler1 = StandardScaler()
columns_to_scale1 = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']
columns_scaled1 = ['C1_scaled', 'C2_scaled', 'C3_scaled', 'C4_scaled', 'C5_scaled', 'C6_scaled', 'C7_scaled', 'C8_scaled', 'C9_scaled', 'C10_scaled']
scaled_features = scaler1.fit_transform(df1[columns_to_scale1])
df11=pd.DataFrame(scaled_features, columns = columns_scaled1)
df1 = pd.concat([df1, df11], axis=1, join="inner")
scaling_end_1 = time.time()
print("Scaling values for dataset 1: ",(scaling_end_1-read_end_2)," sec")
#print(df1.head())
# print memory usage
print(df1.memory_usage())

# scale the values using standardscaler
scaler2 = StandardScaler()
columns_to_scale2 = ['C21', 'C22', 'C23', 'C24', 'C25']
columns_scaled2 = ['C21_scaled', 'C22_scaled', 'C23_scaled', 'C24_scaled', 'C25_scaled']
scaled_features = scaler2.fit_transform(df2[columns_to_scale2])
df21=pd.DataFrame(scaled_features, columns = columns_scaled2)
df2 = pd.concat([df2, df21], axis=1, join="inner")
scaling_end_2 = time.time()
print("Scaling values for dataset 2: {:.3f}".format(scaling_end_2-scaling_end_1)," sec")
#print(df2.head())
# print memory usage
print(df2.memory_usage())
# Process values for a new features
print("Starting feature computations")
df1['feature11'] = df1['C1_scaled']*df1['C2_scaled']
print("Completed feature11")
df1['feature12'] = df1.apply (lambda row: compute_feature12(row), axis=1)
print("Completed feature12")
df1['feature13'] = df1.apply (lambda row: compute_feature13(row), axis=1)
print("Completed feature13")
new_col_end_1 = time.time()
print("Computing values for features in dataset 1: {:.3f}".format(new_col_end_1 - scaling_end_2)," sec")
# print memory usage
#print(df1.memory_usage())

# Process values for a new features
df2['feature21'] = df2['C21_scaled']*df2['C22_scaled']
print("Completed feature21")
df2['feature22'] = df2.apply (lambda row: compute_feature22(row), axis=1)
print("Completed feature22")
df2['feature23'] = df2.apply (lambda row: compute_feature23(row), axis=1)
print("Completed feature23")

new_col_end_2 = time.time()
print("Computing values for features in dataset 2: {:.3f}".format(new_col_end_2 - new_col_end_1)," sec")
# print memory usage
#print(df2.memory_usage())
#print(df1.head())
#print(df2.head())

#Join df1 & df2
df1=df1.set_index("event_time1")
df2=df2.set_index("event_time2")
df3= pd.concat([df1, df2], axis=1, join="inner")

join_end = time.time()
print("Joining datasets completed: {:.3f}".format(join_end - new_col_end_2)," sec")
# print memory usage
#print(df3.memory_usage())

#print(df3.head(20))
#print(df3.columns)
# partition and write to 32 csv file
k = 32
size = math.ceil(df3.shape[0]/16)
for i in range(k):
    df = df3[size*i:size*(i+1)]
    df.to_csv(f'./pandas_output_csv_dir/part_{i+1}.csv')
writing_end = time.time()

print("Writing output to multiple csv files: {:.3f}".format(writing_end - join_end)," sec")
print("Total processing time: {:.3f}".format(writing_end-start)," sec")
print("Pandas csv processing end time: ", datetime.now())
