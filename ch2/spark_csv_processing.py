# Importing packages
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F #, types as T
from pyspark.sql.functions import col, pandas_udf, monotonically_increasing_id, sqrt
import pandas as pd
from pyspark.ml import Pipeline,PipelineModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.types import *
from pyspark.ml.functions import vector_to_array
import time
from datetime import datetime
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import math
def multiply_columns_func(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a * b)
multiplyColumns = pandas_udf(multiply_columns_func, returnType=DoubleType())


print("Spark csv processing start time: ", datetime.now())

start = time.time()

spark = SparkSession.builder.master("local[*]").config("spark.driver.host", "localhost").config("spark.sql.shuffle.partitions", "3000").config("spark.sql.broadcastTimeout", "36000").appName('PySpark Read CSV').config("spark.driver.memory", "230g").config("spark.network.timeout", "10000000").config("spark.executor.memory", "230g").config("spark.driver.maxResultSize", "230g").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df1 = spark.read.options(header="True", inferSchema="True").csv("/work/data/csv_dir/sample_dataset_1.csv")
df1=df1.sort("event_time1")

df2 = spark.read.options(header="True", inferSchema="True").csv("/work/data/csv_dir/sample_dataset_2.csv")
df2=df2.sort("event_time2")

columns_to_scale1 = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']
assemblers1 = [VectorAssembler(inputCols=[col], outputCol=col + "_vec") for col in columns_to_scale1]
scalers1 = [StandardScaler(inputCol=col + "_vec", outputCol=col + "_scaled") for col in columns_to_scale1]
pipeline1 = Pipeline(stages=assemblers1 + scalers1)
scalerModel1 = pipeline1.fit(df1)
scaledData1 = scalerModel1.transform(df1)
firstelement=F.udf(lambda v:float(v[0]),FloatType())
scaledData1 = scaledData1.withColumn("C1_scaled", firstelement("C1_scaled")).withColumn("C2_scaled", firstelement("C2_scaled")).withColumn("C3_scaled", firstelement("C3_scaled")).withColumn("C4_scaled", firstelement("C4_scaled")).withColumn("C5_scaled", firstelement("C5_scaled")).withColumn("C6_scaled", firstelement("C6_scaled")).withColumn("C7_scaled", firstelement("C7_scaled")).withColumn("C8_scaled", firstelement("C8_scaled")).withColumn("C9_scaled", firstelement("C9_scaled")).withColumn("C10_scaled", firstelement("C10_scaled"))

columns_to_scale2 = ['C21', 'C22', 'C23', 'C24', 'C25']
assemblers2 = [VectorAssembler(inputCols=[col], outputCol=col + "_vec") for col in columns_to_scale2]
scalers2 = [StandardScaler(inputCol=col + "_vec", outputCol=col + "_scaled") for col in columns_to_scale2]
pipeline2 = Pipeline(stages=assemblers2 + scalers2)
scalerModel2 = pipeline2.fit(df2)
scaledData2 = scalerModel2.transform(df2)
scaledData2 = scaledData2.withColumn("C21_scaled", firstelement("C21_scaled")).withColumn("C22_scaled", firstelement("C22_scaled")).withColumn("C23_scaled", firstelement("C23_scaled")).withColumn("C24_scaled", firstelement("C24_scaled")).withColumn("C25_scaled", firstelement("C25_scaled"))

# create the UDFs for feature computations

@pandas_udf("double")
def compute_feature12(series1, series2):
	results = []
	for a, b in zip(series1, series2):
		if (a > 0.65) or (b > 0.65):
			result =  a + b
		elif (a < 0.35) or (b > 0.35):
			result = a - b
		else:
			result = (a + b)/2
		results.append(result)
	return pd.Series(results)

@pandas_udf("double")
def compute_feature13 (series1, series2, series3):
	results = []
	for a, b, c in zip(series1, series2, series3):
	   	if (a > 0.5):
	   		result =  abs(b - c)
	   	elif (a < 0.25):
	   		result = b * c
	   	else:
	   		result = math.sqrt(abs(a + c))
	   	results.append(result)
	return pd.Series(results)


@pandas_udf("double")
def compute_feature22 (series1, series2):
	results = []
	for a, b in zip(series1, series2):
		if (a < 0.25):
			result = a + b
		elif (a >= 0.25) and (a < 0.5):
			result = a * a
		elif (a >= 0.5) and (a < 0.75):
			result = a * b
		else:
			result = (a + b)/2
		results.append(result)
	return pd.Series(results)

@pandas_udf("double")
def compute_feature23 (series1, series2, series3):
	results = []
	for a, b, c in zip(series1, series2, series3):
   		if (a > 0.5):
   			result = abs(b + c)
   		elif (a < 0.25):
   			result = a * c
   		else:
   			result = math.sqrt(abs(a + c))
   		results.append(result)
	return pd.Series(results)

df_result1=scaledData1.withColumn("feature11", multiplyColumns(col("C1_scaled"), col("C2_scaled")))
df_result1 = df_result1.withColumn("feature12", compute_feature12(df_result1["C2_scaled"], df_result1["C4_scaled"]))
df_result1 = df_result1.withColumn("feature13", compute_feature13(df_result1["C6_scaled"], df_result1["C8_scaled"], df_result1["C10_scaled"]))
#df_result1.show()

df_result2=scaledData2.withColumn("feature2", multiplyColumns(col("C21_scaled"), col("C22_scaled")))
df_result2 = df_result2.withColumn("feature22", compute_feature22(df_result2["C21_scaled"], df_result2["C22_scaled"]))
df_result2 = df_result2.withColumn("feature23", compute_feature23(df_result2["C23_scaled"], df_result2["C24_scaled"], df_result2["C25_scaled"]))
#df_result2.show()

# For sorting
df_result2 = df_result2.withColumn("seq_no", F.monotonically_increasing_id())

df_final=df_result2.join(df_result1, df_result2.event_time2 ==  df_result1.event_time1,"inner").orderBy("seq_no")
df_final = df_final.drop(df_final.seq_no)
column_names = df_final.columns

selected = [s for s in column_names if '_vec' not in s]
df_final=df_final.select(selected)

df_final.write.mode("overwrite").option("header",True).csv("/work/data/spark_output_csv")
end = time.time()

# print the total processing time taken
print("Total time: {:.3f}".format(end-start)," sec")
print("Spark csv processing end time: ", datetime.now())
