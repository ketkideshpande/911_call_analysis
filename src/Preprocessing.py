# This programs preprocesses and prepares data for input to regession 

# create the sparkSession
spark = SparkSession.builder \
    .master("local") \
    .appName("Spark Project") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

# read the data from hdfs
df = spark.read.format("csv").load("hdfs://worker2.hdp-internal:8020/user/sdeshpa1/911_final_processed.csv",
                                   header='true', inferSchema='true')

# show the first row of the dataframe
df.head()

# drop the first column of the dataframe
df = df.drop("_c0")

## Data Prepreprocesing ##

# Check count of null or missing values for each column
from pyspark.sql.functions import isnan

df.filter((df["CallNumber"] == "") | df["CallNumber"].isNull() | isnan(df["CallNumber"])).count()  # 0 records

df.filter(df.CallDateTime.isNull()).count()  # 0 records

df.filter(df.Priority.isNull()).count()  # 4 records

# Filter these null records from the dataframe
df = df.filter((df["Priority"].isNotNull()))

df.filter(df.Description.isNull()).count()  # 0 records

df.filter(df.IncidentLocation.isNull()).count()  # 0 records

df.filter(df.ZipCode.isNull()).count()  # 0 records

df.filter(df.Neighborhood.isNull()).count()

df.filter(df.PoliceDistrict.isNull()).count()  # 15,934 records

# Filter the null records for Police District and Neighborhood to standardize the data
df = df.filter((df["PoliceDistrict"].isNotNull()))
df = df.filter(df.Neighborhood.isNotNull())

# Add Time dimentions for getting based on CallDateTime

import datetime
from pyspark.sql.functions import year, month, dayofmonth, dayofweek, dayofyear, hour, minute, weekofyear

df = df.withColumn('year', year("CallDateTime")) \
    .withColumn('month', month("CallDateTime")) \
    .withColumn('dayofmonth', dayofmonth("CallDateTime")) \
    .withColumn('dayofweek', dayofweek("CallDateTime")) \
    .withColumn('dayofyear', dayofyear("CallDateTime")) \
    .withColumn('hour', hour("CallDateTime")) \
    .withColumn('minute', minute("CallDateTime")) \
    .withColumn('weekofyear', weekofyear("CallDateTime"))

# check to see if latitude is null
df.filter(df.latitude.isNull()).count()
df.filter(df.longitude.isNull()).count()

# consider only data which has geo loctaion specified
df = df.filter(df.latitude.isNotNull())

df_rolledup = df.groupBy("latitude", "longitude", "month", "dayofmonth").count()
df_rolledup = df_rolledup.select("count","latitude", "longitude", "month", "dayofmonth")


# convert to RDD first
df_rdd = df_rolledup.rdd

# Code for running a liner regression model on the same
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint

# FROM RDD OF TUPLE TO A RDD OF LABELEDPOINT for training and testing
df_libsvm = df_rdd.map(lambda line: LabeledPoint(line[0],line[1:]))

# SAVE AS LIBSVM
MLUtils.saveAsLibSVMFile(df_libsvm, "hdfs://worker2.hdp-internal:8020/user/sdeshpa1/data_final_test.txt")
:
