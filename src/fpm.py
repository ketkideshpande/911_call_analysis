#! /usr/bin/env python
# This program creates a data for frequent pattern mining. Saves it to HDFS.

from pyspark.sql import SparkSession, SQLContext
from pyspark.sql import functions as F
#from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import concat, col, lit, split
from pyspark.sql.types import *


spark = SparkSession \
.builder \
.appName("Frequent Pattern Mining Algorithm") \
.enableHiveSupport() \
.getOrCreate()


#def preprocess_data():
# read the input file into a dataframe
df = spark.read.format("csv").load("hdfs://worker2.hdp-internal:8020/user/ketkid1/911_new.csv",header = 'true' ,inferSchema = 'true')

#  drop few columns which are not required for this algorithn
df = df.drop("_c0","IncidentLocation","VRIZones","CallNumber")

# extract 'day of week' information and convert time into morning, aftrnoon, evening and night period and append to dataframe
newDf = df.withColumn('dow_number', F.date_format('CallDateTime', 'u')) \
            .withColumn('dow_string', F.date_format('CallDateTime', 'E'))  \
            .withColumn('period', F.when((F.date_format('CallDateTime', 'H') >= 6) & (F.date_format('CallDateTime', 'H') < 12), 'Morning') \
                            .when((F.date_format('CallDateTime', 'H') >= 12) & (F.date_format('CallDateTime', 'H') < 16), 'Afternoon') \
                            .when((F.date_format('CallDateTime', 'H') >= 16) & (F.date_format('CallDateTime', 'H') < 21), 'Evening') \
                            .when((F.date_format('CallDateTime', 'H') >= 21) & (F.date_format('CallDateTime', 'H') < 6), 'Night'))                         

# drop timestamp column once information is extracted
newDf = newDf.drop("CallDateTime")

# filter data, select only rows where priority is not equal to 'non-emergency'
selectColumnList = [ 
#    'Priority', 
    'Description',
#    'ZipCode',
#    'Neighborhood',
    'PoliceDistrict',
    'dow_string',
    'period'
]
# minSupport = 0.02
# minConfidence = 0.4

# Modified

concatUDF = F.udf(lambda cols: ','.join([x for x in cols if x is not None ]), StringType())

CallDF = (newDf.filter(newDf['Priority'] != 'Non-Emergency') \
             .select(*selectColumnList) \
             .withColumn('CallData', concatUDF(F.array(*selectColumnList)))
#             .withColumn('CallData', split(col('CallData'), ',\s*').cast('array<string>')) \
         )[['CallData']]
CallDF.show()         

# save data to text file on HDFS
#conDf.coalesce(1).write.format('text').option('header','true').mode('append').save('hdfs://worker2.hdp-internal:8020/user/ketkid1/calldata.txt')

CallDF.coalesce(1).write.format('text').option('header','true').mode('append').save('hdfs://worker2.hdp-internal:8020/user/ketkid1/calldata.txt')

