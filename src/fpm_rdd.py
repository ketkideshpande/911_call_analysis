# This program trains and fits a FPGrowth model using RDD for finding frequent 
# patterns from the data

from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.mllib.fpm import FPGrowth

sc = SparkContext.getOrCreate(SparkConf())
#data = sc.textFile("hdfs://worker2.hdp-internal:8020/user/ketkid1/calls.txt")
data = sc.textFile("hdfs://worker2.hdp-internal:8020/user/ketkid1/calldata.txt")

# remove the empty lines present in RDD
data = data.filter(lambda line:line not in '')

# split each line on comma
calls = data.map(lambda line: line.strip().split(','))

# remove duplicates if any and cache the input data
unique = calls.map(lambda x:list(set(x))).cache()

# train the FP Growth model and predict the result
model = FPGrowth.train(calls, minSupport=0.02, numPartitions=2)
result = model.freqItemsets().collect()

# print the result
for i in result:
    print(i)
