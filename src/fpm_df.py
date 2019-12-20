# This program trains FPGrowth model using dataframe as input for finding 
# frequent patterns and association rules from data 

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split
from pyspark.ml.fpm import FPGrowth

spark = SparkSession \
.builder \
.appName("Frequent Pattern Mining Algorithm") \
.enableHiveSupport() \
.getOrCreate()

# read the input file into a dataframe
#df = spark.read.format("text").load("hdfs://worker2.hdp-internal:8020/user/ketkid1/calls.txt")
df = spark.read.format("text").load("hdfs://worker2.hdp-internal:8020/user/ketkid1/calldata.txt")
df.show()

# convert single string type column of a dataframe to array type column
df1 = df.withColumn('value', split(col('value'), ',\s*').cast('array<string>'))

# fit the model on dataframe input
fpm = FPGrowth(itemsCol='value', minSupport=0.02, minConfidence=0.3,numPartitions=1)
model = fpm.fit(df1)

# Display frequent itemsets
model.freqItemsets.show(300,False)

# Display association rules generated by the model
model.associationRules.show(50,False)

# Examine input against all generated association rules to generate prediction by summerizing consequents
#model.transform(df1).show(30,False)
