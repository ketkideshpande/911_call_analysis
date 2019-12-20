# Random Forest regression model

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator

# create the sparkSession
spark = SparkSession.builder \
    .master("local") \
    .appName("MLib App") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

# Load and parse the data file, converting it to a DataFrame.
data = spark.read.format("libsvm").load("hdfs://worker2.hdp-internal:8020/user/sdeshpa1/data_final_test.txt")

# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Split the data into training and test sets (30% held out for testing)
(train_df, test_df) = data.randomSplit([0.7, 0.3])

# Train a RandomForest model.
rf = RandomForestRegressor(featuresCol="indexedFeatures")

# Chain indexer and forest in a Pipeline
pipeline = Pipeline(stages=[featureIndexer, rf])

# Train model.  This also runs the indexer.
model = pipeline.fit(train_df)

# Make predictions.
predictions = model.transform(test_df)

# Select example rows to display.
predictions.select("prediction", "label", "features").show(5)

# cast the prediction column to nearest integer

import pyspark.sql.functions as func

predictions = predictions.withColumn("predicted_label", func.round(predictions["prediction"]).cast('integer'))

# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

dt_evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="mae")
mae = dt_evaluator.evaluate(dt_predictions)
print("Mean Absolute Error (MAE) on test data = %g" % mae) 

rfModel = model.stages[1]
print(rfModel)  # summary only

# Select example rows to display.
predictions.show(100)

