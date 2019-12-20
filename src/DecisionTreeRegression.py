############ Decision Tree Regression model building using Dataframe API ###############################

from pyspark.sql import SparkSession
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# create the sparkSession
spark = SparkSession.builder \
    .master("local") \
    .appName("MLib App") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

# Load the data stored in LIBSVM format as a DataFrame.
data = spark.read.format("libsvm").load("hdfs://worker2.hdp-internal:8020/user/sdeshpa1/data_final_test.txt")

# Split the data into training and test sets (30% held out for testing)
(train_df, test_df) = data.randomSplit([0.7, 0.3])

dt = DecisionTreeRegressor(featuresCol ='features', labelCol = 'label')

dt_model = dt.fit(train_df)

# gives the feature importance
dt_model.featureImportances

dt_predictions = dt_model.transform(test_df)

# Select (prediction, true label) and compute test error
dt_evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="r2")
rmse = dt_evaluator.evaluate(dt_predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

dt_evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="mae")
mae = dt_evaluator.evaluate(dt_predictions)
print("Mean Absolute Error (MAE) on test data = %g" % mae)

# cast the prediction column to nearest integer

import pyspark.sql.functions as func

predictions = dt_predictions.withColumn("predicted_label", func.round(dt_predictions["prediction"]).cast('integer'))

# Select example rows to display.
predictions.show(100)

