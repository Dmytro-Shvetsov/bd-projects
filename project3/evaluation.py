from pyspark.ml.classification import LogisticRegressionModel, DecisionTreeClassificationModel, RandomForestClassificationModel, GBTClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import findspark
findspark.init()


from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from functools import reduce
from pyspark.sql import DataFrame
import matplotlib.pyplot as plt
import os

# Create a directory to store the trained models
os.makedirs("trained_models", exist_ok=True)

spark = SparkSession.builder.master("local")\
    .appName("FlightPrediction")\
    .config("spark.driver.memory", "4g").getOrCreate()

filepaths = ['data/2009.csv',
             'data/2010.csv', 'data/2011.csv',
             'data/2012.csv',
             'data/2013.csv', 'data/2014.csv',
             'data/2015.csv']
dfs = [spark.read.csv(file, header=True) for file in filepaths]
df = reduce(DataFrame.union, dfs)

# Drop unnamed columns
df = df.drop(*[c for c in df.columns if 'Unnamed' in c])

# Select columns of interest
selected_columns = ['FL_DATE', 'OP_CARRIER', 'ORIGIN', 'DEST', 'CRS_DEP_TIME', 'CRS_ARR_TIME', 'CRS_ELAPSED_TIME',
                    'DISTANCE', 'CANCELLED', "CANCELLATION_CODE"]
df = df.select(selected_columns)

# Replace null values in the 'CANCELLED' column with 0
df = df.fillna({'CANCELLED': 0, "CANCELLATION_CODE": 0})

# Remove rows with null values only in the selected columns

df = df.na.drop(subset=selected_columns)

# Cast 'CANCELLED' column as integer
df = df.withColumn("CANCELLED", df["CANCELLED"].cast("integer"))
#top10_airlines = df.groupBy("OP_CARRIER").count().orderBy("count", ascending=False).limit(10)
#top10_airlines.show()

#cancellation_reasons = df.groupBy("CANCELLATION_CODE").count()
#cancellation_reasons.show()
#cancellation_reasons_pd = cancellation_reasons.toPandas()

# Set up the pie chart
#plt.figure(figsize=(8, 8))
#plt.pie(cancellation_reasons_pd['count'], labels=cancellation_reasons_pd['CANCELLATION_CODE'], autopct='%1.1f%%')
#plt.title("Proportion of Total Flight Cancellation Reasons (2009-2015)")
#plt.show()

# Calculate the fractions of canceled and not-canceled flights to balance the dataset
#df = df.withColumn("CANCELLED", df["CANCELLED"].cast("integer"))

num_canceled = df.filter(df.CANCELLED == 1).count()
num_not_canceled = df.filter(df.CANCELLED == 0).count()
total_rows = df.count()

# Calculate target number of samples for each class
target_samples_per_class = total_rows // 2

# Calculate fractions, ensuring they are within the valid range [0, 1]
fraction_canceled = min(target_samples_per_class / num_canceled, 1)
fraction_not_canceled = min(target_samples_per_class / num_not_canceled, 1)

fractions = {0: fraction_not_canceled, 1: fraction_canceled}
# Perform stratified sampling using the calculated fractions
stratified_df = df.sampleBy("CANCELLED", fractions, seed=42)
stratified_df.show()

# Create categorical_columns
categorical_columns = ['OP_CARRIER', 'ORIGIN', 'DEST', "CANCELLATION_CODE"]

# Use StringIndexer to index categorical columns
indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="skip") for col in categorical_columns]

# Use OneHotEncoderEstimator to create one-hot encoded vectors
encoder = OneHotEncoderEstimator(inputCols=[f"{col}_index" for col in categorical_columns],
                                 outputCols=[f"{col}_vec" for col in categorical_columns])

# Fit and transform indexers and encoder
stages = indexers + [encoder]
for stage in stages:
    stratified_df = stage.fit(stratified_df).transform(stratified_df)

# Create a VectorAssembler to assemble feature vectors
assembler = VectorAssembler(inputCols=[f"{col}_vec" for col in categorical_columns], outputCol="features")
stratified_df = assembler.transform(stratified_df)

# Split the dataset into training and testing sets
train, test = stratified_df.randomSplit([0.7, 0.3])

# Assume the models were saved to these paths
logistic_regression_model_path = "/path/to/logistic_regression_model"
decision_tree_model_path = "/path/to/decision_tree_model"
random_forest_model_path = "/path/to/random_forest_model"
gbt_model_path = "/path/to/gbt_model"

# Load the models
lr_model = LogisticRegressionModel.load(logistic_regression_model_path)
dt_model = DecisionTreeClassificationModel.load(decision_tree_model_path)
rf_model = RandomForestClassificationModel.load(random_forest_model_path)
gbt_model = GBTClassificationModel.load(gbt_model_path)

# Assume that 'test_data' is the DataFrame containing the test data
# Initialize evaluator
evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")

# Compute accuracy on the test set for each model
lr_accuracy = evaluator.evaluate(lr_model.transform(test))
dt_accuracy = evaluator.evaluate(dt_model.transform(test))
rf_accuracy = evaluator.evaluate(rf_model.transform(test))
gbt_accuracy = evaluator.evaluate(gbt_model.transform(test))

print("Logistic Regression Test set accuracy = " + str(lr_accuracy))
print("Decision Tree Test set accuracy = " + str(dt_accuracy))
print("Random Forest Test set accuracy = " + str(rf_accuracy))
print("Gradient Boosted Trees Test set accuracy = " + str(gbt_accuracy))
