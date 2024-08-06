import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark session
spark = SparkSession.builder.appName("Taxi Tip Prediction").getOrCreate()

# Load the preprocessed data
input_file = sys.argv[1]
output_file_path = sys.argv[2]
df = spark.read.csv(input_file, header=True, inferSchema=True)

# Select the important features
feature_cols = ['fare_amount', 'total_amount', 'payment_type']

# Assemble features into a feature vector
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Index the label column
indexer = StringIndexer(inputCol="tip_given", outputCol="label")

# Define the logistic regression model
lr = LogisticRegression(featuresCol="features", labelCol="label")

# Create a pipeline
pipeline = Pipeline(stages=[assembler, indexer, lr])

# Define parameter grid for hyperparameter tuning
paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.01, 0.1, 1.0]).build()

# Define cross-validator
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=BinaryClassificationEvaluator(), numFolds=5)

# Split the data into training and test sets
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Train the model
cv_model = cv.fit(train_df)

# Evaluate the model
predictions = cv_model.transform(test_df)

# Initialize evaluators for different metrics
binary_evaluator = BinaryClassificationEvaluator(labelCol="label")
multiclass_evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")

# Calculate metrics
accuracy = multiclass_evaluator.evaluate(predictions)
precision = predictions.filter(predictions['label'] == 1).filter(predictions['prediction'] == 1).count() / predictions.filter(predictions['prediction'] == 1).count()
recall = predictions.filter(predictions['label'] == 1).filter(predictions['prediction'] == 1).count() / predictions.filter(predictions['label'] == 1).count()
f1_score = 2 * ((precision * recall) / (precision + recall))

# Print metrics
print(f"Model Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")


# Stop the Spark session
spark.stop()
