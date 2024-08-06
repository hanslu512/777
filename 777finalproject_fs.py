import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
import pandas as pd

# Arguments for input and output file paths
input_file = sys.argv[1]
output_file_path = sys.argv[2]

# Create a SparkSession
spark = SparkSession.builder \
    .appName("Taxi Tip Feature Analysis") \
    .getOrCreate()

# Read the preprocessed data
df = spark.read.csv(input_file, header=True, inferSchema=True)

# Select all columns except the target column 'tip_given'
feature_columns = df.columns[:-1]

# Prepare the features column
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
assembled_df = assembler.transform(df)

# Standardize the features
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
scaler_model = scaler.fit(assembled_df)
scaled_df = scaler_model.transform(assembled_df)

# Train Lasso Regression model with non-zero regParam
lasso = LinearRegression(featuresCol="scaled_features", labelCol="tip_given", elasticNetParam=1.0, regParam=0.1)
lasso_model = lasso.fit(scaled_df)

# Get the coefficients and selected features
coefficients = lasso_model.coefficients
important_features = [(feature, coef) for coef, feature in zip(coefficients, feature_columns) if coef != 0]

# Save important features to a CSV file
important_features_df = pd.DataFrame(important_features, columns=["Feature", "Coefficient"])
output_csv_path = f"{output_file_path.rstrip('/')}/important_features.csv"
important_features_df.to_csv(output_csv_path, index=False)

# Stop SparkSession
spark.stop()