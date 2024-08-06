import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, hour, unix_timestamp
from pyspark.sql.types import DoubleType, IntegerType

input_file = sys.argv[1]
output_file_path = sys.argv[2]

# Create a SparkSession
spark = SparkSession.builder \
    .appName("Taxi Data Processing") \
    .getOrCreate()

# Read the data
df = spark.read.csv(input_file, header=True, inferSchema=True)
df.printSchema()

# Manually parse datetime columns
df = df.withColumn("tpep_pickup_datetime", unix_timestamp(col("tpep_pickup_datetime"), "yyyy/M/d H:m").cast("timestamp"))
df = df.withColumn("tpep_dropoff_datetime", unix_timestamp(col("tpep_dropoff_datetime"), "yyyy/M/d H:m").cast("timestamp"))

# Define a UDF to calculate trip duration in minutes
def calculate_trip_duration(start, end):
    if start is None or end is None:
        return None
    return (end - start).total_seconds() / 60

trip_duration_udf = udf(calculate_trip_duration, DoubleType())

# Extract trip duration and pickup hour
df = df.withColumn("trip_duration", trip_duration_udf(col("tpep_pickup_datetime"), col("tpep_dropoff_datetime")))
df = df.withColumn("pickup_hour", hour(col("tpep_pickup_datetime")))

# Show the first 5 rows of the dataframe to verify the new columns
df.select("tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_duration", "pickup_hour").show(5)

# Select the necessary columns
columns_to_keep = [
    'passenger_count', 'trip_distance', 'pickup_longitude', 'pickup_latitude',
    'dropoff_longitude', 'dropoff_latitude', 'RateCodeID', 'payment_type',
    'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount',
    'improvement_surcharge', 'total_amount', 'trip_duration', 'pickup_hour'
]
df = df.select(columns_to_keep)

# Drop rows with missing values
df = df.dropna()

# Drop duplicate rows
df = df.dropDuplicates()

# Define a UDF to create the target variable 'tip_given'
tip_given_udf = udf(lambda tip: 1 if tip > 0 else 0, IntegerType())
df = df.withColumn("tip_given", tip_given_udf(col("tip_amount")))

# Save the processed data to a new CSV file, coalesce to one partition
df.coalesce(1).write.csv(output_file_path, header=True, mode='overwrite')
print(f"Data written to {output_file_path}")

# Show the first 5 rows of the dataframe
df.show(5)

spark.stop()





