from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

# Step 1: Initialize Spark Session
spark = SparkSession.builder \
    .appName("DataCleaningExample") \
    .getOrCreate()

# Step 2: Load data
df = spark.read.csv("sample_data.csv", header=True, inferSchema=True)

print("=== Original Data ===")
df.show()

# Step 3: Remove duplicates
df = df.dropDuplicates()

# Step 4: Handle missing values
# Fill missing names with "Unknown"
df = df.fillna({"name": "Unknown"})
# Fill missing age with average age
avg_age = df.selectExpr("avg(age)").collect()[0][0]
df = df.fillna({"age": avg_age})
# Drop rows where 'city' or 'income' is null
df = df.dropna(subset=["city", "income"])

# Step 5: Correct data types (income to float)
df = df.withColumn("income", col("income").cast("double"))

# Step 6: Clean inconsistent values
# Example: Standardize city names (New york → New York)
df = df.withColumn("city", when(col("city") == "New york", "New York").otherwise(col("city")))

print("=== Cleaned Data ===")
df.show()
                                         
# Stop Spark
spark.stop()
