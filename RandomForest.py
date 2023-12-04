from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Create a Spark session
spark = SparkSession.builder.appName("Random Forest").master("local[*]").getOrCreate()

# Load the data
df = spark.read.options(header='True', inferSchema='True', delimiter=',').csv("C:/Users/siben/Downloads/Occupancy.csv")

# Feature engineering: Create a vector column of features
feature_cols = [col for col in df.columns if col not in ['date', 'Occupancy']]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

df = assembler.transform(df)

# Split the data
(train_data, test_data) = df.randomSplit([0.66, 0.33], seed=42)

# Build the RandomForestClassifier
rf_classifier = RandomForestClassifier(labelCol="Occupancy", numTrees=100, featureSubsetStrategy="auto")

# Fit the model
model = rf_classifier.fit(train_data)

# Make predictions
predictions = model.transform(test_data)

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(labelCol="Occupancy", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f'Accuracy: {accuracy}')
while(True):
    c=0

