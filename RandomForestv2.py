import findspark
findspark.init()

from pyspark import SparkFiles
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import joblib
import numpy as np
from sklearn.datasets import load_iris
spark = SparkSession.builder.appName("RandomForestExample").getOrCreate()


df = spark.read.options(header='True', inferSchema='True', delimiter=',').csv("C:/Users/siben/Downloads/Occupancy.csv")
df.show(5)
# Preprocessing: StringIndexer for categorical labels
stringIndexer  = StringIndexer(inputCol="Occupancy", outputCol="label")

# Define the feature and label columns & Assemble the feature vector
assembler = VectorAssembler(inputCols=["Temperature","Humidity","Light","CO2","HumidityRatio"], outputCol="features")

rf = RandomForestClassifier(labelCol="label", featuresCol="features")

# Split the data into training and test sets
train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)
#We will now create a pipeline that includes the feature assembler and the Random Forest classifier.
pipeline = Pipeline(stages=[stringIndexer, assembler, rf])

# Define the hyperparameter grid
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 20, 30]) \
    .addGrid(rf.maxDepth, [5, 10, 15]) \
    .build()

# Create the cross-validator
cross_validator = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy"),
                          numFolds=5, seed=42)

# Train the model with the best hyperparameters
cv_model = cross_validator.fit(train_data)


best_rf_model = cv_model.bestModel.stages[-1]
importances = best_rf_model.featureImportances
feature_list = ["Temperature","Humidity","Light","CO2","HumidityRatio"]

print("Feature Importances:")
for feature, importance in zip(feature_list, importances):
    print(f"{feature}: {importance:.4f}")
# Make predictions on the test data
predictions = cv_model.transform(test_data)

evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")

# Evaluate the model
accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = {:.2f}".format(accuracy))
# Save the model with hdfs hadoop

best_rf_model.save("hdfs://localhost:9000/user/hadoopuser/rf_model")
# Load the model
from pyspark.ml.classification import RandomForestClassificationModel
loaded_model = RandomForestClassificationModel.load("rf_model")
#loaded_model.transform(test_data).show()
loaded_model.transform(test_data).show()