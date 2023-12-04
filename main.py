# This is a sample Python script.
from sklearn.metrics import mean_squared_error, accuracy_score

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from pyspark.sql import SparkSession

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pyspark.ml.classification import RandomForestClassifier

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   # print('PySpark Application SSaterted ...')
    #spark = SparkSession.builder.appName("RAndom Forest").master("local[*]").getOrCreate()
    #print(spark.sparkContext.appName)
   #
    #nameList=["feojkjknf","éihb","ffffff"]
    #print(spark.sparkContext.parallelize(nameList))
    #print("PySpark Application Completed")
    spark = SparkSession.builder.appName("Random Forest").master("local[*]").getOrCreate()
    df=spark.read.options(header='True',inferSchema='False',delimiter=',').csv("C:/Users/siben/Downloads/Occupancy.csv")
    df.printSchema()
    y = df['Occupancy']
    X = df.drop('date', 'Occupancy')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    rf_classifier=RandomForestClassifier(labelCol="Occupancy")
    rf_classifier.fit(X_train,y_train)
    # Make predictionsrf
    y_pred = rf_classifier.predict(X_test)

# Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Accuracy: {accuracy}')

# Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
