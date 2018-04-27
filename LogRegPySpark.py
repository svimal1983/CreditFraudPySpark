# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 18:10:30 2018

"""
#import findspark
#findspark.init()
#findspark.find()

import pyspark
from pyspark import SparkContext
sc =SparkContext()
from pyspark.sql import SparkSession

# SparkSession
spark = SparkSession.builder \
   .master("local") \
   .appName("Classification") \
   .config("spark.executor.memory", "1gb") \
   .getOrCreate()
sc = spark.sparkContext
df = spark.read.format("csv").option("header", "true").load("hdfs://Master:9000/home/biagroup27/creditcard.csv")
df.show(1)

# Check the data type of all the colbumns
df.printSchema()


# Function to convert into floadt type
from pyspark.sql.types import *
def convertColumn(df, names, newType):
  for name in names: 
     df = df.withColumn(name, df[name].cast(newType))
  return df 

columns = ['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount','Class']
df = convertColumn(df, columns, FloatType())

#Recheck to ensure if the type is float
df.printSchema()


# To create a dense vector with 'dependent variable' as 'label' and independent variables as a dense vector 'features'
from pyspark.ml.linalg import DenseVector

# Define the `input_data` 
input_data = df.rdd.map(lambda x: (x[0], DenseVector(x[1:])))

# Replace df with the new DataFrame; Label and features
df = spark.createDataFrame(input_data, ["label", "features"])



from pyspark.ml.feature import StandardScaler

# Initialize the `standardScaler`
standardScaler = StandardScaler(inputCol="features", outputCol="features_scaled")

# Fit the DataFrame to the scaler
scaler = standardScaler.fit(df)

# Transform the data in `df` with the scaler
scaled_df = scaler.transform(df)

# Inspect the result
scaled_df.take(2)

from pyspark.sql.functions import rand 
df = df.orderBy(rand())
train_data, test_data = df.randomSplit([0.8, 0.2],seed=1234)


# Fitting the LogisticRegression: Change the below code for all the types of algorithms that we need for project
from pyspark.ml.classification import LogisticRegression
mlr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)

# Fit the model
mlrModel = mlr.fit(train_data)

#Predict the values for test_data
predicted = mlrModel.transform(test_data)
predicted.head(5)


#To evaluate AUC

from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
evaluator.evaluate(predicted)


# To find performance metrics
predicted = mlrModel.transform(test_data)
predictions = predicted.select("prediction").rdd.map(lambda x: x[0])
labels = predicted.select("label").rdd.map(lambda x: x[0])


#To zip actual vs predicted values as a list
b = predictions.collect()
a = labels.collect()
predictionAndLabel = zip(a,b)
print(list(predictionAndLabel))

#Compute percision and recall
tp=0
for i in range(0,len(a)):
    if a[i] == 0.0 and b[i] == 0.0:
        tp += 1
print("Correct Legit Case:",tp)

tn=0
for i in range(0,len(a)):
    if a[i] == 1.0 and b[i] == 1.0:
        print(i)
        tn += 1
print("Correct Fraud Case:",tn)

fp=0
for i in range(0,len(a)):
    if a[i] == 0 and b[i] == 1:
        fp += 1
print("False Fraud Case:",fp)

fn=0
for i in range(0,len(a)):
    if a[i] == 1 and b[i] == 0:
        fn += 1
print("False Legit Case:",fn)

r = float(tp)/(tp + fn)
print ("Recall", r)

p = float(tp) / (tp + fp)
print ("Precision", p)

