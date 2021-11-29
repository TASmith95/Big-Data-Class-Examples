# Databricks notebook source
# MAGIC %md # Decision Tree to classify the priority of an incident

# COMMAND ----------

# MAGIC %md The Incident Management dataset has about 141712 records of 24918 incidents. Each state of the incident is being captured as an individual record with few exceptions where the closed state of an incident is recorded more than once. With the help of the below segment of the code, we load and clean the Incident Management data so that only one record representing the truly closed state per incident is obtained.

# COMMAND ----------

# MAGIC %md ------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md ##### Create a spark session and load the Incident Management Data set

# COMMAND ----------

from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.appName('IMMLDDT').getOrCreate()

# COMMAND ----------

file_location = "/FileStore/tables/incident_event_log_reduced.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)\

# COMMAND ----------

# MAGIC %md ------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md ##### Data pre-processing

# COMMAND ----------

# The data set has multiple states(New, Active, Awaiting user info, Resolved, Closed etc. ) of an incident. With the help 
# of the below command, we are just filtering one record per incident, that has the truly closed state of the incident. 

df_unique_incidents=df.filter("incident_state=='Closed'").sort("sys_mod_count",ascending=False).dropDuplicates(["number"])

# COMMAND ----------

# Selecting the dependent and the independent variables that are identified as most useful attributes to make predictions

data=df_unique_incidents.select(['caller_id','opened_by','location','category','subcategory',
                                 'u_symptom','assignment_group','priority'])

# COMMAND ----------

data=data.dropna()

# COMMAND ----------

# Create a 70-30 train test split

train_data,test_data=data.randomSplit([0.7,0.3])

# COMMAND ----------

# MAGIC %md ------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md ### Building the Decision Tree Classifier

# COMMAND ----------

# Import the required libraries

from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler,StringIndexer
from pyspark.ml import Pipeline

# COMMAND ----------

# Use StringIndexer to convert the categorical columns to hold numerical data

caller_id_indexer = StringIndexer(inputCol='caller_id',outputCol='caller_id_index',handleInvalid='keep')
opened_by_indexer = StringIndexer(inputCol='opened_by',outputCol='opened_by_index',handleInvalid='keep')
location_indexer = StringIndexer(inputCol='location',outputCol='location_index',handleInvalid='keep')
category_indexer = StringIndexer(inputCol='category',outputCol='category_index',handleInvalid='keep')
subcategory_indexer = StringIndexer(inputCol='subcategory',outputCol='subcategory_index',handleInvalid='keep')
u_symptom_indexer = StringIndexer(inputCol='u_symptom',outputCol='u_symptom_index',handleInvalid='keep')
assignment_group_indexer = StringIndexer(inputCol='assignment_group',outputCol='assignment_group_index',handleInvalid='keep')
priority_indexer = StringIndexer(inputCol='priority',outputCol='priority_index',handleInvalid='keep')

# COMMAND ----------

# Vector assembler is used to create a vector of input features

assembler = VectorAssembler(inputCols=['caller_id_index','opened_by_index','location_index','category_index',
                                       'subcategory_index','u_symptom_index','assignment_group_index'],
                            outputCol="features")

# COMMAND ----------

# Create an object for the Logistic Regression model
# Use the parameter maxBins and assign a value that is equal to or more than the number of categories in any sigle feature

dt_model = DecisionTreeClassifier(labelCol='priority_index',maxBins=5000)

# COMMAND ----------

# Pipeline is used to pass the data through indexer and assembler simultaneously. Also, it helps to pre-rocess the test data
# in the same way as that of the train data

pipe = Pipeline(stages=[caller_id_indexer,opened_by_indexer,location_indexer,category_indexer,subcategory_indexer,
                        u_symptom_indexer,assignment_group_indexer,priority_indexer,assembler,dt_model])

# COMMAND ----------

# It took 8 minutes for this step to execute

fit_model=pipe.fit(train_data)

# COMMAND ----------

# Store the results in a dataframe

results = fit_model.transform(test_data)

# COMMAND ----------

results.select(['priority_index','prediction']).show()

# COMMAND ----------

# MAGIC %md -------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md ##### Evaluating the model

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# COMMAND ----------

ACC_evaluator = MulticlassClassificationEvaluator(
    labelCol="priority_index", predictionCol="prediction", metricName="accuracy")

# COMMAND ----------

accuracy = ACC_evaluator.evaluate(results)

# COMMAND ----------

print("The accuracy of the decision tree classifier is {}".format(accuracy))

# COMMAND ----------

