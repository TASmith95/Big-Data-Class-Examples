# Databricks notebook source
# MAGIC %md
# MAGIC # Logistic Regression to predict whether an incident met SLA

# COMMAND ----------

# MAGIC %md
# MAGIC The Incident Management dataset has about 141712 records of 24918 incidents. Each state of the incident is being captured as an individual record with few exceptions where the closed state of an incident is recorded more than once. With the help of the below segment of the code, we load and clean the Incident Management data so that only one record representing the truly closed state per incident is obtained.

# COMMAND ----------

# MAGIC %md
# MAGIC ------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Create a spark session and load the Incident Management Data set

# COMMAND ----------

from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.appName('IMMLLogReg').getOrCreate()

# COMMAND ----------

spark.conf.set("spark.sql.legacy.timeParserPolicy","LEGACY")

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

#df = spark.read.csv('incident_event_log.csv',inferSchema=True,header=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Data pre-processing

# COMMAND ----------

# Import the required libraries

from pyspark.sql.functions import datediff,date_format,to_date,to_timestamp

# COMMAND ----------

import pyspark.sql.functions as f

# COMMAND ----------

# Create new timestamp and date columns for all the attributes that had timestamp details stored as string
# The target column made_sla is converted to hold numeric values
# Two durations (resolved and closed) are calculated to be passed as the independent variables

df=df.withColumn('resolved_ts',to_timestamp(df.resolved_at, 'dd/MM/yyyy HH:mm')).\
        withColumn('opened_ts',to_timestamp(df.opened_at, 'dd/MM/yyyy HH:mm')).\
        withColumn('sys_created_ts',to_timestamp(df.sys_created_at, 'dd/MM/yyyy HH:mm')).\
        withColumn('sys_updated_ts',to_timestamp(df.sys_updated_at, 'dd/MM/yyyy HH:mm')).\
        withColumn('closed_ts',to_timestamp(df.closed_at, 'dd/MM/yyyy HH:mm')).\
        withColumn('resolved',to_date(df.resolved_at, 'dd/MM/yyyy HH:mm')).\
        withColumn('opened',to_date(df.opened_at, 'dd/MM/yyyy HH:mm')).\
        withColumn('sys_created',to_date(df.sys_created_at, 'dd/MM/yyyy HH:mm')).\
        withColumn('sys_updated',to_date(df.sys_updated_at, 'dd/MM/yyyy HH:mm')).\
        withColumn('closed',to_date(df.closed_at, 'dd/MM/yyyy HH:mm')).\
        withColumn('knowledge', f.col('knowledge').cast('string')).\
        replace(['TRUE',], 'True', subset='knowledge').\
        replace(['FALSE'], 'False', subset='knowledge').\
        withColumn('resolved_duration',datediff(to_date(df.resolved_at, 'dd/MM/yyyy HH:mm'),\
                                                to_date(df.opened_at, 'dd/MM/yyyy HH:mm'))).\
        withColumn('closed_duration',datediff(to_date(df.closed_at, 'dd/MM/yyyy HH:mm'),\
                                                to_date(df.opened_at, 'dd/MM/yyyy HH:mm'))).\
        withColumn('made_sla_int',df.made_sla.cast('integer'))

# COMMAND ----------

# The data set has multiple states(New, Active, Awaiting user info, Resolved, Closed etc. ) of an incident. With the help 
# of the below command, we are just filtering one record per incident, that has the truly closed state of the incident. 

df_unique_incidents=df.filter("incident_state=='Closed'").sort("sys_mod_count",ascending=False).dropDuplicates(["number"])

# COMMAND ----------

# Selecting the dependent and the independent variables that are identified as most useful attributes to make predictions

data=df_unique_incidents.select(['sys_mod_count','opened_by','location','category','priority','assignment_group',
                                 'knowledge','resolved_duration','closed_duration','made_sla_int'])

# COMMAND ----------

data=data.dropna()

# COMMAND ----------

# Create a 70-30 train test split

train_data,test_data=data.randomSplit([0.7,0.3])

# COMMAND ----------

# MAGIC %md
# MAGIC ------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ### Building the Logistic Regression model

# COMMAND ----------

# Import the required libraries

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler,StringIndexer ,OneHotEncoder
from pyspark.ml import Pipeline

# COMMAND ----------

# Use StringIndexer to convert the categorical columns to hold numerical data

opened_by_indexer = StringIndexer(inputCol='opened_by',outputCol='opened_by_index',handleInvalid='keep')
location_indexer = StringIndexer(inputCol='location',outputCol='location_index',handleInvalid='keep')
category_indexer = StringIndexer(inputCol='category',outputCol='category_index',handleInvalid='keep')
priority_indexer = StringIndexer(inputCol='priority',outputCol='priority_index',handleInvalid='keep')
assignment_group_indexer = StringIndexer(inputCol='assignment_group',outputCol='assignment_group_index',handleInvalid='keep')
knowledge_indexer = StringIndexer(inputCol='knowledge',outputCol='knowledge_index',handleInvalid='keep')

# COMMAND ----------

# OneHotEncoderEstimator converts the indexed data into a vector which will be effectively handled by Logistic Regression model

data_encoder = OneHotEncoder(inputCols=['opened_by_index','location_index','category_index',
                                                 'priority_index','assignment_group_index','knowledge_index'], outputCols= ['opened_by_vec','location_vec','category_vec','priority_vec',
                                                  'assignment_group_vec','knowledge_vec'],
                                      handleInvalid='keep')

# COMMAND ----------

# Vector assembler is used to create a vector of input features

assembler = VectorAssembler(inputCols=["opened_by_vec",'location_vec','category_vec',
                                      'priority_vec','assignment_group_vec','knowledge_vec'],
                            outputCol="features")



# COMMAND ----------

# Create an object for the Logistic Regression model

lr_model = LogisticRegression(labelCol='made_sla_int')

# COMMAND ----------

# Pipeline is used to pass the data through indexer and assembler simultaneously. Also, it helps to pre-rocess the test data
# in the same way as that of the train data. It also 

pipe = Pipeline(stages=[opened_by_indexer,location_indexer,category_indexer,priority_indexer,
                        assignment_group_indexer,knowledge_indexer,data_encoder,assembler,lr_model])
  


# COMMAND ----------

fit_model=pipe.fit(train_data)

# COMMAND ----------

# Store the results in a dataframe

results = fit_model.transform(test_data)

# COMMAND ----------

results.select(['made_sla_int','prediction']).show()

# COMMAND ----------

# MAGIC %md
# MAGIC -------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluating the model

# COMMAND ----------

# MAGIC %md
# MAGIC #####  1. Area under the ROC

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

AUC_evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='made_sla_int',metricName='areaUnderROC')

# COMMAND ----------

AUC = AUC_evaluator.evaluate(results)

# COMMAND ----------

print("The area under the curve is {}".format(AUC))

# COMMAND ----------

# MAGIC %md
# MAGIC A roughly 73% area under ROC denotes the model has performed reasonably well in predicting whether an incident has met the sla

# COMMAND ----------

# MAGIC %md
# MAGIC ------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC #####  2. Area under the PR

# COMMAND ----------

PR_evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='made_sla_int',metricName='areaUnderPR')

# COMMAND ----------

PR = PR_evaluator.evaluate(results)

# COMMAND ----------

print("The area under the PR curve is {}".format(PR))

# COMMAND ----------

# MAGIC %md
# MAGIC ------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC #####  3. Accuracy

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# COMMAND ----------

ACC_evaluator = MulticlassClassificationEvaluator(
    labelCol="made_sla_int", predictionCol="prediction", metricName="accuracy")

# COMMAND ----------

accuracy = ACC_evaluator.evaluate(results)

# COMMAND ----------

print("The accuracy of the model is {}".format(accuracy))

# COMMAND ----------

# MAGIC %md
# MAGIC ------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC #####  4. Confusion Matrix

# COMMAND ----------

from sklearn.metrics import confusion_matrix

# COMMAND ----------

y_true = results.select("made_sla_int")
y_true = y_true.toPandas()

y_pred = results.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred)
print("Below is the confusion matrix \n {}".format(cnf_matrix))

# COMMAND ----------

