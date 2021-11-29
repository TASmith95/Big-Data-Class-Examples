# Databricks notebook source
# MAGIC %md # Linear Regression to estimate duration

# COMMAND ----------

# MAGIC %md The Incident Management dataset has about 141712 records of 24918 incidents. Each state of the incident is being captured as an individual record with few exceptions where the closed state of an incident is recorded more than once. With the help of the below segment of the code, we load and clean the Incident Management data so that only one record representing the truly closed state per incident is obtained.

# COMMAND ----------

# MAGIC %md ------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md ##### Create a spark session and load the Incident Management Data set

# COMMAND ----------

from pyspark.sql import SparkSession

# COMMAND ----------

#spark = SparkSession.builder.appName('IMMLLR2').getOrCreate()

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

# Import the required libraries

from pyspark.sql.functions import datediff,date_format,to_date,to_timestamp

# COMMAND ----------

import pyspark.sql.functions as f

# COMMAND ----------

# Create new timestamp and date columns for all the attributes that had timestamp details stored as string
# Convert the boolean value of 'knowledge' to string
# Create the duration column (difference in number of days between the incident is opened and resolved)
spark.conf.set("spark.sql.legacy.timeParserPolicy","LEGACY")

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
                withColumn('duration',datediff(to_date(df.resolved_at, 'dd/MM/yyyy HH:mm'),to_date(df.opened_at, 'dd/MM/yyyy HH:mm')))


display(df)

# COMMAND ----------

# The data set has multiple states(New, Active, Awaiting user info, Resolved, Closed etc. ) of an incident. With the help 
# of the below command, we are just filtering one record per incident, that has the truly closed state of the incident. 

df_unique_incidents=df.filter("incident_state=='Closed'").sort("sys_mod_count",ascending=False).dropDuplicates(["number"])

# COMMAND ----------

# Selecting the dependent and the independent variables that are identified as most useful attributes to estimate duration

data=df_unique_incidents.select(['reassignment_count','reopen_count','sys_mod_count','opened_by',
                                 'location','category','subcategory','priority','assignment_group',
                                 'assigned_to','knowledge','resolved_by','duration'])

# COMMAND ----------

data.count()

# COMMAND ----------

data=data.dropna()

# COMMAND ----------

data.count()
display(data)

# COMMAND ----------

# Create a 70-30 train test split

train_data,test_data=data.randomSplit([0.7,0.3])

# COMMAND ----------

# MAGIC %md ------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md ### Building the Linear Regression model

# COMMAND ----------

# Import the required libraries

from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler,StringIndexer
from pyspark.ml import Pipeline

# COMMAND ----------

# Use StringIndexer to convert the categorical columns to hold numerical data

opened_by_indexer = StringIndexer(inputCol='opened_by',outputCol='opened_by_index',handleInvalid='keep')
location_indexer = StringIndexer(inputCol='location',outputCol='location_index',handleInvalid='keep')
category_indexer = StringIndexer(inputCol='category',outputCol='category_index',handleInvalid='keep')
subcategory_indexer = StringIndexer(inputCol='subcategory',outputCol='subcategory_index',handleInvalid='keep')
priority_indexer = StringIndexer(inputCol='priority',outputCol='priority_index',handleInvalid='keep')
assignment_group_indexer = StringIndexer(inputCol='assignment_group',outputCol='assignment_group_index',handleInvalid='keep')
assigned_to_indexer = StringIndexer(inputCol='assigned_to',outputCol='assigned_to_index',handleInvalid='keep')
knowledge_indexer = StringIndexer(inputCol='knowledge',outputCol='knowledge_index',handleInvalid='keep')
resolved_by_indexer = StringIndexer(inputCol='resolved_by',outputCol='resolved_by_index',handleInvalid='keep')

# COMMAND ----------

# Vector assembler is used to create a vector of input features

assembler = VectorAssembler(inputCols=["opened_by_index",'location_index','category_index',
                                       'subcategory_index','priority_index','assignment_group_index',
                                       'assigned_to_index','knowledge_index','resolved_by_index'],
                            outputCol="features")

# COMMAND ----------

# Pipeline is used to pass the data through indexer and assembler simultaneously. Also, it helps to pre-rocess the test data
# in the same way as that of the train data

pipe = Pipeline(stages=[opened_by_indexer,location_indexer,category_indexer,subcategory_indexer,
                        priority_indexer,assignment_group_indexer,assigned_to_indexer,
                        knowledge_indexer,resolved_by_indexer,assembler])

# COMMAND ----------

fitted_pipe=pipe.fit(train_data)

# COMMAND ----------

train_data=fitted_pipe.transform(train_data)
display(train_data)

# COMMAND ----------

# Create an object for the Linear Regression model

lr_model = LinearRegression(labelCol='duration')

# COMMAND ----------

# Fit the model on the train data

fit_model = lr_model.fit(train_data.select(['features','duration']))

# COMMAND ----------

# Transform the test data using the model to predict the duration

test_data=fitted_pipe.transform(test_data)
display(test_data)

# COMMAND ----------

# Store the results in a dataframe

results = fit_model.transform(test_data)
display(results)

# COMMAND ----------

results.select(['duration','prediction']).show()

# COMMAND ----------

# MAGIC %md -------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md ##### Evaluating the model

# COMMAND ----------

test_results = fit_model.evaluate(test_data)

# COMMAND ----------

test_results.residuals.show()

# COMMAND ----------

test_results.rootMeanSquaredError

# COMMAND ----------

# MAGIC %md The root mean squared error is very high indicating that the models prediction is really on the poorer side

# COMMAND ----------

test_results.r2

# COMMAND ----------

# MAGIC %md The r-squared value implies that the model explains only about 8% variance

# COMMAND ----------

