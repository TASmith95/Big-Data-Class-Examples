# Databricks notebook source
# MAGIC %md # K Means to cluster the incidents into 4 groups (Priority levels)

# COMMAND ----------

# MAGIC %md The Incident Management dataset has about 141712 records of 24918 incidents. Each state of the incident is being captured as an individual record with few exceptions where the closed state of an incident is recorded more than once. With the help of the below segment of the code, we load and clean the Incident Management data so that only one record representing the truly closed state per incident is obtained.

# COMMAND ----------

# MAGIC %md ------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md ##### Create a spark session and load the Incident Management Data set

# COMMAND ----------

from pyspark.sql import SparkSession

# COMMAND ----------

#spark = SparkSession.builder.appName('IMMLKM2').getOrCreate()

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

# MAGIC %md ------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md ##### Data pre-processing

# COMMAND ----------

# The data set has multiple states(New, Active, Awaiting user info, Resolved, Closed etc. ) of an incident. With the help 
# of the below command, we are just filtering one record per incident, that has the truly closed state of the incident. 

df_unique_incidents=df.filter("incident_state=='Closed'").sort("sys_mod_count",ascending=False).dropDuplicates(["number"])

# COMMAND ----------

# Selecting the dependent and the independent variables that are identified as most useful attributes to make predictions

data=df_unique_incidents.select(['opened_by','location','category','subcategory',
                                 'u_symptom','assignment_group','priority'])

# COMMAND ----------

data=data.dropna()

# COMMAND ----------

# MAGIC %md ------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

# MAGIC %md ### Building the K Means model

# COMMAND ----------

# Import the required libraries

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler,StringIndexer
from pyspark.ml import Pipeline

# COMMAND ----------

# Use StringIndexer to convert the categorical columns to hold numerical data

opened_by_indexer = StringIndexer(inputCol='opened_by',outputCol='opened_by_index',handleInvalid='keep')
location_indexer = StringIndexer(inputCol='location',outputCol='location_index',handleInvalid='keep')
category_indexer = StringIndexer(inputCol='category',outputCol='category_index',handleInvalid='keep')
subcategory_indexer = StringIndexer(inputCol='subcategory',outputCol='subcategory_index',handleInvalid='keep')
u_symptom_indexer = StringIndexer(inputCol='u_symptom',outputCol='u_symptom_index',handleInvalid='keep')
assignment_group_indexer = StringIndexer(inputCol='assignment_group',outputCol='assignment_group_index',handleInvalid='keep')
priority_indexer = StringIndexer(inputCol='priority',outputCol='priority_index',handleInvalid='keep')

# COMMAND ----------

# Vector assembler is used to create a vector of input features

assembler = VectorAssembler(inputCols=['opened_by_index','location_index','category_index',
                                       'subcategory_index','u_symptom_index','assignment_group_index'],
                            outputCol="features")

# COMMAND ----------

# Pipeline is used to pass the data through indexer and assembler simultaneously. Also, it helps to pre-rocess the test data
# in the same way as that of the train data.

pipe = Pipeline(stages=[opened_by_indexer,location_indexer,category_indexer,subcategory_indexer,
                        u_symptom_indexer,assignment_group_indexer,priority_indexer,assembler])

# COMMAND ----------

# It took 5 minutes for this step to complete execution

final_data=pipe.fit(data).transform(data)

# COMMAND ----------

# Create an object for the Logistic Regression model

kmeans_model = KMeans(k=4)

# COMMAND ----------

fit_model = kmeans_model.fit(final_data)

# COMMAND ----------

# wsse = fit_model.computeCost(final_data) for spark 2.7
wssse = fit_model.summary.trainingCost # for spark 3.0
print("The within set sum of squared error of the mode is {}".format(wssse))

# COMMAND ----------

centers = fit_model.clusterCenters()

# COMMAND ----------

#assembler = VectorAssembler(inputCols=['opened_by_index','location_index','category_index',
#                                      'subcategory_index','u_symptom_index','assignment_group_index'],
#                            outputCol="features")

# COMMAND ----------

print("Cluster Centers")
index=1
for cluster in centers:
    print("Centroid {}: {}".format(index,cluster))
    index+=1
#'opened_by_index','location_index','category_index',
# 'subcategory_index','u_symptom_index','assignment_group_index'

# COMMAND ----------

# Store the results in a dataframe

results = fit_model.transform(final_data)

# COMMAND ----------

results.select(['opened_by_index','location_index','category_index','subcategory_index',
                'u_symptom_index','assignment_group_index','prediction']).show()

# COMMAND ----------

results.groupby('prediction').count().sort('prediction').show()

# COMMAND ----------

# MAGIC %md -------------------------------------------------------------------------------------------------------------------------------

# COMMAND ----------

