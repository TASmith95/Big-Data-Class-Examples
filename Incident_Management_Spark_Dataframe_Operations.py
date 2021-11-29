# Databricks notebook source
from pyspark.sql import SparkSession

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/incident_event_log_reduced.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "false"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

from pyspark.sql.functions import datediff,date_format,to_date,to_timestamp

# COMMAND ----------

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
                withColumn('duration',datediff(to_date(df.resolved_at, 'dd/MM/yyyy HH:mm'),to_date(df.opened_at, 'dd/MM/yyyy HH:mm')))

# COMMAND ----------

df_unique_incidents=df.filter("incident_state=='Closed'").sort("sys_mod_count",ascending=False).dropDuplicates(["number"])

# COMMAND ----------

# MAGIC %md #### 1. Top 5 people with most resolved incidents

# COMMAND ----------

A1=df_unique_incidents.groupby("resolved_by").count().sort("count",ascending=False)

# COMMAND ----------

A1.show(n=5)

# COMMAND ----------

# MAGIC %md #### 2. Based on least average duration, find the top 5 people with maxmium number of incidents resolved

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

A2= df_unique_incidents.groupby("resolved_by").agg(F.count("duration"),F.mean("duration")).\
    withColumnRenamed("count(duration)","Incidents Resolved").\
    withColumnRenamed("avg(duration)","Average Duration").\
    orderBy(["Average Duration","Incidents Resolved"],ascending=[True,False])

# COMMAND ----------

A2.show(n=5)

# COMMAND ----------

# MAGIC %md #### 3. People with maximum number of high impact incidents resolved

# COMMAND ----------

A3= df_unique_incidents.select(["resolved_by","impact","duration"]).\
    groupby(["impact","resolved_by"]).count().\
    sort(["impact","count"],ascending=[True,False])

# COMMAND ----------

A3.show(n=10)

# COMMAND ----------

# MAGIC %md #### 4a. In each impact levels, find the person with most number of incidents resolved

# COMMAND ----------

A4a=df_unique_incidents.select(["resolved_by","impact","duration"]).\
    groupby(["impact","resolved_by"]).count().\
    sort(["impact","count"],ascending=[True,False]).\
    dropDuplicates(["impact"])

# COMMAND ----------

A4a.show()

# COMMAND ----------

# MAGIC %md #### 4b. In each urgency levels, find the person with most number of incidents resolved

# COMMAND ----------

A4b=df_unique_incidents.select(["resolved_by","urgency","duration"]).\
    groupby(["urgency","resolved_by"]).count().\
    sort(["urgency","count"],ascending=[True,False]).dropDuplicates(["urgency"])

# COMMAND ----------

A4b.show()

# COMMAND ----------

# MAGIC %md #### 4c. In each priority levels, find the person with most number of incidents resolved

# COMMAND ----------

A4c=df_unique_incidents.select(["resolved_by","priority","duration"]).\
    groupby(["priority","resolved_by"]).count().\
    sort(["priority","count"],ascending=[True,False]).dropDuplicates(["priority"])

# COMMAND ----------

A4c.show()

# COMMAND ----------

# MAGIC %md #### 5. Find each contact type as a percentage of total incidents

# COMMAND ----------

from pyspark.sql.window import Window

# COMMAND ----------

A5= df_unique_incidents.select(["contact_type"]).\
    groupby(["contact_type"]).count().\
    withColumn("percentage",F.round(F.col("count")*100/F.sum("count").over(Window.partitionBy()),2))

# COMMAND ----------

A5.show()

# COMMAND ----------

# MAGIC %md #### 6. On each priority level, find the percentage of incidents which made SLA and which did not.

# COMMAND ----------

A6= df_unique_incidents.select(["priority","made_sla"]).\
    groupby(["priority","made_sla"]).count().\
    withColumnRenamed("count","Population").\
    withColumn("Made SLA %",F.round(F.col("Population")*100/F.sum("Population").over(Window.partitionBy("priority")),2)).\
    sort(["priority","made_sla"],ascending=[True,False])

# COMMAND ----------

A6.show()

# COMMAND ----------

# MAGIC %md #### 7. Top 5 location with the maximum number of incidents reported

# COMMAND ----------

A7= df_unique_incidents.groupby(["location"]).agg({"number":"count"}).\
    withColumnRenamed("count(number)","Incidents Reported").sort(["Incidents Reported"],ascending=False)

# COMMAND ----------

A7.show(5)

# COMMAND ----------

# MAGIC %md #### 8. Which category of issues missed meeting the SLA the most?

# COMMAND ----------

A8= df_unique_incidents.filter("made_sla==false").groupby(["category"]).\
    agg({"made_sla":"count"}).withColumnRenamed("count(made_sla)","Incidents failed to make SLA").\
    sort(["Incidents failed to make SLA"],ascending=False)

# COMMAND ----------

A8.show(5)

# COMMAND ----------

