#!/usr/bin/env python
# coding: utf-8

# In[1]:



import ibmos2spark
# @hidden_cell
credentials = {
    'endpoint': 'https://s3-api.us-geo.objectstorage.service.networklayer.com',
    'service_id': 'iam-ServiceId-f93426ea-ec32-4125-b2cb-11f5014b87d2',
    'iam_service_endpoint': 'https://iam.ng.bluemix.net/oidc/token',
    'api_key': 'WViBQAjXiCQrkPV0f_VAy50t81-9_0X6wKg4BSgJYfVz'
}

configuration_name = 'os_94f68d61bbd04e64ae0af5a23ca4b3b2_configs'
cos = ibmos2spark.CloudObjectStorage(sc, credentials, configuration_name, 'bluemix_cos')

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
df_data_1 = spark.read  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')  .option('header', 'true')  .option('inferSchema', 'true')  .load(cos.url('train_mosaic.csv', 'ddosprediction-donotdelete-pr-qixygkc7opzg7d'))
df_data_1.take(5)


# In[2]:


df_data_1.limit(10).toPandas()


# In[4]:


df_data_1.count()


# In[5]:


df_data_1.groupBy("Label").count().show()


# In[6]:


from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol="Label", outputCol="LabelIndex")
indexed = indexer.fit(df_data_1).transform(df_data_1)
new_data = indexed.drop("Label")
new_data.limit(10).toPandas()


# In[8]:


#feature_columns = new_data.columns['Destination_Port','Flow_Duration','Total_Fwd_Packets','Total_Backward_Packets','Total_Length_of_Fwd_Packets','Total_Length_of_Bwd_Packets'] # here we omit the final 2 columns
#feature_columns = ['Destination_Port','Flow_Duration','Total_Fwd_Packets','Total_Backward_Packets','Total_Length_of_Fwd_Packets','Total_Length_of_Bwd_Packets']
feature_columns = df_data_1.columns[:-2]
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=feature_columns,outputCol="features")


# In[9]:


data_2 = assembler.transform(new_data)


# In[10]:


data_2.select("features").show(truncate=False)


# In[11]:


data_2.limit(10).toPandas()


# In[12]:


train, test = data_2.randomSplit([0.7, 0.3])


# In[13]:


from pyspark.ml.regression import LinearRegression


# In[14]:


algo = LinearRegression(featuresCol="features", labelCol="LabelIndex")


# In[15]:


model = algo.fit(train)


# In[16]:


evaluation_summary = model.evaluate(test)


# In[17]:


evaluation_summary.meanAbsoluteError


# In[18]:


evaluation_summary.rootMeanSquaredError


# In[19]:


evaluation_summary.r2


# In[20]:


predictions = model.transform(test)


# In[21]:


predictions.select(predictions.columns[75:]).limit(20).toPandas()


# In[ ]:




