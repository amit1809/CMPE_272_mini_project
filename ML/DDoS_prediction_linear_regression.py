#!/usr/bin/env python
# coding: utf-8

# In[1]:


import findspark
findspark.init('/home/aarav/Amit/SJSU/CMPE_272/mini_project/spark_setup/spark-2.4.5-bin-hadoop2.6')


# In[2]:


from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()


# In[4]:


data = spark.read.csv('../dataset/kaggle_small_dataset/applicationlayer-ddos-dataset/train_mosaic.csv', header=True, inferSchema=True)


# In[ ]:





# In[6]:


data.limit(10).toPandas()


# In[7]:


data.count()


# In[8]:


data.groupBy("Label").count().show()


# In[10]:


from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol="Label", outputCol="LabelIndex")
indexed = indexer.fit(data).transform(data)
new_data = indexed.drop("Label")
new_data.limit(10).toPandas()


# In[11]:


#feature_columns = new_data.columns['Destination_Port','Flow_Duration','Total_Fwd_Packets','Total_Backward_Packets','Total_Length_of_Fwd_Packets','Total_Length_of_Bwd_Packets'] # here we omit the final 2 columns
#feature_columns = ['Destination_Port','Flow_Duration','Total_Fwd_Packets','Total_Backward_Packets','Total_Length_of_Fwd_Packets','Total_Length_of_Bwd_Packets']
feature_columns = data.columns[:-2]
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=feature_columns,outputCol="features")


# In[12]:


data_2 = assembler.transform(new_data)


# In[13]:


data_2.select("features").show(truncate=False)


# In[14]:


data_2.limit(10).toPandas()


# In[15]:


train, test = data_2.randomSplit([0.7, 0.3])


# In[16]:


from pyspark.ml.regression import LinearRegression


# In[ ]:





# In[17]:


algo = LinearRegression(featuresCol="features", labelCol="LabelIndex")


# In[18]:


model = algo.fit(train)


# In[19]:


evaluation_summary = model.evaluate(test)


# In[21]:


evaluation_summary.meanAbsoluteError


# In[22]:


evaluation_summary.rootMeanSquaredError


# In[23]:


evaluation_summary.r2


# In[24]:


predictions = model.transform(test)


# In[27]:


predictions.select(predictions.columns[75:]).limit(20).toPandas()


# In[ ]:




