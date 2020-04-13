#!/usr/bin/env python
# coding: utf-8

# In[1]:


import findspark
findspark.init('/home/aarav/Amit/SJSU/CMPE_272/mini_project/spark_setup/spark-2.4.5-bin-hadoop2.6')


# In[2]:


from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()


# In[3]:


data = spark.read.csv('../dataset/train_mosaic.csv', header=True, inferSchema=True)


# In[ ]:





# In[4]:


data.show()


# In[ ]:





# In[6]:


from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol="Label", outputCol="LabelIndex")
indexed = indexer.fit(data).transform(data)
new_data = indexed.drop("Label")
new_data.show()


# In[7]:


#feature_columns = new_data.columns['Destination_Port','Flow_Duration','Total_Fwd_Packets','Total_Backward_Packets','Total_Length_of_Fwd_Packets','Total_Length_of_Bwd_Packets'] # here we omit the final 2 columns
#feature_columns = ['Destination_Port','Flow_Duration','Total_Fwd_Packets','Total_Backward_Packets','Total_Length_of_Fwd_Packets','Total_Length_of_Bwd_Packets']
feature_columns = data.columns[:-2]
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=feature_columns,outputCol="features")


# In[8]:


data_2 = assembler.transform(new_data)


# In[155]:


#data_2.select("features").show(truncate=False)


# In[9]:


data_2.show()


# In[10]:


train, test = data_2.randomSplit([0.7, 0.3])


# In[11]:


from pyspark.ml.regression import LinearRegression


# In[ ]:





# In[12]:


algo = LinearRegression(featuresCol="features", labelCol="LabelIndex")


# In[13]:


model = algo.fit(train)


# In[14]:


evaluation_summary = model.evaluate(test)


# In[15]:


evaluation_summary.meanAbsoluteError


# In[16]:


evaluation_summary.rootMeanSquaredError


# In[17]:


evaluation_summary.r2


# In[18]:


predictions = model.transform(test)


# In[96]:


predictions.select(predictions.columns[75:]).limit(10).toPandas()


# In[ ]:




