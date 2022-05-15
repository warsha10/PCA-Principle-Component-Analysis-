#!/usr/bin/env python
# coding: utf-8

# # Principle component analysis (PCA)

# ### 1. Import libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ### 2. Import data

# In[2]:


wine_data=pd.read_csv('wine.csv')
wine_data


# ### 3. Data understanding

# ### 3.1 Initial Analysis

# In[3]:


#checking for null values
wine_data.isna().sum()


# In[4]:


#checking datatypes
wine_data.dtypes


# ### 4. Model Building
# 
# PCA Visualization
# 
# Since its difficult to visualize high dimensional data, we use PCA to find the first two principal components, and visualize the data in this new, two-dimensional space, with a single scatter-plot. 

# In[5]:


X=wine_data.drop(labels='Type', axis=1)


# In[6]:


#scaling our data so that each feature has a single unit variance.

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
scaled_X = std_scaler.fit_transform(X)
scaled_X


# #### Correlation matrix - PCA will function better if the features are highly correlated.

# In[7]:


wine_data.corr().round(2)


# In[8]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
transformed_x = pca.fit_transform(scaled_X)
transformed_x
pca_data = pd.DataFrame(transformed_x)
pca_data.columns = ['PC1','PC2']
pca_data


# In[9]:


variance = pca.explained_variance_ratio_ #How much information each Principal component takes
variance


# In[10]:


import numpy as np
np.cumsum(np.round(a = variance,decimals=4)*100)


# In[11]:


pca_data.shape


# In[12]:


y = wine_data[['Type']]


# In[13]:



from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(pca_data,y)
#rf_classifier.fit(X,y)
y_pred = rf_classifier.predict(pca_data)
#y_pred = rf_classifier.predict(X)
y_pred


# In[14]:


from sklearn.metrics import accuracy_score,precision_score,confusion_matrix


# In[15]:


accuracy_score(y,y_pred)


# In[16]:


precision_score(y,y_pred,pos_label='positive', average='micro' )


# In[17]:


confusion_matrix(y,y_pred)


# ### PCA for Vizualization

# In[18]:


from matplotlib import pyplot as plt
plt.figure(figsize=(15,8))
plt.scatter(pca_data['PC1'],pca_data['PC2'],c=wine_data['Type'],)
plt.show()


# ### 1. Hierarchial clustering

# In[19]:


# Import Libraries
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize


# In[21]:


# create Dendrograms
plt.figure(figsize=(10,8))
dendrogram=sch.dendrogram(sch.linkage(pca_data,'complete'))


# In[22]:


# Create Clusters (y)
hclusters=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
hclusters


# In[26]:


y=pd.DataFrame(hclusters.fit_predict(wine_data),columns=['clustersid'])
y['clustersid'].value_counts()


# In[27]:


# Adding clusters to dataset
wine3=wine_data.copy()
wine3['clustersid']=hclusters.labels_
wine3


# ### 2. K-Means Clustering

# In[28]:


# Import Libraries
from sklearn.cluster import KMeans


# In[29]:


# within-cluster sum-of-squares criterion 
wcss=[]
for i in range (1,6):
    kmeans=KMeans(n_clusters=i,random_state=2)
    kmeans.fit(wine_data)
    wcss.append(kmeans.inertia_)


# In[30]:


# Plot K values range vs WCSS to get Elbow graph for choosing K (no. of clusters)
plt.plot(range(1,6),wcss)
plt.title('Elbow Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# ### Build Cluster algorithm using K=2
# 

# In[31]:


# Cluster algorithm using K=3
clusters2=KMeans(2,random_state=30).fit(wine_data)
clusters2


# In[32]:


clusters2.labels_


# In[34]:


# Assign clusters to the data set
wine4=wine_data.copy()
wine4['clusters2id']=clusters2.labels_
wine4


# In[36]:


wine4['clusters2id'].value_counts()


# In[ ]:




