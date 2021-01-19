#!/usr/bin/env python
# coding: utf-8

# #### df가 원본 데이터
# #### data가 유전자ID 열만 모아놓은 것
#     batch와 Condition 열은 제외함.

# In[2]:


import pandas as pd
import csv
import numpy

df = pd.read_table('../dataFile/201126/Colon_merged_273samples.txt')

print("Contion열 포함 Original 데이터 프레임 크기:", df.shape,"\n")    

#유전자 이름 list로
only_geneID=df.columns.tolist()[:-2] #제일 마지막 칼럼이 batch, 마지막에서 2번째 칼럼이 condition

# Condition-샘플 이름
condition_list=df['Condition'].tolist()
my_set = set(condition_list) #집합set으로 변환
sampleName = list(my_set) #list로 변환

print(sampleName)
print("샘플 종류",len(sampleName),"개\n")

# batch
batch_list=df['batch'].tolist()
batch_set = set(batch_list) #집합set으로 변환
batchName = list(batch_set) #list로 변환

print(batchName)
print("batch 개수",len(batchName),"개\n")


data=df.loc[:,only_geneID]

print("Condition열 뺀 유전자 발현도만 모아있는 데이터 프레임 크기:", data.shape)
data.head()
#df.head()


# In[3]:


df.head()


# In[4]:


data.columns


# In[5]:


print(len(sampleName)) #질병 개수 106개
column=sampleName[:len(sampleName)+1]
sampleName


# In[6]:


# 각 샘플별 개수 
for i in sampleName[1:]:
    print(i,"\t+",df.loc[df["Condition"] == i,:].shape[0])


# In[ ]:





# In[ ]:





# In[7]:


df.loc[df["Condition"].isna()] # df.loc[["SSP_4","SSP_8"],:] 같은 의미


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from pandas import DataFrame

# https://note.espriter.net/1326

#predict=pd.DataFrame(df['Condition'].tolist())
predict=df[['Condition']]
predict.columns=['predict']
predict.head()


# In[9]:


# nan인 값들 0으로 변경
predict.loc[["SSP_4"],"predict"]=0
predict.loc[["SSP_8"],"predict"]=0


num=1
for i in sampleName[1:]:
    predict.loc[predict["predict"] == i,:] = num
    num =num+1


# In[10]:


predict.tail()


# In[11]:


# concatenate labels to df as a new column
r = pd.concat([data, predict],axis=1)

print(r)
r.tail()


# In[ ]:





# In[ ]:





# ### PCA 수행
# https://ssungkang.tistory.com/entry/%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5-PCA-2-python-%EC%9D%84-%EC%82%AC%EC%9A%A9%ED%95%9C-PCA-%EC%8B%A4%EC%8A%B5 참고

# In[12]:


X=data.to_numpy()


# In[13]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

pca=PCA() #주성분 개수 지정하지 않고 클래스생성
pca.fit(X)  #주성분 분석
cumsum = np.cumsum(pca.explained_variance_ratio_) #분산의 설명량을 누적합
num_d = np.argmax(cumsum >= 0.95) + 1 # 분산의 설명량이 95%이상 되는 차원의 수
num_d


# http://textmining.kr/?p=362

# In[14]:


pca = PCA(n_components = 0.95)
principalComponents = pca.fit_transform(data)
principalDf = pd.DataFrame(data = principalComponents)

principalDf.head()


# In[15]:


y=pd.DataFrame(r['predict'].tolist())
y.columns=['predict']
y.head()


# In[16]:


finalDataFrame =pd.concat([principalDf, y], axis=1)
finalDataFrame.head()


# In[17]:


labels = []
yList = y.values.tolist()
for label in yList:
    if label[0] not in labels:
        labels.append(label[0])
    
labels


# In[18]:


sortedLabels=sorted(labels)
sortedLabels


# In[19]:


fig = plt.figure(figsize = (14,14))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)

colors = ["black","grey","lightgray","lightcoral","maroon",
         "mistyrose","coral","peachpuff","darkorange","darkgoldenrod",
         "olive","yellowgreen","lawngreen","lightgreen","g",
         "mediumseagreen","mediumaquamarine","darkslategray","c","cadetblue",
         "dodgerblue","slategrey","darkblue","rebeccapurple","crimson",
          "fuchsia"]

for label, color in zip(sortedLabels, colors):
    indicesToKeep = finalDataFrame['predict'] == label
    ax.scatter(
        finalDataFrame.loc[indicesToKeep, 0]
               , finalDataFrame.loc[indicesToKeep, 1]
              
               , c = color
               , s = 30
              )

ax.legend(sampleName)
ax.grid()
plt.savefig("only-PCA.png")


# In[ ]:





# In[ ]:





# In[93]:


#pip3 install pyclustering

pip install pyclustering


# ## pyclustering
# ### birch

# https://pyclustering.github.io/  여기에서 패키지 파일 다운 받음

# In[24]:


import sys
sys.executable


# In[25]:


from jupyter_core.paths import jupyter_data_dir
print(jupyter_data_dir())


# In[ ]:





# pc1, pc2만

# In[26]:


from pyclustering.cluster.birch import birch
from pyclustering.cluster import cluster_visualizer
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import FAMOUS_SAMPLES

# list 형태로!
X=finalDataFrame.iloc[:,[0,1]].values.tolist()

for k in range(2,10):   
    birch_instance = birch(X, k, diameter=3.0)
    # Cluster analysis
    birch_instance.process()
    # Obtain results of clustering
    clusters = birch_instance.get_clusters()
    # Visualize allocated clusters
    visualizer = cluster_visualizer()
    visualizer.append_clusters(clusters, X)
    print("\nk=",k)
    visualizer.show()


# In[27]:


X=finalDataFrame.iloc[:,[0,1]].values.tolist()

for k in range(10,20):   
    birch_instance = birch(X, k, diameter=3.0)
    # Cluster analysis
    birch_instance.process()
    # Obtain results of clustering
    clusters = birch_instance.get_clusters()
    # Visualize allocated clusters
    visualizer = cluster_visualizer()
    visualizer.append_clusters(clusters, X)
    print("\nk=",k)
    visualizer.show()


# In[ ]:





# In[ ]:




