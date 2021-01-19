#!/usr/bin/env python
# coding: utf-8

# #### df가 원본 데이터
# #### data가 유전자ID 열만 모아놓은 것
#     batch와 Condition 열은 제외함.

# In[1]:


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


# In[2]:


df.head()


# In[3]:


data.columns


# In[4]:


print(len(sampleName)) #질병 개수 106개
column=sampleName[:len(sampleName)+1]
sampleName


# In[5]:


# 각 샘플별 개수 
for i in sampleName[1:]:
    print(i,"\t+",df.loc[df["Condition"] == i,:].shape[0])


# In[ ]:





# In[ ]:





# In[6]:


df.loc[df["Condition"].isna()] # df.loc[["SSP_4","SSP_8"],:] 같은 의미


# In[7]:


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


# In[8]:


# nan인 값들 0으로 변경
predict.loc[["SSP_4"],"predict"]=0
predict.loc[["SSP_8"],"predict"]=0

num=1
for i in sampleName[1:]:
    predict.loc[predict["predict"] == i,:] = num
    num =num+1


# In[9]:


predict.tail()


# In[10]:


# concatenate labels to df as a new column
r = pd.concat([data, predict],axis=1)

print(r)
r.tail()


# In[ ]:





# ### PCA 수행
# https://ssungkang.tistory.com/entry/%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5-PCA-2-python-%EC%9D%84-%EC%82%AC%EC%9A%A9%ED%95%9C-PCA-%EC%8B%A4%EC%8A%B5 참고

# In[11]:


X=data.to_numpy()


# In[12]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

pca=PCA() #주성분 개수 지정하지 않고 클래스생성
pca.fit(X)  #주성분 분석
cumsum = np.cumsum(pca.explained_variance_ratio_) #분산의 설명량을 누적합
num_d = np.argmax(cumsum >= 0.95) + 1 # 분산의 설명량이 95%이상 되는 차원의 수
num_d


# In[ ]:





# http://textmining.kr/?p=362

# In[13]:


pca = PCA(n_components = 0.95)
principalComponents = pca.fit_transform(data)
principalDf = pd.DataFrame(data = principalComponents)

principalDf.head()


# In[14]:


y=pd.DataFrame(r['predict'].tolist())
y.columns=['predict']
y.head()


# pca = PCA(n_components = 0.95)
# principalComponents = pca.fit_transform(data)
# principalR = pd.DataFrame(data = principalComponents)
# 
# y=pd.DataFrame(r['predict'].tolist())
# y.columns=['predict']
# 
# finalDataFrameR =pd.concat([principalR, y], axis=1)
# finalDataFrameR
# 
# labels = []
# yList = y.values.tolist()
# for label in yList:
#     if label[0] not in labels:
#         labels.append(label[0])
#     
# labels

# In[15]:


finalDataFrame =pd.concat([principalDf, y], axis=1)
finalDataFrame.head()


# In[16]:


labels = []
yList = y.values.tolist()
for label in yList:
    if label[0] not in labels:
        labels.append(label[0])
    
labels


# In[17]:


sortedLabels=sorted(labels)
sortedLabels


# ### PCA만 시각화

# In[18]:


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





# In[ ]:





# In[ ]:





# In[93]:


#pip3 install pyclustering

pip install pyclustering


# ## pyclustering
# ### cure

# https://pyclustering.github.io/  여기에서 패키지 파일 다운 받음

# In[19]:


import sys
sys.executable


# In[20]:


from jupyter_core.paths import jupyter_data_dir
print(jupyter_data_dir())


# cure는 1,2,3차원만 됨.
# - cure는 numpy로 만들고, birch는 list로 만들어서 수행함.
# 
# 고차원 하려면, pyclustering의 cluster_visualizer_multidim 해야 하는데, 별로 좋은 방법 같지는 않음...

# In[21]:


X=data.to_numpy() #cure는 1,2,3차원만 됨.
X


# In[23]:


import pyclustering
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.cure import cure
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import FCPS_SAMPLES

# Input data in following format [ [0.1, 0.5], [0.3, 0.1], ... ].
#input_data = read_sample(FCPS_SAMPLES.SAMPLE_LSUN);
# Allocate three clusters.

X=finalDataFrame.iloc[:,[0,1]].to_numpy()
cure_instance = cure(X, 5)
cure_instance.process()
clusters = cure_instance.get_clusters()

clusters


# In[24]:


# Visualize allocated clusters.
visualizer = cluster_visualizer()
visualizer.append_clusters(clusters, X)
visualizer.show()


# In[ ]:





# k=2~9

# In[25]:


X=finalDataFrame.iloc[:,[0,1]].to_numpy()
for i in range(2,10):
    cure_instance = cure(X, i)
    cure_instance.process()
    clusters = cure_instance.get_clusters()
    # Visualize allocated clusters.
    visualizer = cluster_visualizer()
    visualizer.append_clusters(clusters, X)
    print("\nk=",i)
    visualizer.show()


# k=10~19
# 2차원

# In[26]:


X=finalDataFrame.iloc[:,[0,1]].to_numpy()
for i in range(10,20):
    cure_instance = cure(X, i)
    cure_instance.process()
    clusters = cure_instance.get_clusters()
    # Visualize allocated clusters.
    visualizer = cluster_visualizer()
    visualizer.append_clusters(clusters, X)
    print("\nk=",i)
    visualizer.show()


# In[ ]:




