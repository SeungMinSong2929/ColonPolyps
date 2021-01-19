#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:





# In[ ]:





# In[ ]:





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


# In[11]:


import numpy as np
from sklearn.cluster import SpectralCoclustering
X=data.to_numpy()
clustering = SpectralCoclustering(n_clusters=5, random_state=0).fit(X)
clustering.row_labels_ #doctest: +SKIP

clustering.column_labels_ #doctest: +SKIP

clustering


# In[12]:


from sklearn.metrics import consensus_score
from matplotlib import pyplot as plt
# shuffle clusters
rng = np.random.RandomState(0)
row_idx = rng.permutation(X.shape[0])
col_idx = rng.permutation(X.shape[1])

# 여기서부터 우리 실험 데이터!
model = SpectralCoclustering(n_clusters=7, random_state=0)
model.fit(X)


# In[19]:


fig = plt.figure(figsize = (40,55))
ax = fig.add_subplot(1,1,1)

fit_data = X[np.argsort(model.row_labels_)]
fit_data = fit_data[:, np.argsort(model.column_labels_)]

#plt.matshow(fit_data, cmap=plt.cm.Blues)
ax.matshow(fit_data, cmap=plt.cm.Blues)

#plt.title("After biclustering")
ax.set_title("After biclustering")

#plt.show()
ax.grid()


# ### PCA 수행
# https://ssungkang.tistory.com/entry/%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5-PCA-2-python-%EC%9D%84-%EC%82%AC%EC%9A%A9%ED%95%9C-PCA-%EC%8B%A4%EC%8A%B5 참고

# In[20]:


X=data.to_numpy()


# In[21]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

pca=PCA() #주성분 개수 지정하지 않고 클래스생성
pca.fit(X)  #주성분 분석
cumsum = np.cumsum(pca.explained_variance_ratio_) #분산의 설명량을 누적합
num_d = np.argmax(cumsum >= 0.95) + 1 # 분산의 설명량이 95%이상 되는 차원의 수
num_d


# In[22]:


pca = PCA(n_components = 0.95)
principalComponents = pca.fit_transform(data)
principalDf = pd.DataFrame(data = principalComponents)

principalDf.head()


# y=pd.DataFrame(df['Condition'].tolist())
# y.columns=["Condition"]
# y

# In[23]:


y=pd.DataFrame(r['predict'].tolist())
y.columns=['predict']
y.head()


# In[24]:


finalDataFrame =pd.concat([principalDf, y], axis=1)
finalDataFrame.head()


# http://textmining.kr/?p=362

# In[25]:


labels = []
yList = y.values.tolist()
for label in yList:
    if label[0] not in labels:
        labels.append(label[0])
    
labels


# In[26]:


sortedLabels=sorted(labels)
sortedLabels


# In[27]:


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
               , s = 50
              )

# ax.legend(labels)
ax.legend(sampleName)
ax.grid()
plt.savefig("only-PCA.png")


# In[28]:


# PCA한 데이터프레임으로 클러스터링 하기
X=principalDf.to_numpy()
model = SpectralCoclustering(n_clusters=5, random_state=0)
model.fit(X)


fit_data = X[np.argsort(model.row_labels_)]
fit_data = fit_data[:, np.argsort(model.column_labels_)]

plt.matshow(fit_data, cmap=plt.cm.Blues)
plt.title("After biclustering")

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




