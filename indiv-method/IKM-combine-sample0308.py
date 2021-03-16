#!/usr/bin/env python
# coding: utf-8

# #### df가 원본 데이터
# #### data가 유전자ID 열만 모아놓은 것
#     batch와 Condition 열은 제외함.

# In[3]:


import pandas as pd
import csv
import numpy

df = pd.read_table('../dataFile/210308/Colon_merged_2.txt')

#print("Contion열 포함 Original 데이터 프레임 크기:", df.shape,"\n")    

#유전자 이름 list로
only_geneID=df.columns.tolist()[:-2] #제일 마지막 칼럼이 batch, 마지막에서 2번째 칼럼이 condition

# Condition-샘플 이름
condition_list=df['Condition'].tolist()
my_set = set(condition_list) #집합set으로 변환
sampleName = list(my_set) #list로 변환

#print(sampleName)
#print("샘플 종류",len(sampleName),"개\n")

# batch
#batch_list=df['batch'].tolist()
#batch_set = set(batch_list) #집합set으로 변환
#batchName = list(batch_set) #list로 변환

#print(batchName)
#print("batch 개수",len(batchName),"개\n")


data=df.loc[:,only_geneID]

#print("Condition열 뺀 유전자 발현도만 모아있는 데이터 프레임 크기:", data.shape)
#data.head()
#df.head()


# In[4]:


#df.head()


# In[5]:


#data.columns


# <b> 샘플 종류 파악하기 </b>

# In[6]:


#print(len(sampleName)) #질병 개수 106개
column=sampleName[:len(sampleName)+1]
#sampleName


# <b> 각 샘플별 개수 파악하기</b>

# In[53]:


# 각 샘플별 개수 
#for i in sampleName[:]:
#    print(i,"\t+",df.loc[df["Condition"] == i,:].shape[0])


# In[ ]:





# In[ ]:





# In[65]:


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
#predict.head()


# In[ ]:





# <b> SSA/P -> SSA.P 변경 <br> 
# * /로 하면 경로로 인식하므로 오류가 생기기 때문     

# In[68]:


predict.loc[predict["predict"] == 'SSA/P'] = 'SSA.P'
    
#predict.loc[predict["predict"] == 'SSA.P']. head()


# In[ ]:





# ## correlation 계산 
# combine_sampleName 이용해서, 행 이름 구하기 <br>
# 합치기 전 dataframe인 data변수 이용해서, correlation 구하기 <br>
# pearson correlation 먼저 구하기 <br>
# 그래도 안 나오면, spearman correlation 구하기 <br>

# In[85]:


# concatenate labels to df as a new column
r = pd.concat([data, predict],axis=1)

#r.tail()


# In[87]:


sampleName[2]='SSA.P'
#sampleName


# In[ ]:





# In[90]:


# pearson correlation 구하는 함수 
def corr_pearson(df, sample_Name):
    data_sampleName = df[df.columns.difference(['predict'])].T

    df_corr_pearson = data_sampleName.corr(method='pearson')
    csvName = "corr_pearson_sampleName_"+sample_Name +"0308.csv"
    df_corr_pearson.to_csv(csvName) # csv 파일로 correlation 값 저장 
    
    return df_corr_pearson


# In[91]:


# spearman correlation 구하는 함수 
def corr_spearman(df, sample_Name):
    data_sampleName = df[df.columns.difference(['predict'])].T

    df_corr_spearman = data_sampleName.corr(method='spearman')
    csvName = "corr_spearman_sampleName_"+sample_Name +"0308.csv"
    df_corr_spearman.to_csv(csvName) # csv 파일로 correlation 값 저장 
    return df_corr_spearman


# In[92]:


# pearson heatmap 구하는 함수
def heatmap_pearson(df, sample_Name):

    import matplotlib.pyplot as plt 
    import seaborn as sns   

    plt.figure(figsize=(15,15))
    ax = sns.heatmap(data = df, linewidths=.5, annot=False, cmap="Spectral") # , cmap='Reds'
    plt.title('Heatmap of pearson pandas correlation for sampleName '+sample_Name+'0308', fontsize=20)

    ax.set_xticklabels(labels = df.columns, rotation=45)

    imagename="heatmap_pearson_sampleName_"+sample_Name+"0308.png"
    plt.savefig(imagename)


# In[93]:


# spearman heatmap 구하는 함수
def heatmap_spearman(df, sample_Name):

    import matplotlib.pyplot as plt 
    import seaborn as sns   

    plt.figure(figsize=(15,15))
    ax = sns.heatmap(data = df, linewidths=.5, annot=False, cmap="Spectral") # , cmap='Reds'
    plt.title('Heatmap of spearman pandas correlation for sampleName '+sample_Name+'0308', fontsize=20)

    ax.set_xticklabels(labels = df.columns, rotation=45)

    imagename="heatmap_spearman_sampleName_"+sample_Name+"0308.png"
    plt.savefig(imagename)


# In[ ]:





# In[94]:


# pearson 상관계수, pearson heatmap 구하기 
for i in sampleName[:]:
    print(i)
    r_sampleName = r.loc[r["predict"] == i,:]    
    df_corr_pearson = corr_pearson(r_sampleName, i)
    heatmap_pearson(df_corr_pearson, i)
    print('======'*5)


# In[95]:


# spearman 상관계수, spearman heatmap 구하기 
for i in sampleName[:]:
    print(i)
    r_sampleName = r.loc[r["predict"] == i,:]    
    df_corr_spearman  = corr_spearman(r_sampleName, i)
    heatmap_spearman(df_corr_spearman, i)
    print('======'*5)


# In[ ]:





# <b> 샘플 -> 숫자 매핑 </b>

# In[18]:


num=1
for i in combine_sampleName:
    print(i)
    predict.loc[predict["predict"] == i,:] = num
    num =num+1
#print(num)
#predict.head(50)


# In[ ]:





# <b> Kmeans 수행 </b>

# In[19]:


# create model and prediction
#model = KMeans(n_clusters=3,algorithm='auto')
#model.fit(data)

# concatenate labels to df as a new column
r = pd.concat([data, predict],axis=1)

#print(r)
#r.tail()


# In[20]:


#ks = range(1,20)
#inertias = []

#for k in ks:
#    model = KMeans(n_clusters=k)
#    model.fit(r)
#    inertias.append(model.inertia_)
    
# Plot ks vs inertias
#plt.figure(figsize = (10,10))
#plt.plot(ks, inertias, '-o')
#plt.xlabel('number of clusters, k')
#plt.ylabel('inertia')
#plt.xticks(ks)
#plt.show()


# In[21]:


#ks = range(1,30)
#inertias = []

#for k in ks:
#    model = KMeans(n_clusters=k)
#    model.fit(r)
#    inertias.append(model.inertia_)
    
# Plot ks vs inertias
#plt.figure(figsize = (10,10))
#plt.plot(ks, inertias, '-o')
#plt.xlabel('number of clusters, k')
#plt.ylabel('inertia')
#plt.xticks(ks)
#plt.show()


# In[ ]:





# ### PCA 수행
# https://ssungkang.tistory.com/entry/%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5-PCA-2-python-%EC%9D%84-%EC%82%AC%EC%9A%A9%ED%95%9C-PCA-%EC%8B%A4%EC%8A%B5 참고

# In[22]:


X=data.to_numpy()


# In[23]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

pca=PCA() #주성분 개수 지정하지 않고 클래스생성
pca.fit(X)  #주성분 분석
cumsum = np.cumsum(pca.explained_variance_ratio_) #분산의 설명량을 누적합
num_d = np.argmax(cumsum >= 0.95) + 1 # 분산의 설명량이 95%이상 되는 차원의 수
#num_d


# http://textmining.kr/?p=362

# In[24]:


pca = PCA(n_components = 0.95)
principalComponents = pca.fit_transform(data)
principalDf = pd.DataFrame(data = principalComponents)

#principalDf.head()


# In[25]:


y=pd.DataFrame(r['predict'].tolist())
y.columns=['predict']
y.head()


# In[26]:


finalDataFrame =pd.concat([principalDf, y], axis=1)
finalDataFrame.head()


# In[27]:


labels = []
yList = y.values.tolist()
for label in yList:
    if label[0] not in labels:
        labels.append(label[0])
    
labels


# In[28]:


sortedLabels=sorted(labels)
sortedLabels


# In[29]:


fig = plt.figure(figsize = (14,14))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)

'''
colors = ["black","grey","lightgray","lightcoral","maroon",
         "mistyrose","coral","peachpuff","darkorange","darkgoldenrod",
         "olive","yellowgreen","lawngreen","lightgreen","g",
         "mediumseagreen","mediumaquamarine","darkslategray","c","cadetblue",
         "dodgerblue","slategrey","darkblue","rebeccapurple","crimson",
          "fuchsia"]
'''
colors=["lightgray", "blue", "black", "darkorange", "green", "pink", 
        "coral", "crimson", "fuchsia", "black", "olive", "gold", "red"]

for label, color in zip(sortedLabels, colors):
    indicesToKeep = finalDataFrame['predict'] == label
    ax.scatter(
        finalDataFrame.loc[indicesToKeep, 0]
               , finalDataFrame.loc[indicesToKeep, 1]
              
               , c = color
               , s = 30
              )

ax.legend(combine_sampleName)
ax.grid()
plt.savefig("only-PCA-sampleCombine0308.png")


# In[30]:


ks = range(1,30)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(principalDf)
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.figure(figsize = (10,10))
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()


# In[111]:


#model = KMeans(n_clusters=9)
#model.fit(principalDf)
#centers = pd.DataFrame(model.cluster_centers_)
#print(centers.shape)
#centers.head()


# In[112]:


#centers = pd.DataFrame(model.cluster_centers_)
#center_x = centers[0]
#center_y = centers[1]

# scatter plot
#plt.figure(figsize = (10,10))
#plt.scatter(finalDataFrame[0],finalDataFrame[1], alpha=0.5)#c=finalDataFrame['predict']
#plt.scatter(center_x,center_y,s=10,marker='D',c='r')
#plt.show()


# In[113]:


#fig = plt.figure(figsize = (14,14))
#ax = fig.add_subplot(1,1,1) 
#ax.set_xlabel('Principal Component 1', fontsize = 15)
#ax.set_ylabel('Principal Component 2', fontsize = 15)
#ax.set_title('2 Component PCA', fontsize = 20)

'''
colors = ["black","grey","lightgray","lightcoral","maroon",
         "mistyrose","coral","peachpuff","darkorange","darkgoldenrod",
         "olive","yellowgreen","lawngreen","lightgreen","g",
         "mediumseagreen","mediumaquamarine","darkslategray","c","cadetblue",
         "dodgerblue","slategrey","darkblue","rebeccapurple","crimson",
          "fuchsia"]
'''

#colors=["lightgray", "blue", "black", "darkorange", "green", "pink", 
        "coral", "crimson", "fuchsia", "black", "olive", "gold", "red"]

#for label, color in zip(sortedLabels, colors):
#    indicesToKeep = finalDataFrame['predict'] == label
#    ax.scatter(
#        finalDataFrame.loc[indicesToKeep, 0]
#              , finalDataFrame.loc[indicesToKeep, 1]
#             
#               , c = color
#               , s = 20
#              )

    
#centers = pd.DataFrame(model.cluster_centers_)
#center_x = centers[0]
#center_y = centers[1]

# scatter plot
#plt.scatter(center_x,center_y,s=40,marker='D',c='r')

#labels=combine_sampleName+["centers"]
#ax.legend(labels)
#plt.savefig("PCA_sampleCombine+IKM.png")
#ax.grid()


# In[ ]:





# In[ ]:




