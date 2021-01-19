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


# <b> 샘플 종류 파악하기 </b>

# In[4]:


print(len(sampleName)) #질병 개수 106개
column=sampleName[:len(sampleName)+1]
sampleName


# <b> 각 샘플별 개수 파악하기</b>

# In[5]:


# 각 샘플별 개수 
for i in sampleName[1:]:
    print(i,"\t+",df.loc[df["Condition"] == i,:].shape[0])


# In[ ]:





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


# In[ ]:





# <b> 이름은 다른데, 같은 샘플들 합치기 </b>

# healthy colonic tissue = control = NAN = NS <br>
# Tubulovillous adenoma = Villous/Tubluovillous adenoma = tubulovillous polyp = tubulovillous adenoma <br>
# sessile serrated adenoma = SSA/P = SSP <br> 
# adenomatous polyp = AP <br>
# Cancer = Ca = colorectal adenocarcinoma <br>

# In[8]:


combine_sampleName=sampleName[:]
combine_sampleName


# In[9]:


NS=['healthy colonic tissue', 'Control', 'NS']
print(NS[0])

for i in NS:
    predict.loc[predict["predict"] == i,:] = NS[0]

# NAN값인 샘플들 NS[0]으로 변경
predict.loc[["SSP_4"],"predict"]=NS[0]
predict.loc[["SSP_8"],"predict"]=NS[0]

# predict-샘플 이름
predict_list=predict['predict'].tolist()
predict_set = set(predict_list) #집합set으로 변환
combine_sampleName = list(predict_set) #list로 변환
combine_sampleName


# In[10]:


TubA=['Tubulovillous adenoma', 'Villous / tubulovillous adenoma', 'tubulovillous polyp', 'tubulovillous adenoma', 'villous adenoma']
print(TubA[0])

for i in TubA:
    predict.loc[predict["predict"] == i,:] = TubA[0]
    
# predict-샘플 이름
predict_list=predict['predict'].tolist()
predict_set = set(predict_list) #집합set으로 변환
combine_sampleName = list(predict_set) #list로 변환
combine_sampleName


# In[11]:


SSP=['sessile serrated adenoma', 'SSA/P', 'SSP']
print(SSP[0])

for i in SSP:
    predict.loc[predict["predict"] == i,:] = SSP[0]
    
# predict-샘플 이름
predict_list=predict['predict'].tolist()
predict_set = set(predict_list) #집합set으로 변환
combine_sampleName = list(predict_set) #list로 변환
combine_sampleName


# In[12]:


AP=['adenomatous polyp', 'AP']
print(AP[0])

for i in AP:
    predict.loc[predict["predict"] == i,:] = AP[0]
    
# predict-샘플 이름
predict_list=predict['predict'].tolist()
predict_set = set(predict_list) #집합set으로 변환
combine_sampleName = list(predict_set) #list로 변환
combine_sampleName


# In[13]:


Cancer=['Cancer', 'Ca', 'Colorectal adenocarcinoma']
print(Cancer[0])

for i in Cancer:
    predict.loc[predict["predict"] == i,:] = Cancer[0]
    
# predict-샘플 이름
predict_list=predict['predict'].tolist()
predict_set = set(predict_list) #집합set으로 변환
combine_sampleName = list(predict_set) #list로 변환
combine_sampleName


# In[14]:


TA=['tubular adenoma', 'Tubular adenoma']
print(TA[0])

for i in TA:
    predict.loc[predict["predict"] == i,:] = TA[0]
    
# predict-샘플 이름
predict_list=predict['predict'].tolist()
predict_set = set(predict_list) #집합set으로 변환
combine_sampleName = list(predict_set) #list로 변환
combine_sampleName


# <b> 합치고 난 최종 샘플 종류와 개수 </b>

# In[15]:


for i in combine_sampleName:
    print(i,"\t+",predict.loc[predict["predict"] == i,:].shape[0])


# ## correlation 계산 
# combine_sampleName 이용해서, 행 이름 구하기 <br>
# 합치기 전 dataframe인 data변수 이용해서, correlation 구하기 <br>
# pearson correlation 먼저 구하기 <br>
# 그래도 안 나오면, spearman correlation 구하기 <br>

# NS[0] # healthy colonic tissue <br>
# TubA[0] # Tubulovillous adenoma <br>
# SSP[0] # sessile serrated adenoma <br>
# AP[0] # adenomatous polyp <br>
# Cancer[0] # Cancer <br>
# TA[0] # tubular adenoma <br>

# In[16]:


# concatenate labels to df as a new column
r = pd.concat([data, predict],axis=1)

r.tail()


# In[17]:


# healthy colonic tissue
r_NS = r.loc[r["predict"] == NS[0],:]
r_NS.head() 


# In[18]:


# pearson 상관계수 구하기 
data_NS = r_NS[r_NS.columns.difference(['predict'])].T
df_corr_NS = data_NS.corr(method='pearson')
df_corr_NS.to_csv("corr_healthy_colonic_tissue.csv") # csv 파일로 correlation 값 저장 
df_corr_NS


# In[19]:


df_corr_NS[df_corr_NS<0].count()


# In[20]:


# Tubulovillous adenoma
r_TubA = r.loc[r["predict"] == TubA[0],:] 
r_TubA.head()


# In[21]:


# pearson 상관계수 구하기 
data_TubA = r_TubA[r_TubA.columns.difference(['predict'])].T
df_corr_TubA = data_TubA.corr(method='pearson')
df_corr_TubA.to_csv("corr_Tubulovillous_adenoma.csv") # csv 파일로 correlation 값 저장 
df_corr_TubA


# In[22]:


print(df_corr_TubA[df_corr_TubA<0].count())
df_corr_TubA[df_corr_TubA<0].count().sum()


# In[23]:


# sessile serrated adenoma
r_SSP = r.loc[r["predict"] == SSP[0],:]
r_SSP.head()


# In[24]:


# pearson 상관계수 구하기 
data_SSP = r_SSP[r_SSP.columns.difference(['predict'])].T
df_corr_SSP = data_SSP.corr(method='pearson')
df_corr_SSP.to_csv("corr_sessile_serrated_adenoma.csv") # csv 파일로 correlation 값 저장 
df_corr_SSP


# In[25]:


df_corr_SSP[df_corr_SSP<0].count()


# In[26]:


# adenomatous polyp
r_AP = r.loc[r["predict"] == AP[0],:]
r_AP.head()


# In[27]:


# pearson 상관계수 구하기 
data_AP = r_AP[r_AP.columns.difference(['predict'])].T
df_corr_AP = data_AP.corr(method='pearson')
df_corr_AP.to_csv("corr_adenomatous_polyp.csv") # csv 파일로 correlation 값 저장 
df_corr_AP


# In[28]:


df_corr_AP[df_corr_AP<0].count()


# In[29]:


# Cancer
r_Cancer = r.loc[r["predict"] == Cancer[0],:]
r_Cancer.head()


# In[30]:


# pearson 상관계수 구하기 
data_Cancer = r_Cancer[r_Cancer.columns.difference(['predict'])].T
df_corr_Cancer = data_Cancer.corr(method='pearson')
df_corr_Cancer.to_csv("corr_Cancer.csv") # csv 파일로 correlation 값 저장 
df_corr_Cancer


# In[31]:


df_corr_Cancer[df_corr_Cancer<0].count()


# In[32]:


# tubular adenoma
r_TA = r.loc[r["predict"] == TA[0],:]
r_TA.head()


# In[33]:


# pearson 상관계수 구하기 
data_TA = r_TA[r_TA.columns.difference(['predict'])].T
df_corr_TA = data_TA.corr(method='pearson')
df_corr_TA.to_csv("corr_tubular_adenoma.csv") # csv 파일로 correlation 값 저장 
df_corr_TA


# In[34]:


df_corr_TA[df_corr_TA<0].count()


# In[ ]:





# ## spearman 상관계수 구하기 

# 1. dataframe 갖고오기
# 2. dataframe의 인덱스 저장하기
# 3. spearmanr 구하기
# 4. 인덱스1, 인덱스2, corr, pval 행으로 저장하기 
# 5. 함수 만들기 (시작 인덱스, 인덱스 총 길이 )
# 6. for문으로 함수를 0: 인덱스 끝까지 돌리기 
# 7. 최종적으로 dataframe에 합치기 

# In[35]:


# spearman correlation 구하는 함수
import numpy as np
from scipy import stats

# 미리 df의 df.values로 arr을 구해
# arr을 파라미터로 넣고, arr[:,start_idx], arr[:, la]
def spear_corr(df, start_idx):
    rho_list=[]
    pval_list=[]
    index1=[]
    index2=[] 
    
    # 1. 배열로 바꾸기
    # df로 받는다면, .T한다음에, 배열로 바꿔야 함. 
    arr=df.T.values
    # print("arr: \n", arr)
    
    # 2. index1, index2에 저장할 칼럼 이름 리스트 뽑기
    index_list=df.columns 
    #print(index_list)
    last_idx=len(index_list)
    # print("칼럼 길이:", last_idx, "\n")

    # 3. 시작-끝 spearman 구하기
    # 각 열마다 나오는 값을 list로 저장
    idx_range=range(start_idx, last_idx)
    #print(start_idx, index_list[start_idx])
    
    for i in idx_range:
        rho, pval=stats.spearmanr(arr[:, start_idx], arr[:, i])
        rho_list.append(rho)
        pval_list.append(pval)
        index1.append(index_list[start_idx])
        index2.append(index_list[i])
    
    #print(index2)
    
    return rho_list, pval_list, index1, index2
# 리턴해서, 모두 list로 합친다음, df로 바꿀 예정


# In[36]:


# correlation 데이터프레임 만들기
def spear_df(df):
    last = len(df.columns)
    spearman_list=[]
    pvalue_list=[]
    idx1_list=[]
    idx2_list=[]
    
    for i in range(0, last):
        rho_list, pval_list, index1, index2 = spear_corr(df, i)
        spearman_list.extend(rho_list)
        pvalue_list.extend(pval_list)
        idx1_list.extend(index1)
        idx2_list.extend(index2)
    
    df_spear_pval = pd.DataFrame(data={'spearman': spearman_list, 
                                       'pvalue': pvalue_list, 
                                       'index1': idx1_list, 
                                       'index2': idx2_list})
    return df_spear_pval


# In[ ]:





# In[37]:


# pvalue > 0.05 중에 
# index1, index2에 하나라도 포함되는 샘플 지워기
# 클러스터링 할 샘플만 list형태로 리턴하는 함수
def remove_noise(df_combined, df_spearman):
    index_list=[]
    
    df_noise = df_spearman[(df_spearman['spearman']<0) | (df_spearman['pvalue']>0.05)]
    #df_noise = df_spearman[(df_spearman['pvalue']>0.05)]
    print("df_spearman 크기: ", df_spearman.shape)
    print("df_noise 크기: ", df_noise.shape)
    
    noise_sample_list = list(df_noise['index1']) + list(df_noise['index2'])
    noise_sample_set=set(noise_sample_list)
    print("df_noise에 해당되는 샘플:", noise_sample_set)
    print("df_combined (original)인 전체 샘플:", set(df_combined.columns))
    
    index_list = list(set(df_combined.columns) - noise_sample_set)
    #print(noise_sample_set)
    
    return index_list


# In[ ]:





# In[38]:


# heatmap을 위해 데이터프레임 칼럼명 기준으로 정렬
def sort_df(df):
    # 칼럼명 정렬
    cols_list = list(df.columns)
    cols_list = sorted(cols_list)

    df_list=[]
    df_sorted=df[cols_list[0]]
    for i in range(1, len(cols_list)):
        sub_df=df[cols_list[i]]
        #df_list.extend(sub_df)
        df_sorted = pd.concat([df_sorted, df[cols_list[i]]], axis=1)

    # df_sorted = pd.concat([df_list], axis=1)
    #print(df_list)
    return df_sorted


# In[53]:


# spearman correlation heatmap으로 그리기 
# https://rfriend.tistory.com/419 [R, Python 분석과 프로그래밍의 친구 (by R Friend)] 
# 고치는 중. colorbar 색 고치면, 다시 위로 복사할 것. 
def heatmap_spearman(df, sample_name): # df_spearman_NS
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams['figure.figsize'] = [50, 50]

    df_spearman_pivot = df.pivot('index1', 'index2', 'spearman')
    df_spearman_pivot

    # heatmap by plt.pcolor()
    plt.pcolor(df_spearman_pivot)

    plt.xticks(np.arange(0, len(df_spearman_pivot.columns), 1), df_spearman_pivot.columns)
    plt.yticks(np.arange(0, len(df_spearman_pivot.index), 1), df_spearman_pivot.index)

    plt.title('Heatmap of spearman correlation for '+sample_name, fontsize=20) #healthy_colonic_tissue
    plt.xlabel('index1', fontsize=14)
    plt.ylabel('index2', fontsize=14)
    
    plt.jet()
    
    plt.colorbar()
    plt.grid()
    plt.xticks(rotation=60)
    
    imagename="heatmap_spearman_"+sample_name+".png"
    plt.savefig(imagename) #healthy_colonic_tissue
    
    plt.show()
    
    return df_spearman_pivot


# In[ ]:





# In[54]:


# spearman 상관계수 구하기 

df_corr_spearman_NS = data_NS.corr(method='spearman')
df_corr_spearman_NS.to_csv("corr_spearman_healthy_colonic_tissue.csv") # csv 파일로 correlation 값 저장 
df_corr_spearman_NS


# In[55]:


import numpy as np
from scipy import stats
arr_NS=data_NS.T.values
arr_NS


# In[56]:


data_NS.head()


# In[ ]:





# In[ ]:





# In[ ]:





# df_corr_spearman_NS[df_corr_spearman_NS<0].count()

# sort_df(data_NS)

# In[57]:


df_spearman_NS = spear_df(sort_df(data_NS))
df_spearman_NS.to_csv("corr_spearman_pval_healthy_colonic_tissue.csv")
df_spearman_NS


# In[58]:


df_spearman_NS[df_spearman_NS['spearman']<0]


# In[59]:


df_spearman_NS[df_spearman_NS['pvalue']>0.05]


# In[60]:


df_spearman_NS [(df_spearman_NS['spearman']<0) | (df_spearman_NS['pvalue']>0.05)]


# In[ ]:





# In[61]:


remove_noise(data_NS, df_spearman_NS)


# In[ ]:





# In[62]:


heatmap_spearman(df_spearman_NS, "healthy_colonic_tissue")


# In[ ]:





# In[ ]:





# In[63]:


# spearman 상관계수 구하기 

df_corr_spearman_TubA = data_TubA.corr(method='spearman')
df_corr_spearman_TubA.to_csv("corr_spearman_Tubulovillous_adenoma.csv") # csv 파일로 correlation 값 저장 
df_corr_spearman_TubA


# In[64]:


data_TubA.head()


# In[65]:


print(df_corr_spearman_TubA[df_corr_spearman_TubA<0].count())
df_corr_spearman_TubA[df_corr_spearman_TubA<0].count().sum()


# In[66]:


df_spearman_TubA = spear_df(sort_df(data_TubA))
df_spearman_TubA.to_csv("corr_spearman_pval_Tubulovillous_adenoma.csv")
df_spearman_TubA.head()


# In[67]:


df_spearman_TubA[df_spearman_TubA['spearman']>0]


# In[68]:


remove_noise(data_TubA, df_spearman_TubA)


# In[69]:


heatmap_spearman(df_spearman_TubA, 'Tubulovillous_adenoma')


# In[ ]:





# In[70]:


# spearman 상관계수 구하기 

df_corr_spearman_SSP = data_SSP.corr(method='spearman')
df_corr_spearman_SSP.to_csv("corr_spearman_sessile_serrated_adenoma.csv") # csv 파일로 correlation 값 저장 
df_corr_spearman_SSP


# In[71]:


df_corr_spearman_SSP[df_corr_spearman_SSP<0].count()


# In[ ]:





# In[72]:


df_spearman_SSP = spear_df(sort_df(data_SSP))
df_spearman_SSP.to_csv("corr_spearman_pval_sessile_serrated_adenoma.csv")
df_spearman_SSP.head()


# In[73]:


df_spearman_SSP[df_spearman_SSP['spearman']>0]


# In[74]:


remove_noise(data_SSP, df_spearman_SSP)


# In[75]:


heatmap_spearman(df_spearman_SSP, 'sessile_serrated_adenoma')


# In[ ]:





# In[ ]:





# In[76]:


# spearman 상관계수 구하기 

df_corr_spearman_AP = data_AP.corr(method='spearman')
df_corr_spearman_AP.to_csv("corr_spearman_adenomatous_polyp.csv") # csv 파일로 correlation 값 저장 
df_corr_spearman_AP


# In[77]:


df_corr_spearman_AP[df_corr_spearman_AP<0].count()


# In[ ]:





# In[78]:


df_spearman_AP = spear_df(sort_df(data_AP))
df_spearman_AP.to_csv("corr_spearman_pval_adenomatous_polyp.csv")
df_spearman_AP.head()


# In[79]:


df_spearman_AP[df_spearman_AP['spearman']>0]


# In[80]:


remove_noise(data_AP, df_spearman_AP)


# In[81]:


heatmap_spearman(df_spearman_AP, 'adenomatous_polyp')


# In[ ]:





# In[ ]:





# In[82]:


# spearman 상관계수 구하기 

df_corr_spearman_Cancer = data_Cancer.corr(method='spearman')
df_corr_spearman_Cancer.to_csv("corr_spearman_Cancer.csv") # csv 파일로 correlation 값 저장 
df_corr_spearman_Cancer


# In[83]:


df_corr_spearman_Cancer[df_corr_spearman_Cancer<0].count()


# In[84]:


df_spearman_Cancer = spear_df(sort_df(data_Cancer))
df_spearman_Cancer.to_csv("corr_spearman_pval_Cancer.csv")
df_spearman_Cancer.head()


# In[85]:


df_spearman_Cancer[df_spearman_Cancer['spearman']>0]


# In[86]:


remove_noise(data_Cancer, df_spearman_Cancer)


# In[87]:


heatmap_spearman(df_spearman_Cancer, 'Cancer')


# In[ ]:





# In[ ]:





# In[88]:


# spearman 상관계수 구하기 

df_corr_spearman_TA = data_TA.corr(method='spearman')
df_corr_spearman_TA.to_csv("corr_spearman_tubular adenoma.csv") # csv 파일로 correlation 값 저장 
df_corr_spearman_TA


# In[89]:


df_corr_spearman_TA[df_corr_spearman_TA<0].count()


# In[90]:


df_spearman_TA = spear_df(sort_df(data_TA))
df_spearman_TA.to_csv("corr_spearman_pval_tubular_adenoma.csv")
df_spearman_TA.head()


# In[91]:


df_spearman_TA[df_spearman_TA['spearman']>0]


# In[92]:


remove_noise(data_TA, df_spearman_TA)


# In[93]:


heatmap_spearman(df_spearman_TA, 'tubular_adenoma')


# In[ ]:





# In[ ]:





# <b> 샘플 -> 숫자 매핑 </b>

# In[94]:


num=1
for i in combine_sampleName:
    print(i)
    predict.loc[predict["predict"] == i,:] = num
    num =num+1
print(num)
predict.head(50)


# In[ ]:





# <b> Kmeans 수행 </b>

# In[88]:


# create model and prediction
model = KMeans(n_clusters=3,algorithm='auto')
model.fit(data)

# concatenate labels to df as a new column
r = pd.concat([data, predict],axis=1)

print(r)
r.tail()


# In[89]:


ks = range(1,20)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(r)
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()


# In[90]:


ks = range(1,30)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(r)
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()


# model = KMeans(n_clusters=14)
# model.fit(r)
# 
# centers = pd.DataFrame(model.cluster_centers_)
# center_x = centers[0]
# center_y = centers[1]
# 
# plt.scatter(center_x,center_y,s=20,marker='D',c='r')
# plt.show()

# In[ ]:





# In[ ]:





# ### PCA 수행
# https://ssungkang.tistory.com/entry/%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5-PCA-2-python-%EC%9D%84-%EC%82%AC%EC%9A%A9%ED%95%9C-PCA-%EC%8B%A4%EC%8A%B5 참고

# In[91]:


X=data.to_numpy()


# In[92]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

pca=PCA() #주성분 개수 지정하지 않고 클래스생성
pca.fit(X)  #주성분 분석
cumsum = np.cumsum(pca.explained_variance_ratio_) #분산의 설명량을 누적합
num_d = np.argmax(cumsum >= 0.95) + 1 # 분산의 설명량이 95%이상 되는 차원의 수
num_d


# http://textmining.kr/?p=362

# In[93]:


pca = PCA(n_components = 0.95)
principalComponents = pca.fit_transform(data)
principalDf = pd.DataFrame(data = principalComponents)

principalDf.head()


# In[94]:


y=pd.DataFrame(r['predict'].tolist())
y.columns=['predict']
y.head()


# In[95]:


finalDataFrame =pd.concat([principalDf, y], axis=1)
finalDataFrame.head()


# In[96]:


labels = []
yList = y.values.tolist()
for label in yList:
    if label[0] not in labels:
        labels.append(label[0])
    
labels


# In[97]:


sortedLabels=sorted(labels)
sortedLabels


# In[100]:


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
plt.savefig("only-PCA.png")


# In[101]:


ks = range(1,30)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(principalDf)
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()


# In[102]:


model = KMeans(n_clusters=10)
model.fit(principalDf)
centers = pd.DataFrame(model.cluster_centers_)
print(centers.shape)
centers.head()


# In[103]:


centers = pd.DataFrame(model.cluster_centers_)
center_x = centers[0]
center_y = centers[1]

# scatter plot
plt.scatter(finalDataFrame[0],finalDataFrame[1], alpha=0.5)#c=finalDataFrame['predict']
plt.scatter(center_x,center_y,s=10,marker='D',c='r')
plt.show()


# In[107]:


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
               , s = 20
              )

    
centers = pd.DataFrame(model.cluster_centers_)
center_x = centers[0]
center_y = centers[1]

# scatter plot
plt.scatter(center_x,center_y,s=40,marker='D',c='r')

labels=combine_sampleName+["centers"]
ax.legend(labels)
plt.savefig("PCA+IKM.png")
ax.grid()


# In[ ]:





# In[ ]:




