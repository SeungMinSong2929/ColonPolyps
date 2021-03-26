import pandas as pd
import csv
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import pyclustering
from pandas import DataFrame
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.cure import cure
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import FCPS_SAMPLES
from pyclustering.cluster.birch import birch
from pyclustering.cluster import cluster_visualizer
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import FAMOUS_SAMPLES

def cure_func(df,k):
    #df=DataFrame(df)
    df = pd.read_table('../dataFile/Colon_merged.txt')
    #유전자 이름 list로
    only_geneID=df.columns.tolist()[:-2] #제일 마지막 칼럼이 batch, 마지막에서 2번째 칼럼이 condition
    # Condition-샘플 이름
    condition_list=df['Condition'].tolist()
    my_set = set(condition_list) #집합set으로 변환
    sampleName = list(my_set) #list로 변환
    
    data=df.loc[:,only_geneID]
    column=sampleName[:len(sampleName)+1]
    
    # predict열 새로 생성.
    predict=df[['Condition']]
    predict.columns=['predict']
    
    # SSA/P -> SSA.P로 변경 '/' 인식을 못하기 때문.
    predict.loc[predict["predict"] == 'SSA/P'] = 'SSA.P'
    sampleName[2]='SSA.P' # sampleName에서도 변경.
    
    # 샘플 이름 있는 predict 데이터프레임과 합치기
    r = pd.concat([data, predict],axis=1)
    
    
    # 샘플-> 숫자 매핑(목적: PCA 하기 위함)
    # predict 값들이 모두 숫자로 변경.
    num=1
    for i in sampleName:
        #print(i)
        predict.loc[predict["predict"] == i,:] = num
        num =num+1
        
    
    X=data.to_numpy()
    
    pca=PCA() #주성분 개수 지정하지 않고 클래스생성
    pca.fit(X)  #주성분 분석
    cumsum = np.cumsum(pca.explained_variance_ratio_) #분산의 설명량을 누적합
    num_d = np.argmax(cumsum >= 0.95) + 1 # 분산의 설명량이 95%이상 되는 차원의 수
    
    # http://textmining.kr/?p=362
    pca = PCA(n_components = 0.95)
    principalComponents = pca.fit_transform(data)
    principalDf = pd.DataFrame(data = principalComponents)
    
    y=pd.DataFrame(r['predict'].tolist())
    y.columns=['predict']
    
    finalDataFrame =pd.concat([principalDf, y], axis=1)
    
    X=finalDataFrame.iloc[:,[0,1]].to_numpy()
    cure_instance=cure(X,k)
    cure_instance.process()
    clusters = cure_instance.get_clusters()
    return clusters
    
    
def BIRCH_func(data,k):
    df = pd.read_table('../dataFile/201126/Colon_merged_273samples.txt')
    only_geneID=df.columns.tolist()[:-2] 
    condition_list=df['Condition'].tolist()
    my_set = set(condition_list) #집합set으로 변환
    sampleName = list(my_set) #list로 변환
    
    data=df.loc[:,only_geneID]
    
    column=sampleName[:len(sampleName)+1]
       
    predict=df[['Condition']]
    predict.columns=['predict']
    
    predict.loc[predict["predict"] == 'SSA/P'] = 'SSA.P'
    sampleName[2]='SSA.P' # sampleName에서도 변경.
    
    r = pd.concat([data, predict],axis=1)
    
    num=1
    for i in sampleName:
        #print(i)
        predict.loc[predict["predict"] == i,:] = num
        num =num+1
        
    
    X=data.to_numpy()
    
    pca=PCA() #주성분 개수 지정하지 않고 클래스생성
    pca.fit(X)  #주성분 분석
    cumsum = np.cumsum(pca.explained_variance_ratio_) #분산의 설명량을 누적합
    num_d = np.argmax(cumsum >= 0.95) + 1 # 분산의 설명량이 95%이상 되는 차원의 수

    pca = PCA(n_components = 0.95)
    principalComponents = pca.fit_transform(data)
    principalDf = pd.DataFrame(data = principalComponents)

    y=pd.DataFrame(r['predict'].tolist())
    y.columns=['predict']

    finalDataFrame =pd.concat([principalDf, y], axis=1)

    labels = []
    yList = y.values.tolist()
    for label in yList:
        if label[0] not in labels:
            labels.append(label[0])

    #sortedLabels=sorted(labels)

    X=finalDataFrame.iloc[:,[0,1]].values.tolist()
    birch_instance = birch(X, int(k), diameter=3.0)
    birch_instance.process()
    clusters = birch_instance.get_clusters()
    return clusters
