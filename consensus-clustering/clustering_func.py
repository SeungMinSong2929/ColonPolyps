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
    #������ �̸� list��
    only_geneID=df.columns.tolist()[:-2] #���� ������ Į���� batch, ���������� 2��° Į���� condition
    # Condition-���� �̸�
    condition_list=df['Condition'].tolist()
    my_set = set(condition_list) #����set���� ��ȯ
    sampleName = list(my_set) #list�� ��ȯ
    
    data=df.loc[:,only_geneID]
    column=sampleName[:len(sampleName)+1]
    
    # predict�� ���� ����.
    predict=df[['Condition']]
    predict.columns=['predict']
    
    # SSA/P -> SSA.P�� ���� '/' �ν��� ���ϱ� ����.
    predict.loc[predict["predict"] == 'SSA/P'] = 'SSA.P'
    sampleName[2]='SSA.P' # sampleName������ ����.
    
    # ���� �̸� �ִ� predict �����������Ӱ� ��ġ��
    r = pd.concat([data, predict],axis=1)
    
    
    # ����-> ���� ����(����: PCA �ϱ� ����)
    # predict ������ ��� ���ڷ� ����.
    num=1
    for i in sampleName:
        #print(i)
        predict.loc[predict["predict"] == i,:] = num
        num =num+1
        
    
    X=data.to_numpy()
    
    pca=PCA() #�ּ��� ���� �������� �ʰ� Ŭ��������
    pca.fit(X)  #�ּ��� �м�
    cumsum = np.cumsum(pca.explained_variance_ratio_) #�л��� ������ ������
    num_d = np.argmax(cumsum >= 0.95) + 1 # �л��� ������ 95%�̻� �Ǵ� ������ ��
    
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
    my_set = set(condition_list) #����set���� ��ȯ
    sampleName = list(my_set) #list�� ��ȯ
    
    data=df.loc[:,only_geneID]
    
    column=sampleName[:len(sampleName)+1]
       
    predict=df[['Condition']]
    predict.columns=['predict']
    
    predict.loc[predict["predict"] == 'SSA/P'] = 'SSA.P'
    sampleName[2]='SSA.P' # sampleName������ ����.
    
    r = pd.concat([data, predict],axis=1)
    
    num=1
    for i in sampleName:
        #print(i)
        predict.loc[predict["predict"] == i,:] = num
        num =num+1
        
    
    X=data.to_numpy()
    
    pca=PCA() #�ּ��� ���� �������� �ʰ� Ŭ��������
    pca.fit(X)  #�ּ��� �м�
    cumsum = np.cumsum(pca.explained_variance_ratio_) #�л��� ������ ������
    num_d = np.argmax(cumsum >= 0.95) + 1 # �л��� ������ 95%�̻� �Ǵ� ������ ��

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
