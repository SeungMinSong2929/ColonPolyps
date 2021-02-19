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

def cure_func(k):
    df = pd.read_table('../dataFile/201126/Colon_merged_273samples.txt')
    #유전자 이름 list로
    only_geneID=df.columns.tolist()[:-2] #제일 마지막 칼럼이 batch, 마지막에서 2번째 칼럼이 condition
    # Condition-샘플 이름
    condition_list=df['Condition'].tolist()
    my_set = set(condition_list) #집합set으로 변환
    sampleName = list(my_set) #list로 변환
    # batch
    batch_list=df['batch'].tolist()
    batch_set = set(batch_list) #집합set으로 변환
    batchName = list(batch_set) #list로 변환
    
    data=df.loc[:,only_geneID]
    
    column=sampleName[:len(sampleName)+1]
    
    # 각 샘플별 개수 
    # for i in sampleName[1:]:
    #     print(i,"\t+",df.loc[df["Condition"] == i,:].shape[0])
    
    df.loc[df["Condition"].isna()] # df.loc[["SSP_4","SSP_8"],:] 같은 의미

    
    # https://note.espriter.net/1326
    
    #predict=pd.DataFrame(df['Condition'].tolist())
    predict=df[['Condition']]
    predict.columns=['predict']
    
    # <b> 이름은 다른데, 같은 샘플들 합치기 </b>
    
    # healthy colonic tissue = control = NAN = NS <br>
    # Tubulovillous adenoma = Villous/Tubluovillous adenoma = tubulovillous polyp = tubulovillous adenoma <br>
    # sessile serrated adenoma = SSA/P = SSP <br> 
    # adenomatous polyp = AP <br>
    # Cancer = Ca = colorectal adenocarcinoma <br>
    
    combine_sampleName=sampleName[:]
    
    NS=['healthy colonic tissue', 'Control', 'NS']
    
    for i in NS:
        predict.loc[predict["predict"] == i,:] = NS[0]
    
    # NAN값인 샘플들 NS[0]으로 변경
    predict.loc[["SSP_4"],"predict"]=NS[0]
    predict.loc[["SSP_8"],"predict"]=NS[0]
    
    # predict-샘플 이름
    predict_list=predict['predict'].tolist()
    predict_set = set(predict_list) #집합set으로 변환
    combine_sampleName = list(predict_set) #list로 변환
    
    TubA=['Tubulovillous adenoma', 'Villous / tubulovillous adenoma', 'tubulovillous polyp', 'tubulovillous adenoma', 'villous adenoma']
    
    for i in TubA:
        predict.loc[predict["predict"] == i,:] = TubA[0]
        
    # predict-샘플 이름
    predict_list=predict['predict'].tolist()
    predict_set = set(predict_list) #집합set으로 변환
    combine_sampleName = list(predict_set) #list로 변환
    
    SSP=['sessile serrated adenoma', 'SSA/P', 'SSP']

    for i in SSP:
        predict.loc[predict["predict"] == i,:] = SSP[0]
        
    # predict-샘플 이름
    predict_list=predict['predict'].tolist()
    predict_set = set(predict_list) #집합set으로 변환
    combine_sampleName = list(predict_set) #list로 변환
    
    
    AP=['adenomatous polyp', 'AP']
    
    for i in AP:
        predict.loc[predict["predict"] == i,:] = AP[0]
        
    predict_list=predict['predict'].tolist()
    predict_set = set(predict_list) #집합set으로 변환
    combine_sampleName = list(predict_set) #list로 변환
    
    Cancer=['Cancer', 'Ca', 'Colorectal adenocarcinoma']
    
    for i in Cancer:
        predict.loc[predict["predict"] == i,:] = Cancer[0]
        
    # predict-샘플 이름
    predict_list=predict['predict'].tolist()
    predict_set = set(predict_list) #집합set으로 변환
    combine_sampleName = list(predict_set) #list로 변환

    TA=['tubular adenoma', 'Tubular adenoma']
    
    for i in TA:
        predict.loc[predict["predict"] == i,:] = TA[0]
        
    # predict-샘플 이름
    predict_list=predict['predict'].tolist()
    predict_set = set(predict_list) #집합set으로 변환
    combine_sampleName = list(predict_set) #list로 변환
    
    # for i in combine_sampleName:
    #     print(i,"\t+",predict.loc[predict["predict"] == i,:].shape[0])
    
    num=1
    for i in combine_sampleName:
        predict.loc[predict["predict"] == i,:] = num
        num =num+1
    
    # concatenate labels to df as a new column
    r = pd.concat([data, predict],axis=1)
    
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
    
    
def BIRCH_func(k):
    import pandas as pd
    import csv
    import numpy
    
    df = pd.read_table('../dataFile/201126/Colon_merged_273samples.txt')
    only_geneID=df.columns.tolist()[:-2] #제일 마지막 칼럼이 batch, 마지막에서 2번째 칼럼이 condition
    condition_list=df['Condition'].tolist()
    my_set = set(condition_list) #집합set으로 변환
    sampleName = list(my_set) #list로 변환
    batch_list=df['batch'].tolist()
    batch_set = set(batch_list) #집합set으로 변환
    batchName = list(batch_set) #list로 변환
    
    data=df.loc[:,only_geneID]

    column=sampleName[:len(sampleName)+1]

    for i in sampleName[1:]:
        print(i,"\t+",df.loc[df["Condition"] == i,:].shape[0])


    df.loc[df["Condition"].isna()] # df.loc[["SSP_4","SSP_8"],:] 같은 의미

    predict=df[['Condition']]
    predict.columns=['predict']

    predict.loc[["SSP_4"],"predict"]=0
    predict.loc[["SSP_8"],"predict"]=0

    num=1
    for i in sampleName[1:]:
        predict.loc[predict["predict"] == i,:] = num
        num =num+1
    
    # concatenate labels to df as a new column
    r = pd.concat([data, predict],axis=1)

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

    sortedLabels=sorted(labels)

    X=finalDataFrame.iloc[:,[0,1]].values.tolist()
    birch_instance = birch(X, k, diameter=3.0)
    birch_instance.process()
    clusters = birch_instance.get_clusters()
    return clusters
