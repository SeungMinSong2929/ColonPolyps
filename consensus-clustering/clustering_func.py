import pandas as pd
import numpy as np
import pyclustering
from pandas import DataFrame
from pyclustering.cluster.cure import cure
from pyclustering.cluster.birch import birch

def cure_func(data,k):
    data=DataFrame(data)
    data=data.apply(pd.to_numeric)
    X=data.to_numpy()
    cure_instance=cure(X,int(k))
    cure_instance.process()
    clusters = cure_instance.get_clusters()
    return clusters
    
def BIRCH_func(data,k):
    data=DataFrame(data)
    data=data.apply(pd.to_numeric)
    X=data.values.tolist()
    birch_instance = birch(X, k, diameter=3.0)
    birch_instance.process()
    clusters = birch_instance.get_clusters()
    return clusters
