library(parallel)
library(diceR)
library(ggplot2)
library(ggfortify)
library(cluster)
library(Rtsne)
library(data.table)
library(reticulate)
use_python("/usr/local/bin/python")
source_python("clustering_func.py")

pyclustering<-import('pyclustering')
np<-import('numpy')
pd<-import('pandas')

colon_data_df<-pd$read_table("../dataFile/Colon_merged.txt")
colon_data<-fread("../dataFile/Colon_merged.txt")

colon_data<-colon_data[, -12956]
lable<-colon_data[, 12955]
colon_data<-colon_data[, -12955]
row_name<-unlist(colon_data[,1])
colon_data<-colon_data[, -1]


#return형 고정을 위해 함수 재정의
CURE<-function(df, k){
    arr<-NULL
    cure_cluster<-cure_func(df,as.integer(k))
    for(i in 1:k){
        for(j in 1:length(cure_cluster[[i]])){
            arr[(cure_cluster[[i]][j]+1)]<-i
        }
    }
    arr<-as.integer(arr)
    return(arr)
}

BIRCH<-function(df, k){
    arr<-NULL
    birch_cluster<-BIRCH_func(df,as.integer(k))
    for(i in 1:k){
        for(j in 1:length(birch_cluster[[i]])){
            arr[(birch_cluster[[i]][j]+1)]<-i
        }
    }
    arr<-as.integer(arr)
    return(arr)
}



cc <- consensus_cluster(colon_data, nk=19, reps = 10, algorithms = c("CURE","km"), distance = c("euclidean"), progress = T)
cc

kmeans_result<-as.integer(stats::kmeans(colon_data,19)$cluster)
kmeans_result
