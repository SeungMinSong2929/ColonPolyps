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

colon_data<-pd$read_table("../dataFile/Colon_merged.txt")

colon_data<-colon_data[, -12955]
colon_data<-colon_data[, -12954]
#row_name<-unlist(colon_data[,1])
#colon_data<-colon_data[, -1]


#return형 고정을 위해 함수 재정의
CURE<-function(df, k){
    arr<-NULL
    cure_cluster<-cure_func(df,as.integer(k))
    for(i in 1:k){
        for(j in 1:length(cure_cluster[[i]])){
            arr[(cure_cluster[[i]][j])+1]<-i
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
            arr[(birch_cluster[[i]][j])+1]<-i
        }
    }
    arr<-as.integer(arr)
    return(arr)
}

"%||%" <- devtools:::`%||%`

init_array <- function(data, r, a, k) {
    rn <- rownames(data) %||% seq_len(nrow(data))
    dn <- list(rn, paste0("R", seq_len(r)), a, k)
    array(NA_integer_, dim = purrr::map_int(dn, length), dimnames = dn)
}

#install.packages("devtools")


cc <- consensus_cluster(colon_data, nk=19, reps = 10, algorithms = c("CURE"), distance = c("euclidean"), progress = T, prep.data="sampled")
cc

kmeans_result<-as.integer(stats::kmeans(colon_data,19)$cluster)
kmeans_result

#install.packages("rdist")
library(rdist)
rep<-10
arr <- init_array(colon_data, rep, c("CURE", "BIRCH"), 19)
for(i in 1:rep){
    ind.new <- sample(nrow(colon_data), floor(nrow(colon_data) * 0.8))
    x <- colon_data[ind.new, ]
    dists <- cdist(x,x,"euclidean")
    data<-CURE(dists,19)
    arr[ind.new, i, 1, 1]<-data
}
for(i in 1:rep){
    ind.new <- sample(nrow(colon_data), floor(nrow(colon_data) * 0.8))
    x <- colon_data[ind.new, ]
    dists <- cdist(x,x,"euclidean")
    data<-BIRCH(dists,19)
    arr[ind.new, i, 2, 1]<-data
}

a<-BIRCH(colon_data,19)
a.any()
arr
arr <- apply(arr, 2:4, impute_knn, data = colon_data, seed = 1)
arr
arr<-impute_missing(arr, colon_data, 19)
arr


result<-LCE(E = arr, k = 19, sim.mat = "cts")
result<-CSPA(arr, k = 19)
result


autoplot(prcomp(colon_data, center = T, scale. = T), 
         data = colon_data, colour =as.factor(result), size = 3, label = F,
         loadings = FALSE, loadings.label = FALSE, loadings.label.colour = "black",
         loadings.colour = 'black') +
    theme_bw() + 
    theme(legend.direction = 'horizontal', legend.position = 'top')
