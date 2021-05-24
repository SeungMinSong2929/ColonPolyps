library(parallel)
library(diceR)
library(ggplot2)
library(ggfortify)
library(cluster)
library(Rtsne)
library(data.table)
library(reticulate)
library(tidyverse)
library(rdist)
use_python("/usr/local/bin/python")
source_python("clustering_func.py")

pyclustering<-import('pyclustering')
np<-import('numpy')
pd<-import('pandas')

colon_data<-pd$read_table("../dataFile/Colon_merged.txt")

colon_data<-subset(colon_data,Condition!="Control")
colon_data<-subset(colon_data,Condition!="Cancer")
colon_data$Condition[colon_data$Condition == 'tubular adenoma'] <- "TA"
colon_data$Condition[colon_data$Condition == 'Tubular adenoma'] <- "TA"

colon_data$Condition[colon_data$Condition == 'tubulovillous polyp'] <- "TubA"
colon_data$Condition[colon_data$Condition == 'Tubulovillous adenoma'] <- "TubA"
colon_data$Condition[colon_data$Condition == 'Villous / tubulovillous adenoma'] <- "TubA"
colon_data$Condition[colon_data$Condition == 'tubulovillous adenoma'] <- "TubA"
colon_data$Condition[colon_data$Condition == 'villous adenoma'] <- "TubA"

colon_data$Condition[colon_data$Condition == 'sessile serrated adenoma'] <- "SSP"
colon_data$Condition[colon_data$Condition == 'SSA/P'] <- "SSP"

colon_data$Condition[colon_data$Condition == 'adenomatous polyp'] <- "AP"

condition_name<-colon_data[, 12954]
colon_data<-colon_data[,-12955]
colon_data<-colon_data[,-12954]

# RESULT_ARR<-consensus_cluster(colon_data,nk=19,algorithms = c("CURE","BIRCH","km"), progress = T)

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

# 
# AA<-dice(colon_data,2:20,algorithm=c("km"),progress = T)

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

km <- function(x, k) {
    as.integer(stats::kmeans(x, k)$cluster)
}


"%||%" <- devtools:::`%||%`
init_array <- function(data, r, a, k) {
    rn <- rownames(data) %||% seq_len(nrow(data))
    dn <- list(rn, paste0("R", seq_len(r)), a, k)
    array(NA_integer_, dim = purrr::map_int(dn, length), dimnames = dn)
}

rep<-10
k<-10
arr <- init_array(colon_data, rep, c("CURE", "BIRCH", "km"), k)
for(i in 1:rep){
    ind.new <- sample(nrow(colon_data), floor(nrow(colon_data) * 0.8))
    x <- colon_data[ind.new, ]
    dists <- cdist(x,x,"euclidean")
    data<-CURE(dists,k)
    arr[ind.new, i, 1, 1]<-data
}

for(i in 1:rep){
    ind.new <- sample(nrow(colon_data), floor(nrow(colon_data) * 0.8))
    x <- colon_data[ind.new, ]
    data<-BIRCH(x,k)
    arr[ind.new, i, 2, 1]<-data
}

for(i in 1:rep){
    ind.new <- sample(nrow(colon_data), floor(nrow(colon_data) * 0.8))
    x <- colon_data[ind.new, ]
    data<-km(x,k)
    arr[ind.new, i, 3, 1]<-data
}

save.image(file = "clustering_result.RData")
arr
# 
load("clustering_result.RData")

x <- apply(arr, 2:4, impute_knn, data = colon_data, seed = 1)
x
x_impute <- impute_missing(x, colon_data, k)

result<-CSPA(x,k)
result_lce<-LCE(x_impute,k)
result_lce

result

data_sum<-data.frame()
for(i in 1:10){
    data_sum[i,1]<-length(which(result==i))
    data_sum[i,2]<-length(which(result_lce==i))
}

for(i in 1:10){
    data_sum_lce[1,i]<-length(which(result_lce==i))
}

condition_list<-c("adenovillous polyp", "AP", "Cancerncer", "FAP", "HP", "serrated adenoma", "SSP", "TA", "TSA", "TubA", "villoglandular polyp")

data_sum_condition<-data.frame()
for(i in 1:11){
    data_sum_condition[1,i]<-length(which(condition_name==condition_list[i]))
}

library(RColorBrewer)
colnames(data_sum)<-c("CSPA","LCE")
data_sum
barplot(as.matrix(data_sum), main="consensus clustering summary", 
        beside = T, ylim = c(0,200), col=brewer.pal(n = 10, name = "RdYlBu") )
legend("topright",legend=c(1:10), fill = brewer.pal(n = 10, name = "RdYlBu"))
barplot(data_sum_lce)
barplot(data_sum_condition)

ggplot(data = (prcomp(colon_data, center = T, scale. = T)), aes(x = PC1, y = PC2))+
    geom_point(aes(color=as.factor(result)),size=3)+
    ggtitle("consensus clustering with CSPA")+theme(plot.title = element_text(hjust = 0.5))+
    geom_text(aes(label=condition_name, size=1, vjust=-1, hjust=0), size = 2.5, color = "black", alpha = 0.5)

ggplot(data = (prcomp(colon_data, center = T, scale. = T)), aes(x = PC1, y = PC2))+
    geom_point(aes(color=as.factor(result_lce)),size=3)+
    ggtitle("consensus clustering with LCE")+theme(plot.title = element_text(hjust = 0.5))+
    geom_text(aes(label=condition_name, size=1, vjust=-1, hjust=0), size = 2.5, color = "black", alpha = 0.5)

ggplot(data = (prcomp(colon_data, center = T, scale. = T)), aes(x = PC1, y = PC2))+
    geom_point(aes(color=as.factor(condition_name)),size = 3)+
    ggtitle("reference condition")+theme(plot.title = element_text(hjust = 0.5))+
    geom_text(aes(label=condition_name, size=1, vjust=-1, hjust=0), size = 2.5, color = "black", alpha = 0.5)
