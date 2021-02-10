if(!require(parallel)) {
    install.packages("parallel")
}
library(parallel)
library(diceR)
library(ggplot2)
library(ggfortify)
library(cluster)
library(Rtsne)
library(data.table)
library(reticulate)
use_python("/usr/local/bin/python")

detectCores()

pyclustering<-import('pyclustering')
pyclustering$cluster$cure$cure
cure<-py_to_r(pyclustering$cluster$cure$cure)

colon_data<-fread("../dataFile/201126/Colon_merged_273samples.txt")

colon_data<-colon_data[, -12980]
lable<-colon_data[, 12979]
colon_data<-colon_data[, -12979]
row_name<-unlist(colon_data[,1])
colon_data<-colon_data[, -1]

obj<-dice(colon_data,nk=20,reps=10, algorithms = c("km","diana","hc"), 
          cons.funs = c("majority","CSPA","LCE"), progress = TRUE ,prep.data = "full")

autoplot(prcomp(c_data_df, center = T, scale. = T), 
         data = c_data_df, colour =as.factor(cons_result$majority), size = 3, label = F,
         loadings = FALSE, loadings.label = FALSE, loadings.label.colour = "black",
         loadings.colour = 'black') +
    theme_bw() + 
    theme(legend.direction = 'horizontal', legend.position = 'top')

autoplot(prcomp(c_data_df, center = T, scale. = T), 
         data = c_data_df, colour =as.factor(cons_result$CSPA), size = 3, label = F,
         loadings = FALSE, loadings.label = FALSE, loadings.label.colour = "black",
         loadings.colour = 'black') +
    theme_bw() + 
    theme(legend.direction = 'horizontal', legend.position = 'top')

autoplot(prcomp(c_data_df, center = T, scale. = T), 
         data = c_data_df, colour =as.factor(cons_result$LCE), size = 3, label = F,
         loadings = FALSE, loadings.label = FALSE, loadings.label.colour = "black",
         loadings.colour = 'black') +
    theme_bw() + 
    theme(legend.direction = 'horizontal', legend.position = 'top')


#현재 cure 오류남
obj<-dice(colon_data,nk=20,reps=10, algorithms = c("km","diana","hc","cure"), 
          cons.funs = c("majority","CSPA","LCE"), progress = TRUE , evaluate = T ,prep.data = "full")



