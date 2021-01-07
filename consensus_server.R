library(diceR)
library(ggplot2)
library(ggfortify)
library(cluster)
library(Rtsne)
library(data.table)

colon_data<-fread("Colon_merged_273samples.txt")
colon_data<-colon_data[, -12980]
lable<-colon_data[, 12979]
colon_data<-colon_data[, -12979]
row_name<-unlist(colon_data[,1])
colon_data<-colon_data[, -1]

#header <- read.table("Colon_merged_273.txt", header = TRUE)
#indata <- fread("Colon_merged.txt", skip=1, header=FALSE)

obj<-dice(colon_data,nk=2:10,reps=10, algorithms = c("hc","km","diana"), k.method = 10 , cons.funs = c("majority","CSPA","LCE"), progress = TRUE, evaluate = F, prep.data = "full")
cons_result<-as.data.frame(obj[["clusters"]],row.names = row_name)
c_data_df<-as.data.frame(colon_data, row.names=row_name)

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
