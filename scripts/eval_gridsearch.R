rm(list=ls())

# import libraries
library("ggplot2")
library("reshape2")
library("dplyr")

# import data
setwd('#placeholder#')

data <- read.csv('CV_history_gridsearch.tsv', header=TRUE, sep="\t", row.names = NULL)
data <- subset(data, select = -c(epoch))
colnames(data)[colnames(data) == 'row.names'] <- 'epoch'
data$epoch <- as.numeric(data$epoch)

data_auc <- read.csv('AUC_history_gridsearch.tsv', header=TRUE, sep="\t")

# find gridsearch variables
group_vars <- setdiff(names(data), c('acc', 'val_acc', 
                                     'loss', 'val_loss', 
                                     'matthews', 'val_matthews', 
                                     'precision', 'val_precision',
                                     'recall', 'val_recall'))

group_vars_auc <- setdiff(names(data_auc), c('auc_train', 'auc_val', 'auc_cum_val', 
                                             'matthews_train', 'matthews_val',
                                             'n_train', 'n_val'))

grid_vars <- data[setdiff(group_vars, c('epoch', 'cv_num'))] %>% 
  #mutate_all(funs(length(unique(.)))) %>%
  mutate_all(list(~length(unique(.)))) %>%
  distinct() %>%
  select_if(~(max(.) > 1))

grid_formula <- as.formula(paste(paste(names(grid_vars), collapse="+"), '~ variable'))

measurevar <- setdiff(names(grid_vars), c('n_window'))[1]  
groupvars <- setdiff(names(grid_vars), c('n_window', measurevar))
groupvars <- ifelse(length(groupvars)==0, '.', groupvars)
grid_formula_auc <-  as.formula(paste(measurevar, paste(groupvars, collapse=" + "), sep=" ~ "))


# find model with highest mean AUC in validation data
best_model_auc <- data_auc %>%
  group_by(.dots=names(grid_vars)) %>%
  mutate(auc_mean = mean(auc_cum_val),
         n = n()) %>%
  ungroup() %>%
  filter(n == max(cv_num)) %>%
  filter(auc_mean == max(auc_mean))

# find model with highest mean MCC in validation data
best_model_mcc <- data_auc %>%
  group_by(.dots=names(grid_vars)) %>%
  mutate(mcc_mean = mean(matthews_val),
         n = n()) %>%
  ungroup() %>%
  filter(n == max(cv_num)) %>%
  filter(mcc_mean == max(mcc_mean))


# melt data
data.melt <- melt(data, id=c(group_vars))
data_auc.melt <- melt(data_auc, id=group_vars_auc) 


# extract matthews and loss
data_matthews <- subset(data.melt, variable == 'matthews' | variable == 'val_matthews')
data_loss <- subset(data.melt, variable == 'loss' | variable == 'val_loss')
data_recall <- subset(data.melt, variable == 'recall' | variable == 'val_recall')
data_precision <- subset(data.melt, variable == 'precision' | variable == 'val_precision')



matthews <- ggplot(data_matthews, aes(x=epoch, y=value, color=factor(cv_num))) +
  facet_grid(grid_formula) +
  geom_smooth(alpha=0.5, size = 0.3)
#print(matthews)


loss <- ggplot(data_loss, aes(x=epoch, y=value, color=factor(cv_num))) +
  facet_grid(grid_formula) +
  geom_smooth(alpha=0.5, size = 0.3)
#print(loss)

data_auc.melt.subset  = subset(data_auc.melt, variable %in% c('auc_train', 'auc_val', 'auc_cum_val'))

auc <- ggplot(data_auc.melt.subset, aes(y=value, x=as.factor(n_window))) +
  facet_grid(grid_formula_auc) +
  #geom_boxplot(aes(fill=as.factor(units)), outlier.colour = "red", outlier.shape = 1) +
  geom_boxplot(aes(fill=variable)) +
  theme_bw() +
  theme(strip.background=element_rect(fill="black")) +
  theme(strip.text=element_text(color="white", face="bold")) +
  ylab("AUC") + 
  xlab("Number of windows")
print(auc)

auc <- ggplot(data_auc.melt.subset, aes(y=value, x=as.factor(n_window))) +
  facet_grid(grid_formula_auc) +
  stat_summary(aes(y = value, group=variable, color = variable), fun.y=median, geom="line") +
  theme_bw() +
  theme(strip.background=element_rect(fill="black")) +
  theme(strip.text=element_text(color="white", face="bold")) +
  ylab("AUC") + 
  xlab("Number of windows")
print(auc)
