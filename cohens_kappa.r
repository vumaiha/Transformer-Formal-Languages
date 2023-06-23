library(psych)
library(tidyverse)

#dataset_output_target <- read_tsv("~/programs/Transformer-Formal-Languages/results/SL_results/231-bbb/target_output/SL_2_3_1_u-bbb_d_model_28_depth_2_heads_4_lr_0.001_model_type_SAN_output_target.tsv", 
#                                  col_names = c("epoch", "bins", "output", "target"), 
#                                  trim_ws = TRUE)

setwd("~/programs/Transformer-Formal-Languages/results/SL_results")

stripped_output_target <- read_tsv("./231-bbb/target_output/SL_2_3_1_u-bbb_d_model_28_depth_2_heads_4_lr_0.001_model_type_SAN_output_target.tsv", 
                                   col_names = c("epoch", "bins", "output", "target"), 
                                   trim_ws = TRUE) %>%
  mutate(across(everything(), ~gsub("[[:punct:]]", "", .x))) %>% #strip values of all punctuation ([,])
  group_by(epoch, bins) %>% #group values by epochs and bins
  summarise(output=paste0(output,collapse=""),target=paste0(target,collapse="")) #concatenate all values in the same epoch and bins

target_output_ck <- function(epoch,bin) {
  cohen.kappa(x=cbind(targetselect(epoch,bin),outputselect(epoch,bin)))
}

targetselect <- function(epoch, bin) {
  substring2(stripped_output_target$target[stripped_output_target$epoch==epoch & stripped_output_target$bins==bin])
}

outputselect <- function(epoch, bin) {
  substring2(stripped_output_target$output[stripped_output_target$epoch==epoch & stripped_output_target$bins==bin])
}

substring2 <- function(text) {
  sst <- strsplit(text, "")[[1]]
  paste0(sst[c(TRUE, FALSE)], sst[c(FALSE, TRUE)])
}

