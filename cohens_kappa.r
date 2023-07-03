library(psych)
library(tidyverse)
library(caret)

#dataset_output_target <- read_tsv("~/programs/Transformer-Formal-Languages/results/SL_results/231-bbb/target_output/SL_2_3_1_u-bbb_d_model_28_depth_2_heads_4_lr_0.001_model_type_SAN_output_target.tsv", 
#                                  col_names = c("epoch", "bins", "output", "target"), 
#                                  trim_ws = TRUE)

setwd("~/programs/Transformer-Formal-Languages/results/SL_results")

stripped_output_target <- read_tsv("./231-bbb/target_output/SL_2_3_1_u-bbb_d_model_18_depth_2_heads_3_lr_0.001_model_type_SAN_output_target.tsv", 
                                   col_names = c("epoch", "bins", "output", "target"), 
                                   trim_ws = TRUE) %>%
  mutate(across(c(output, target), ~gsub("[[:punct:]]", "", .x))) %>%  #strip values of all punctuation ([,])
  mutate(across(c(output, target), ~map(.x, substring2))) %>%
  mutate(output_2 = map2(output, target, ~trim_00(unlist(.x), unlist(.y))), target_2 = map2(target, output, ~trim_00(unlist(.x), unlist(.y)))) %>%
  group_by(epoch, bins) %>% #group values by epochs and bins 
  summarise(output=paste0(unlist(output_2),collapse=""),target=paste0(unlist(target_2),collapse="")) #concatenate all values in the same epoch and bins

target_output_ck <- function(epoch,bin) {
  cohen.kappa(x=cbind(targetselect(epoch,bin), outputselect(epoch,bin)))
}

target_output_confmatrix <- function(epoch,bin){
  confusionMatrix(data=factor(cbind(outputselect(epoch,bin))), reference=factor(cbind(targetselect(epoch,bin))))
}

targetselect <- function(epoch, bin) {
  substring2(stripped_output_target$target[stripped_output_target$epoch==epoch & stripped_output_target$bins==bin])
}

outputselect <- function(epoch, bin) {
  substring2(stripped_output_target$output[stripped_output_target$epoch==epoch & stripped_output_target$bins==bin])
}

trim_00 <- function(txt1, txt2){
  txt1_00 <- which(txt1== "00") #vector of all positions with "00" in txt1
  txt2_00 <- which(txt2== "00")
  if(identical(txt1_00, integer(0)) | identical(txt2_00, integer(0))){
    return(txt1)
  }
  else {
    pos_00 <- txt2_00[which(match(txt1_00,txt2_00, nomatch=0) != 0 )[1]]
    head(txt1, pos_00-1)
  }
}

# find_first_common_00 <- function(txt1, txt2) {
#   txt1_00 <- which(txt1== "00")
#   txt2_00 <- which(txt2== "00")
#   txt2_00[which(match(txt1_00,txt2_00, nomatch=0) != 0 )[1]]
# }

substring2 <- function(text) {
  sst <- strsplit(text, "")[[1]]
  paste0(sst[c(TRUE, FALSE)], sst[c(FALSE, TRUE)])
}

