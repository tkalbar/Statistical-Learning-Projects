### Tomasz Kalbarczyk

# Assignment 2 - Predicting transcoding times based on video features

This report has been structured as a walkthrough of all of the code involved in the building the models and some of the results of running that code. An appendix is provided at the end which contains the code in its entirety.

## Walkthrough

Begin by setting up the work environment and retrieving the dataset.
```r
## declare libraries
library("glmnet", lib.loc="/usr/local/lib/R/3.3/site-library")
library("leaps", lib.loc="/usr/local/lib/R/3.3/site-library")
library("caret", lib.loc="/usr/local/Cellar/r/3.3.2/R.framework/Versions/3.3/Resources/library")

## Clear environment variables
rm(list=ls())  

## Set working directory for assignment2 dataset
setwd("~/Dropbox/SDS385/assignment2/online_video_dataset")

## Get dataset
transcoding_data = read.table("transcoding_mesurment.tsv", header = TRUE)
```

Next, we split the dataset in half making sure there is a uniform distribution of utime values across the training and test sets.

```r
## Split data set in half
set.seed(62)
partition = createDataPartition(transcoding_data$utime, p = 0.5, list = FALSE) # partition the data into two sets in a balanced fashion
transcoding_data_train = transcoding_data[partition,]
transcoding_data_test = transcoding_data[-partition,]
```

In order to improve the results of our linear model, we take a number of preprocessing steps which we include in the `preProcessCustom` function.

```r
preProcessCustom<- function(dataset_input)
```
- Remove `id` and zero variance predictors (`b_size`)
```r
## Remove id and constant variables
cur_data = dataset_input
cur_data$id = NULL # not needed
cur_data$b_size = NULL # constant variable
```
- Deal with categorical variables by converting the data frame into a design matrix.
```r
## Deal with categorical variables
matrix = model.matrix(~.,subset(cur_data,select = -utime))
```
- Remove near zero variance predictors (`b`)
```r
## Remove zero and near zero variance predictors
nzv = nearZeroVar(matrix)
matrix_nzv = matrix[,-nzv]
```
- Find correlated predictors, and use an 80% threshold for pruning correlated variables.
```r
## Find correlation in dataset
matrix_cor = cor(matrix_nzv)
  
# print distribution of correlated variables
matrix_cor_uppertri = matrix_cor[upper.tri(matrix_cor)]
summary(matrix_cor_uppertri)
  
# find highly correlated predictors
matrix_highly_cor = findCorrelation(matrix_cor, cutoff = 0.8)
matrix_highly_cor
  
# print info regarding highly correlated predictors
findCorrelation(matrix_cor, cutoff = 0.8, names=TRUE, verbose=TRUE)
  
# printing column names for specific rows that have been identified
for (i in 1:dim(matrix_cor)[2]) {
  cat(colnames(matrix_cor)[i]," ",i, "\n")
}
  
# update matrix, removing correlated predictors
matrix_nocor = matrix_nzv[,-matrix_highly_cor]
```
The pruning process removed an additional six predictors. Below are the results for the flagged predictors and the corresponding correlation values.
```
Compare row 14  and column  13 with corr  0.998 
  Means:  0.306 vs 0.166 so flagging column 14 
Compare row 10  and column  11 with corr  1 
  Means:  0.279 vs 0.153 so flagging column 10 
Compare row 11  and column  9 with corr  0.831 
  Means:  0.24 vs 0.144 so flagging column 11 
Compare row 6  and column  5 with corr  0.991 
  Means:  0.26 vs 0.129 so flagging column 6 
Compare row 5  and column  7 with corr  0.815 
  Means:  0.209 vs 0.118 so flagging column 5 
Compare row 20  and column  21 with corr  0.995 
  Means:  0.089 vs 0.111 so flagging column 21 
All correlations <= 0.8 
```
I have also provided a printed list of which rows the predictors correspond to (for convenience). Notice that `size`, `p`, `height`, `o_height`, `frames`, and `width` were flagged for removal.
```
duration   1 
codech264   2 
codecmpeg4   3 
codecvp8   4 
width   5 
height   6 
bitrate   7 
framerate   8 
i   9 
p   10 
frames   11 
i_size   12 
p_size   13 
size   14 
o_codech264   15 
o_codecmpeg4   16 
o_codecvp8   17 
o_bitrate   18 
o_framerate   19 
o_width   20 
o_height   21 
umem   22 
```
- Center and scale the predictors, and then rebind the response variable to the final dataset before returning it.
```r
# center and scale the predictors
params = preProcess(matrix_nocor, method = c("center", "scale"))
matrix_processed = predict(params, matrix_nocor)
  
# rebind the response variable to the dataset
dataset_processed = cbind(as.data.frame(matrix_processed),cur_data["utime"])
return(dataset_processed)
```

The `preProcessCustom` function described above is called on both the training and test datasets. Now, we are ready to build our four types of linear models.
- Linear regression with all predictors
```r
model_linear_all = lm(utime ~ (.), data=as.data.frame(pre_processed_train))
model_linear_all_predict = predict(model_linear_all, newdata=as.data.frame(pre_processed_test))
model_linear_all_rmse = sqrt(mean((transcoding_data_test$utime - model_linear_all_predict)^2))
model_linear_all_rmse
```
Gives an RMSE of: `9.675779`
- We run three variants for linear regression with BIC: exhaustive, forward and backward. Below is the code for retrieving the subsets.
```r
## Linear regression using BIC
all_ss = regsubsets(utime ~ (.), data = pre_processed_train, nvmax = 400)
forward_ss = regsubsets(utime ~ (.)^2, data = pre_processed_train, nvmax = 400, method = "forward")
backward_ss = regsubsets(utime ~ (.)^2, data = pre_processed_train, nvmax = 400, method = "backward")
```
* Exhaustive
```r
all_bic = summary(all_ss)$bic
u1               = toString(names(coef(all_ss, which.min(all_bic)))[-1])
u2               = gsub(pattern = ", ",  replacement = " + ", x = toString(u1))
all_formula  = as.formula(paste("utime ~ ", u2, sep = ""))
all_formula

model_linear_bic_all = lm(all_formula, data=as.data.frame(pre_processed_train))
model_linear_bic_all_predict = predict(model_linear_bic_all, newdata=as.data.frame(pre_processed_test))
model_linear_bic_all_rmse = sqrt(mean((transcoding_data_test$utime - model_linear_bic_all_predict)^2))
model_linear_bic_all_rmse
```
We get the following formula for minimum BIC.
```
utime ~ duration + codecvp8 + bitrate + framerate + i + i_size + 
    p_size + o_codech264 + o_codecmpeg4 + o_codecvp8 + o_bitrate + 
    o_framerate + o_width + umem
```
And we get an RMSE of: `9.675355`
* Forward
The code is similar (and can be found in its entirety in the Appendix). The formula is below:
```
utime ~ bitrate + framerate + i + i_size + o_codecmpeg4 + o_codecvp8 + 
    o_bitrate + o_framerate + o_width + umem + duration:codech264 + 
    duration:codecvp8 + duration:bitrate + duration:framerate + 
    duration:i_size + duration:p_size + duration:o_codech264 + 
    duration:o_codecvp8 + duration:o_bitrate + duration:o_width + 
    duration:umem + codech264:bitrate + codech264:framerate + 
    codech264:i + codech264:p_size + codech264:o_codech264 + 
    codech264:o_codecmpeg4 + codecmpeg4:framerate + codecmpeg4:i + 
    codecmpeg4:i_size + codecmpeg4:p_size + codecmpeg4:o_codech264 + 
    codecmpeg4:o_codecmpeg4 + codecmpeg4:o_bitrate + codecmpeg4:o_width + 
    codecvp8:framerate + codecvp8:i + codecvp8:i_size + codecvp8:p_size + 
    codecvp8:o_codech264 + codecvp8:o_codecmpeg4 + codecvp8:o_codecvp8 + 
    codecvp8:o_bitrate + codecvp8:umem + bitrate:i + bitrate:i_size + 
    bitrate:p_size + bitrate:o_codech264 + bitrate:o_codecmpeg4 + 
    bitrate:o_codecvp8 + bitrate:o_bitrate + bitrate:o_width + 
    bitrate:umem + framerate:p_size + framerate:o_codech264 + 
    framerate:o_codecmpeg4 + framerate:o_codecvp8 + framerate:o_bitrate + 
    framerate:o_width + framerate:umem + i:i_size + i:p_size + 
    i:o_codech264 + i:o_codecmpeg4 + i:o_codecvp8 + i:o_bitrate + 
    i:umem + i_size:o_codech264 + i_size:o_codecmpeg4 + i_size:o_codecvp8 + 
    i_size:o_bitrate + i_size:o_width + i_size:umem + p_size:o_codecmpeg4 + 
    p_size:o_codecvp8 + p_size:o_bitrate + p_size:o_framerate + 
    p_size:o_width + p_size:umem + o_codech264:o_framerate + 
    o_codech264:o_width + o_codecmpeg4:o_bitrate + o_codecmpeg4:o_framerate + 
    o_codecmpeg4:o_width + o_codecmpeg4:umem + o_codecvp8:o_bitrate + 
    o_codecvp8:umem + o_bitrate:o_framerate + o_bitrate:umem + 
    o_framerate:o_width + o_framerate:umem + o_width:umem + codech264:codecmpeg4 + 
    codech264:codecvp8 + codecmpeg4:codecvp8 + o_codech264:o_codecmpeg4 + 
    o_codech264:o_codecvp8 + o_codecmpeg4:o_codecvp8
```
We obtain an improved RMSE of : `6.820561`
* Backward
Code can be found in Appendix. The formula is below:
```
utime ~ duration + codech264 + codecmpeg4 + bitrate + framerate + 
    i + i_size + p_size + o_codech264 + o_codecmpeg4 + o_codecvp8 + 
    o_bitrate + o_framerate + o_width + umem + duration:codecmpeg4 + 
    duration:framerate + duration:i_size + duration:p_size + 
    duration:o_codech264 + duration:o_codecvp8 + duration:o_bitrate + 
    duration:o_width + duration:umem + codech264:i + codech264:p_size + 
    codech264:o_codech264 + codech264:o_codecmpeg4 + codecmpeg4:framerate + 
    codecmpeg4:p_size + codecmpeg4:o_codech264 + codecmpeg4:o_codecmpeg4 + 
    codecmpeg4:o_codecvp8 + codecmpeg4:o_bitrate + codecmpeg4:o_width + 
    codecvp8:framerate + codecvp8:i + codecvp8:i_size + codecvp8:p_size + 
    codecvp8:o_codech264 + codecvp8:o_codecmpeg4 + codecvp8:o_codecvp8 + 
    codecvp8:o_bitrate + codecvp8:o_width + codecvp8:umem + bitrate:i + 
    bitrate:i_size + bitrate:p_size + bitrate:o_codech264 + bitrate:o_codecmpeg4 + 
    bitrate:o_bitrate + bitrate:o_framerate + bitrate:o_width + 
    bitrate:umem + framerate:p_size + framerate:o_codech264 + 
    framerate:o_codecmpeg4 + framerate:o_codecvp8 + framerate:o_bitrate + 
    framerate:o_width + framerate:umem + i:i_size + i:p_size + 
    i:o_codech264 + i:o_codecmpeg4 + i:o_codecvp8 + i:o_bitrate + 
    i:o_width + i:umem + i_size:o_codech264 + i_size:o_codecvp8 + 
    i_size:o_bitrate + i_size:umem + p_size:o_codecmpeg4 + p_size:o_codecvp8 + 
    p_size:o_bitrate + p_size:o_width + p_size:umem + o_codech264:o_framerate + o_codecmpeg4:o_framerate + o_codecmpeg4:o_width + o_codecmpeg4:umem + 
    o_codecvp8:o_bitrate + o_codecvp8:umem + o_bitrate:o_framerate + 
    o_bitrate:umem + o_framerate:o_width + o_framerate:umem + 
    o_width:umem + codech264:codecmpeg4 + codech264:codecvp8 + 
    codecmpeg4:codecvp8 + o_codech264:o_codecmpeg4 + o_codech264:o_codecvp8 + 
    o_codecmpeg4:o_codecvp8
```
We obtain an RMSE of: `7.088572`
It is notable that for the backward and forward models, there are some linear dependencies in the newly introduced variables. A few more passes of pruning collinearity and rerunning BIC might have yielded better results. Moreover, although not shown here, using a log link function with the current predictor set did not improve RMSE.
- Lasso Regression. Below is the code for lasso regression. Using `cv.glmnet` we automate the cross validation in order to retrieve the lambda that minimizes RMSE. Below is the code.
```r
## Lasso regression model
pre_processed_train_matrix = model.matrix(utime~.,pre_processed_train)

cvfit = cv.glmnet(alpha=1, pre_processed_train_matrix, y=pre_processed_train[,"utime"])
coef(cvfit, s = "lambda.min")
cvfit$lambda.min

pre_processed_test_matrix = model.matrix(utime~.,pre_processed_test)

model_lasso_predict = predict(cvfit, newx = pre_processed_test_matrix, s = "lambda.min")
model_lasso_rmse = sqrt(mean((transcoding_data_test$utime - model_lasso_predict)^2))
model_lasso_rmse
```
Using this could the following coefficients and `lambda.min` were obtained.
```
> coef(cvfit, s = "lambda.min")
18 x 1 sparse Matrix of class "dgCMatrix"
                      1
(Intercept)   9.9673093
(Intercept)   .        
duration      0.5361586
codech264     0.1549945
codecmpeg4   -0.0104022
codecvp8     -0.1202084
bitrate       1.8959802
framerate     0.5498684
i            -0.6050087
i_size       -0.3375222
p_size        0.4234441
o_codech264   5.1885559
o_codecmpeg4  1.1581159
o_codecvp8    3.7077789
o_bitrate     2.3726768
o_framerate   1.6383689
o_width       5.7791001
umem          6.7447330
> cvfit$lambda.min
[1] 0.01190226
```
And running the model itself, we obtained an RMSE of: 9.677331
- Ridge regression. We have a similar model to that for Lasso. `glmnet` allows us to simply change `alpha` and obtain the ridge regression model.
```r
pre_processed_train_matrix = model.matrix(utime~.,pre_processed_train)

cvfit = cv.glmnet(alpha=0, pre_processed_train_matrix, y=pre_processed_train[,"utime"])
coef(cvfit, s = "lambda.min")
cvfit$lambda.min

pre_processed_test_matrix = model.matrix(utime~.,pre_processed_test)

model_ridge_predict = predict(cvfit, newx = pre_processed_test_matrix, s = "lambda.min")
model_ridge_rmse = sqrt(mean((transcoding_data_test$utime - model_ridge_predict)^2))
model_ridge_rmse
```
Below are the coefficients and `lambda.min`
```
> coef(cvfit, s = "lambda.min")
18 x 1 sparse Matrix of class "dgCMatrix"
                       1
(Intercept)   9.96730928
(Intercept)   .         
duration      0.38251257
codech264     0.16393902
codecmpeg4   -0.08592764
codecvp8     -0.10322271
bitrate       1.70693664
framerate     0.45758273
i            -0.49579469
i_size       -0.29012000
p_size        0.51496016
o_codech264   4.61601048
o_codecmpeg4  0.69290961
o_codecvp8    3.10620608
o_bitrate     2.22229439
o_framerate   1.53901155
o_width       5.48743349
umem          6.48613781
> cvfit$lambda.min
[1] 1.162862
```
And, finally, the RMSE: `9.726458`

In summary, we obtained the following RMSE values for the six variations of the linear models.
```
> model_linear_all_rmse
[1] 9.675779
> model_linear_bic_all_rmse
[1] 9.675355
> model_linear_bic_forward_rmse
[1] 6.820561
> model_linear_bic_backward_rmse
[1] 7.088572
> model_lasso_rmse
[1] 9.677331
> model_ridge_rmse
[1] 9.726458
```
Forward and backward BIC performed the best, and could probably have performed substantially better since there were many linear dependencies amongst the predictors used by each. Also, quite surprisingly, they each had ~100 predictors compared to ~20 or less predictors for the approaches and still outperformed them.

## Appendix containing complete code
```r
## declare libraries
library("glmnet", lib.loc="/usr/local/lib/R/3.3/site-library")
library("leaps", lib.loc="/usr/local/lib/R/3.3/site-library")
library("caret", lib.loc="/usr/local/Cellar/r/3.3.2/R.framework/Versions/3.3/Resources/library")

## Clear environment variables
rm(list=ls())  

## Set working directory for assignment2 dataset
setwd("~/Dropbox/SDS385/assignment2/online_video_dataset")

## Get dataset
transcoding_data = read.table("transcoding_mesurment.tsv", header = TRUE)

## Split data set in half
set.seed(62)
partition = createDataPartition(transcoding_data$utime, p = 0.5, list = FALSE) # partition the data into two sets in a balanced fashion
transcoding_data_train = transcoding_data[partition,]
transcoding_data_test = transcoding_data[-partition,]

preProcessCustom<- function(dataset_input)
{
  ## Remove id and constant variables
  cur_data = dataset_input
  cur_data$id = NULL # not needed
  cur_data$b_size = NULL # constant variable
  
  ## Deal with categorical variables
  matrix = model.matrix(~.,subset(cur_data,select = -utime))
  
  ## Remove zero and near zero variance predictors
  nzv = nearZeroVar(matrix)
  matrix_nzv = matrix[,-nzv]
  
  ## Find correlation in dataset
  matrix_cor = cor(matrix_nzv)
  
  # print distribution of correlated variables
  matrix_cor_uppertri = matrix_cor[upper.tri(matrix_cor)]
  summary(matrix_cor_uppertri)
  
  # find highly correlated predictors
  matrix_highly_cor = findCorrelation(matrix_cor, cutoff = 0.8)
  matrix_highly_cor
  
  # print info regarding highly correlated predictors
  findCorrelation(matrix_cor, cutoff = 0.8, names=TRUE, verbose=TRUE)
  
  # printing column names for specific rows that have been identified
  for (i in 1:dim(matrix_cor)[2]) {
    cat(colnames(matrix_cor)[i]," ",i, "\n")
  }
  
  # update matrix, removing correlated predictors
  matrix_nocor = matrix_nzv[,-matrix_highly_cor]
  
  # center and scale the predictors
  params = preProcess(matrix_nocor, method = c("center", "scale"))
  matrix_processed = predict(params, matrix_nocor)
  
  # rebind the response variable to the dataset
  dataset_processed = cbind(as.data.frame(matrix_processed),cur_data["utime"])
  return(dataset_processed)
} 

pre_processed_train = preProcessCustom(transcoding_data_train)
pre_processed_test = preProcessCustom(transcoding_data_test)

mean(pre_processed_train$utime)
mean(pre_processed_test$utime)
#test = subset(pre_processed_train, select = -utime)
#findLinearCombos(test)

## Linear regression with all predictors
#model_linear_all = lm(utime ~ (.), data=as.data.frame(pre_processed_train))
model_linear_all = lm(utime ~ (.), data=as.data.frame(pre_processed_train))
model_linear_all_predict = predict(model_linear_all, newdata=as.data.frame(pre_processed_test))
model_linear_all_rmse = sqrt(mean((transcoding_data_test$utime - model_linear_all_predict)^2))
model_linear_all_rmse

## Linear regression using BIC
all_ss = regsubsets(utime ~ (.), data = pre_processed_train, nvmax = 400)
forward_ss = regsubsets(utime ~ (.)^2, data = pre_processed_train, nvmax = 400, method = "forward")
backward_ss = regsubsets(utime ~ (.)^2, data = pre_processed_train, nvmax = 400, method = "backward")

# exhaustive BIC
all_bic = summary(all_ss)$bic
u1               = toString(names(coef(all_ss, which.min(all_bic)))[-1])
u2               = gsub(pattern = ", ",  replacement = " + ", x = toString(u1))
all_formula  = as.formula(paste("utime ~ ", u2, sep = ""))
all_formula

model_linear_bic_all = lm(all_formula, data=as.data.frame(pre_processed_train))
model_linear_bic_all_predict = predict(model_linear_bic_all, newdata=as.data.frame(pre_processed_test))
model_linear_bic_all_rmse = sqrt(mean((transcoding_data_test$utime - model_linear_bic_all_predict)^2))
model_linear_bic_all_rmse

# forward BIC
forward_bic = summary(forward_ss)$bic
u1               = toString(names(coef(forward_ss, which.min(forward_bic)))[-1])
u2               = gsub(pattern = ", ",  replacement = " + ", x = toString(u1))
forward_formula  = as.formula(paste("utime ~ ", u2, sep = ""))
forward_formula
summary(model_linear_bic_forward)
model_linear_bic_forward = glm(forward_formula, data=as.data.frame(pre_processed_train))
model_linear_bic_forward_predict = predict(model_linear_bic_forward, newdata=as.data.frame(pre_processed_test))
model_linear_bic_forward_rmse = sqrt(mean((transcoding_data_test$utime - model_linear_bic_forward_predict)^2))
model_linear_bic_forward_rmse

# backward BIC
backward_bic = summary(backward_ss)$bic
u1               = toString(names(coef(backward_ss, which.min(backward_bic)))[-1])
u2               = gsub(pattern = ", ",  replacement = " + ", x = toString(u1))
backward_formula  = as.formula(paste("utime ~ ", u2, sep = ""))
backward_formula

model_linear_bic_backward = lm(backward_formula, data=as.data.frame(pre_processed_train))
model_linear_bic_backward_predict = predict(model_linear_bic_backward, newdata=as.data.frame(pre_processed_test))
model_linear_bic_backward_rmse = sqrt(mean((transcoding_data_test$utime - model_linear_bic_backward_predict)^2))
model_linear_bic_backward_rmse

## Lasso regression model
pre_processed_train_matrix = model.matrix(utime~.,pre_processed_train)

cvfit = cv.glmnet(alpha=1, pre_processed_train_matrix, y=pre_processed_train[,"utime"])
coef(cvfit, s = "lambda.min")
cvfit$lambda.min

pre_processed_test_matrix = model.matrix(utime~.,pre_processed_test)

model_lasso_predict = predict(cvfit, newx = pre_processed_test_matrix, s = "lambda.min")
model_lasso_rmse = sqrt(mean((transcoding_data_test$utime - model_lasso_predict)^2))
model_lasso_rmse

## Ridge regression model
pre_processed_train_matrix = model.matrix(utime~.,pre_processed_train)

cvfit = cv.glmnet(alpha=0, pre_processed_train_matrix, y=pre_processed_train[,"utime"])
coef(cvfit, s = "lambda.min")
cvfit$lambda.min

pre_processed_test_matrix = model.matrix(utime~.,pre_processed_test)

model_ridge_predict = predict(cvfit, newx = pre_processed_test_matrix, s = "lambda.min")
model_ridge_rmse = sqrt(mean((transcoding_data_test$utime - model_ridge_predict)^2))
model_ridge_rmse

model_linear_all_rmse
model_linear_bic_all_rmse
model_linear_bic_forward_rmse
model_linear_bic_backward_rmse
model_lasso_rmse
model_ridge_rmse

```
