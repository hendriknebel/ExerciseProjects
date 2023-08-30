# Title: CV 0.01 LB 0.12 Kernel for the House Prices Dataset
# Author: Hendrik Nebel
# Date: August 20, 2023

### TOC - Table of Contents -------------------------------------------
#
# Libraries
# Data Loading
# Data Preparation
# Renaming Variables
# New Features
# Recipe
# Residual Mean Squared Log Error (rmsle - Evaluation Metric)
# Predictive Modeling
# Ensembling
# Submission File
# Sources
#
# ---------------------------------------------------------------------

### Libraries ---------------------------------------------------------
require(data.table)
require(skimr)
require(plyr)
require(caret)
require(caretEnsemble)
require(xgboost)
require(kernlab)
require(Matrix)
require(recipes)
require(randomForest)
# ---------------------------------------------------------------------

### Data Loading ------------------------------------------------------
data_train_loaded <- fread("../input/house-prices-advanced-regression-techniques/train.csv")      # read Train
data_test_loaded <- fread("../input/house-prices-advanced-regression-techniques/test.csv")        # read Test
data_test_loaded[["SalePrice"]] <- rep(NA, nrow(data_test_loaded))                                # add empty SalePrice to Test Set
data_entire <- rbind(data_train_loaded, data_test_loaded)                                         # combine Train & Test to huge DF
remove(data_train_loaded); remove(data_test_loaded)                                               # rm(Train), rm(Test)
# ---------------------------------------------------------------------

### Data Preparation --------------------------------------------------
# Get Columns Types in Vector
vec_type <- sapply(data_entire, typeof)                                                 # get Type
# Select Factor Variables
col.fac <- names(vec_type)[vec_type == "character"]                                     # chr -> factor
col.fac <- c(col.fac, "MSSubClass")                                                     # append MSSubClass
# Select Numeric Variables
col.num <- names(vec_type)[vec_type == "integer"]                                       # num
col.num <- col.num[c(-1,-2)]                                        ##exclude Id, and MSSubClass from numeric variable list

# Convert Characters -> Factors
for(col in col.fac){
    data_entire[[col]] <-  factor(data_entire[[col]])
}

options("max.print" = 100000)                                                           # limit the printed amount

skim_with(integer = list(complete = NULL,
                         n = NULL,
                         sd = NULL,
                         p25 = NULL,
                         p75 = NULL),
          factor = list(ordered = NULL))

skim(data_entire)
# ---------------------------------------------------------------------

### Renaming Variables ------------------------------------------------
names(data_entire)[which(names(data_entire) == "1stFlrSF")] <- 'FirstFlrSF'         # rename 1stFlrSF to R-readable
names(data_entire)[which(names(data_entire) == "2ndFlrSF")] <- 'SecondFlrSF'        # rename 2ndFlrSF to R-readable
names(data_entire)[which(names(data_entire) == "3SsnPorch")] <- 'ThreeSsnPorch'     # rename 3SsnPorch to R-readable

data_entire$RoofMatl <- revalue(data_entire$RoofMatl, c('Tar&Grv' = 'TarGrv'))                # rename factor lvls
data_entire$Exterior1st <- revalue(data_entire$Exterior1st, c('Wd Sdng' = 'WdSdng'))          # rename factor lvls
data_entire$Exterior2nd <- revalue(data_entire$Exterior2nd,
                                   c('Brk Cmn' = 'BrkComm', 'CmentBd' = 'CemntBd',
                                     'Wd Sdng' = 'WdSdng', 'Wd Shng' = 'WdShing'))            # rename factor lvls

data_entire[GarageYrBlt > 2010, GarageYrBlt := NA]                                            # impute NA for obvious erroneous value of year

summary(data_entire$GarageYrBlt)
# ---------------------------------------------------------------------

### New Features ------------------------------------------------------
data_entire[["hasBsmt"]] <- as.numeric(data_entire$TotalBsmtSF > 0)
data_entire[["has2ndFloor"]] <- as.numeric(data_entire$SecondFlrSF > 0)
data_entire[["hasPool"]] <- as.numeric(data_entire$PoolArea > 0)
data_entire[["hasPorch"]] <- as.numeric((data_entire$OpenPorchSF + data_entire$EnclosedPorch + data_entire$ThreeSsnPorch + data_entire$ScreenPorch) > 0)
data_entire[["hasRemod"]] <- as.numeric(data_entire$YearRemodAdd != data_entire$YearBuilt)
data_entire[["hasFireplace"]] <- as.numeric(data_entire$Fireplaces > 0)
data_entire[["isNew"]] <- as.numeric(data_entire$YrSold == data_entire$YearBuilt)
data_entire[["totalSF"]] <- as.numeric(data_entire$GrLivArea + data_entire$TotalBsmtSF)
data_entire[["totalBath"]] <- as.numeric(data_entire$FullBath + 0.5 * data_entire$HalfBath + 0.5 * data_entire$BsmtHalfBath + data_entire$BsmtFullBath)
data_entire[["TimeSinceRemod"]] <- as.numeric(data_entire$YrSold - data_entire$YearRemodAdd)
data_entire[["DateSold"]] <- as.numeric(paste0(as.character(data_entire$YrSold), as.character(data_entire$MoSold)))
data_entire[["totalPorchSF"]] <- as.numeric(data_entire$OpenPorchSF + data_entire$EnclosedPorch + data_entire$ThreeSsnPorch + data_entire$ScreenPorch)
data_entire[["bsmtBath"]] <- as.numeric((data_entire$BsmtHalfBath + data_entire$BsmtFullBath) > 0)
data_entire$bsmtUnf <- as.numeric(data_entire$TotalBsmtSF == data_entire$BsmtUnfSF)
# ---------------------------------------------------------------------


### Using Recipe ------------------------------------------------------
data_select_train <- data_entire[!is.na(SalePrice)]                                     # get Train
data_select_test <- data_entire[is.na(SalePrice)]                                       # get Test
set.seed(1)
rec_obj <- recipe(SalePrice ~ ., data = data_select_train) %>%                          # rec
  update_role(Id, new_role = "id var") %>%
  step_impute_knn(all_predictors()) %>%                                                 # impute all numericals via knn
  step_dummy(all_predictors(), -all_numeric()) %>%                                      # create factor dummies
  step_BoxCox(all_predictors()) %>%                                                     # BoxCox
  step_center(all_predictors())  %>%                                                    # center
  step_scale(all_predictors()) %>%                                                      # scale
  step_zv(all_predictors()) %>%                                                         # zv (Zero Values)
  step_corr(all_predictors(), threshold = .9) %>%                                       # corr
  step_log(all_outcomes()) %>%                                                          # log(outcomes)
  check_missing(all_predictors())                                                       # check missing 

rec_obj
trained_rec <- prep(rec_obj, training = data_select_train)                              # get trained rec
data_train <- bake(trained_rec, new_data = data_select_train)                           # obj train
data_test <- bake(trained_rec, new_data = data_select_test)                             # obj test
# ---------------------------------------------------------------------

### Define Residual Mean Squared Log Error (here) ---------------------
# rmsle function
rmsle_here <- function(observed, predicted) {
  n <- length(observed)
  # print(predicted)
  # Compute squared errors
  squared_errors <- (log(observed+1) - log(predicted+1)) ^ 2
  # Compute mean squared error
  mse <- sum(squared_errors) / n
  # Compute root mean squared error
  rmse <- sqrt(mse)
  # print(log_rmse)
  return(rmse)
}

# rmsle_summary function - required by the caret package
rmsle_summary <- function(data, lev, model) {
  # Extract predicted class probabilities or predictions
  predictions <- as.numeric(data$pred)
  
  # Extract the actual outcomes
  actual <- as.numeric(data$obs)

  rmsle_here <- rmsle_here(actual, predictions)

  c(rmsle_here = rmsle_here)
}
# ---------------------------------------------------------------------


### Predictive Modeling -----------------------------------------------
## Define my_control --------------------------------------------------
my_control <- trainControl(method = 'cv',
                           number = 10,
                           summaryFunction = rmsle_summary)
# ---------------------------------------------------------------------

# ### Ridge Modeling ..................................................
# # tune_grid <- expand.grid(lambda = seq(0.1, 1, 0.1))
# # tune_grid <- expand.grid(lambda = seq(0.01, 0.2, 0.01))
# tune_grid <- expand.grid(lambda = 0.1)

# set.seed(1)
# ridge_cv <- train(y = data_train$SalePrice,
#                   x = subset(data_train, select=-c(Id, SalePrice)),
#                   method = "ridge",
#                   metric = "rmsle_here",
#                   maximize = F,
#                   trControl = my_control,
#                   tuneGrid = tune_grid)

# # Print the results
# print(ridge_cv)
# print(ridge_cv$resample)
# message("CV Mean: ", round(mean(ridge_cv$resample[[1]]), digits = 4))
# message("CV SD: ", round(sd(ridge_cv$resample[[1]]), digits = 4))
# # ...................................................................
# ### Lasso Modeling ..................................................
# # tune_grid <- expand.grid(fraction = seq(0.1, 1, 0.1))
# # tune_grid <- expand.grid(fraction = seq(0.7, 0.9, 0.01))
# tune_grid <- expand.grid(fraction = 0.1)

# set.seed(1)
# lasso_cv <- train(y = data_train$SalePrice,
#                   x = subset(data_train, select=-c(Id, SalePrice)),
#                   method = "lasso",
#                   metric = "rmsle_here",
#                   maximize = F,
#                   trControl = my_control,
#                   tuneGrid = tune_grid)

# # Print the results
# print(lasso_cv)
# print(lasso_cv$resample)
# message("CV Mean: ", round(mean(lasso_cv$resample[[1]]), digits = 4))
# message("CV SD: ", round(sd(lasso_cv$resample[[1]]), digits = 4))
# # ...................................................................
# ### Random Forest Modeling ..........................................
# # tune_grid <- expand.grid(.mtry = seq(2, 12, 1))
# tune_grid <- expand.grid(.mtry = 12)

# # Best
# # mtry = 4, 5, 12

# set.seed(1)
# random_forest_cv <- train(y = data_train$SalePrice,
#                           x = subset(data_train, select=-c(Id, SalePrice)),
#                           method = 'rf',
#                           metric = "rmsle_here",
#                           maximize = F,
#                           trControl = my_control,
#                           tuneGrid = tune_grid,
#                           ntree = 200,
#                           importance = TRUE)

# print(random_forest_cv)
# print(random_forest_cv$resample)
# message("CV Mean: ", round(mean(random_forest_cv$resample[[1]]), digits = 4))
# message("CV SD: ", round(sd(random_forest_cv$resample[[1]]), digits = 4))

# vec_feature_importance <- importance(random_forest_cv$finalModel)[,1]
# # print(vec_feature_importance)
# for(i in 1:length(vec_feature_importance)){
#     message(round(vec_feature_importance[i]/sum(vec_feature_importance), digits = 4), ": ", names(vec_feature_importance)[i])
# }
# # ...................................................................
# ### SVM Linear Kernel Modeling ......................................
# # tune_grid <- expand.grid(list(C = c(0.1, 0.5, 1, 5, 10)))
# # tune_grid <- expand.grid(C = seq(0.0001, 0.003, 0.0001))
# tune_grid <- expand.grid(C = 0.0016)
# # 0.0016 -> 0.002
# set.seed(1)
# svmlin_cv <- train(y = data_train$SalePrice,
#                    x = subset(data_train, select=-c(Id, SalePrice)),
#                    method = "svmLinear",
#                    metric = "rmsle_here",
#                    maximize = F,
#                    trControl = my_control,
#                    tuneGrid = tune_grid)

# print(svmlin_cv)
# print(svmlin_cv$resample)
# message("CV Mean: ", round(mean(svmlin_cv$resample[[1]]), digits = 4))
# message("CV SD: ", round(sd(svmlin_cv$resample[[1]]), digits = 4))
# # ...................................................................
# ### SVM Radial Kernel Modeling ......................................
# # tune_grid <- expand.grid(list(C = c(0.1, 0.5, 1, 5, 10),
# #                               sigma = c(0.1, 0.5, 1, 5, 10)))
# # tune_grid <- expand.grid(list(C = seq(0.5, 4, 0.5),
# #                               sigma = seq(0.05, 0.3, 0.05)))
# tune_grid <- expand.grid(C = 2.5, sigma = 0.05)
# # C = 1;    sigma = 0.1
# # C = 2.5;  sigma = 0.05
# set.seed(1)
# svmrad_cv <- train(y = data_train$SalePrice,
#                    x = subset(data_train, select=-c(Id, SalePrice)),
#                    method = "svmRadial",
#                    metric = "rmsle_here",
#                    maximize = F,
#                    trControl = my_control,
#                    tuneGrid = tune_grid)

# print(svmrad_cv)
# print(svmrad_cv$resample)
# message("CV Mean: ", round(mean(svmrad_cv$resample[[1]]), digits = 4))
# message("CV SD: ", round(sd(svmrad_cv$resample[[1]]), digits = 4))
# # ...................................................................
# ### GBM Modeling ....................................................
# # tune_grid <- expand.grid(.n.trees = c(50, 100, 150),
# #                          .interaction.depth = c(3, 5, 7, 9),
# #                          .shrinkage = c(0.001, 0.01, 0.1, 1, 10),
# #                          .n.minobsinnode = c(30, 35, 40))
# tune_grid <- expand.grid(n.trees = c(150),
#                          interaction.depth = c(5),
#                          shrinkage = c(0.1),
#                          n.minobsinnode = c(35))

# set.seed(1)
# gbm_cv <- train(y = data_train$SalePrice,
#                 x = subset(data_train, select=-c(Id, SalePrice)),
#                 method = "gbm",
#                 metric = "rmsle_here",
#                 maximize = F,
#                 trControl = my_control,
#                 tuneGrid = tune_grid)

# print(gbm_cv)
# print(gbm_cv$resample)
# message("CV Mean: ", round(mean(gbm_cv$resample[[1]]), digits = 4))
# message("CV SD: ", round(sd(gbm_cv$resample[[1]]), digits = 4))
# # ...................................................................
# ### GLMNET Modeling .................................................
# # tune_grid <- expand.grid(alpha = 1,
# #                          lambda = seq(0.001, 0.1, by = 0.001))
# tune_grid <- expand.grid(alpha = 1,
#                          lambda = 0.004)

# set.seed(1)
# glmnet_cv <- train(y = data_train$SalePrice,
#                    x = subset(data_train, select=-c(Id, SalePrice)),
#                    method = "glmnet",
#                    metric = "rmsle_here",
#                    maximize = F,
#                    trControl = my_control,
#                    tuneGrid = tune_grid)

# print(glmnet_cv)
# print(glmnet_cv$resample)
# message("CV Mean: ", round(mean(glmnet_cv$resample[[1]]), digits = 4))
# message("CV SD: ", round(sd(glmnet_cv$resample[[1]]), digits = 4))
# # ...................................................................
# ### XGBM Modeling ...................................................
# tune_grid <- expand.grid(nrounds = 200,
#                          max_depth = 25,
#                          eta = 0.1,
#                          gamma = 2,
#                          colsample_bytree = 0.4,
#                          min_child_weight = 4,
#                          subsample = 0.63)

# set.seed(1)
# xgbm_cv <- train(y = data_train$SalePrice,
#                  x = subset(data_train, select=-c(Id, SalePrice)),
#                  method = "xgbTree",
#                  metric = "rmsle_here",
#                  maximize = F,
#                  trControl = my_control,
#                  tuneGrid = tune_grid)

# print(xgbm_cv)
# print(xgbm_cv$resample)
# message("CV Mean: ", round(mean(xgbm_cv$resample[[1]]), digits = 4))
# message("CV SD: ", round(sd(xgbm_cv$resample[[1]]), digits = 4))
# # ...................................................................
# ---------------------------------------------------------------------



### Ensembling ---------------------------------------------------------
## Define my_ens_control ..............................................
my_ens_control <- trainControl(method ='cv',                                     
                               savePredictions = "final",                        
                               index = createFolds(data_train$SalePrice, k = 10, returnTrain = TRUE),   # Define Index: ifelse, each model would have its own data for Cross-Validation
                               allowParallel = TRUE,                                                    
                               verboseIter = TRUE,                                                      
                               summaryFunction = rmsle_summary)
# .....................................................................

# Define Grids ........................................................
ridgeGrid <- expand.grid(lambda = 0.1)

lassoGrid <- expand.grid(fraction = 0.1)

randomForestGrid <- expand.grid(mtry = 12)

svmlinGrid <- expand.grid(C = 0.0016)

svmradGrid <- expand.grid(C = 2.5, sigma = 0.05)

gbmGrid <- expand.grid(n.trees = c(150),
                       interaction.depth = c(5),
                       shrinkage = c(0.1),
                       n.minobsinnode = c(35))

glmnetGrid <- expand.grid(alpha = 1,
                          lambda = seq(0.001,0.01,by = 0.001))              # Define GBM Grid

xgbTreeGrid <- expand.grid(nrounds = 200,
                           max_depth = 25,
                           eta = 0.1,
                           gamma = 2,
                           colsample_bytree = 0.4,
                           subsample = 0.63,
                           min_child_weight = 4)                            # Define XGBM Params
# .....................................................................
### tune_list .........................................................
tune_list <- list(Ridge = caretModelSpec(method = "ridge", tuneGrid = ridgeGrid, preProcess = c("nzv", "pca")),
                  Lasso = caretModelSpec(method = "lasso", tuneGrid = lassoGrid, preProcess = c("nzv", "pca")),
                  RF = caretModelSpec(method = "rf", tuneGrid = randomForestGrid),
                  SVMlin = caretModelSpec(method = "svmLinear", tuneGrid = svmlinGrid, preProcess = c("nzv", "pca")),
                  SVMrad = caretModelSpec(method = "svmRadial", tuneGrid = svmradGrid, preProcess = c("nzv", "pca")),
                  GBM = caretModelSpec(method = "gbm", tuneGrid = gbmGrid),
                  GLMNET = caretModelSpec(method = "glmnet", tuneGrid = glmnetGrid),
                  XGBM = caretModelSpec(method = "xgbTree", tuneGrid = xgbTreeGrid))
# .....................................................................
### Run the Models ....................................................
set.seed(1)
modelList <<- caretList(x = subset(data_train, select = -c(Id, SalePrice)),             # Define CaretList (list(models))
                        y = data_train$SalePrice,                                       # ...incl, X & y
                        trControl = my_ens_control,                                     # trControl = my_control
                        metric = "rmsle_here",                                          # metric = 'rmsle_here'
                        tuneList = tune_list)

### Stacking Model ....................................................
set.seed(1)
greedyEnsemble <- caretEnsemble(modelList,                                      ### equals caretStack (method='glm')
                                metric = "rmsle_here",
                                trControl = trainControl(number = 10,
                                                         method = "repeatedcv",
                                                         repeats = 3,
                                                         summaryFunction = rmsle_summary))
summary(greedyEnsemble)
# .....................................................................
# ---------------------------------------------------------------------



### Submission --------------------------------------------------------
data_preds_target <- predict(greedyEnsemble,
                             newdata = subset(data_test, select = -c(Id, SalePrice)))

data_submission <- data.frame('Id' = data_entire[is.na(SalePrice), 1],
                              'SalePrice' = exp(data_preds_target))

write.csv(data_submission,
          file = "submission.csv",
          row.names = F)
# ---------------------------------------------------------------------

### Sources -----------------------------------------------------------
# Zong Tseng - recipes,caret,caretensemble (xgboost, lasso, svm)
#   - Great EDA, esp. on Outliers
#   - Great Pre-Processing using Recipes
#   - Great Model Discussion, esp. Ensembling via 'caret' Package
#   https://www.kaggle.com/code/zongtseng/recipes-caret-caretensemble-xgboost-lasso-svm
# ---------------------------------------------------------------------

# --------- THANKS FOR READING! Comments & Recommendations appreciated! --------
# ----- Don't forget to upvote, if you like the kernel. -----
