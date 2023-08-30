# Title: CV 0.82 LB 0.79 Kernel for the Titanic Dataset
# Author: Hendrik Nebel
# Date: August 20, 2023

### TOC - Table of Contents ---------------------------------------------------
#
# Libraries
# Data Sets
# Pre-Processing Training Set
# Pre-Processing Test Set
# Predictive Modeling
# Ensembler
# Submission File
# Sources
#
# -----------------------------------------------------------------------------

### Libraries -----------------------------------------------------------------
library(ggplot2)        # for Visualization
library(gridExtra)      # for arranging stuff
library(randomForest)   # for creating Random Forests
library(class)          # for creating KNN
library(MASS)           # for LDA, QDA
library(e1071)          # Prediction: SVM, Naive Bayes, Parameter Tuning
library(caret)          # for k-fold-Cross-Validation
library(boot)           # for CV + Bootstrapping
library(readr)          # for reading Data - CSV, etc.
library(caTools)        # for Validation
# ---------------------------------------------------------------------

#### Data Sets --------------------------------------------------------
data_train <- read_csv('../input/titanic/train.csv')
data_test <- read_csv('../input/titanic/test.csv')
# ---------------------------------------------------------------------



### Pre-Processing - Training Set
##### set_VariableType ------------------------------------------------
data_train[['PassengerId']] <- as.numeric(data_train[['PassengerId']])
data_train[['Survived']] <- as.factor(data_train[['Survived']])
data_train[['Pclass']] <- as.factor(data_train[['Pclass']])
data_train[['Name']] <- as.character(data_train[['Name']])
data_train[['Sex']] <- as.factor(data_train[['Sex']])
data_train[['Age']] <- as.numeric(data_train[['Age']])
data_train[['SibSp']] <- as.factor(data_train[['SibSp']])
data_train[['Parch']] <- as.factor(data_train[['Parch']])
data_train[['Ticket']] <- as.character(data_train[['Ticket']])
data_train[['Fare']] <- as.numeric(data_train[['Fare']])
data_train[['Cabin']] <- as.character(data_train[['Cabin']])
data_train[['Embarked']] <- as.factor(data_train[['Embarked']])
# ---------------------------------------------------------------------
### create_GroupsOrdinallyScaled --------------------------------------
# Age_Group ...........................................................
input_dataframe_02 <- 'data_train'
old_variable_name <- 'Age'
                                                    
new_variable_values <- numeric(nrow(data_train))
for(i in 1:nrow(data_train)){
    for(j in 1:4){
        if(is.na(get(input_dataframe_02)[[old_variable_name]][i])){
            new_variable_values[i] <- 'is.na'
            break
        }
        if(get(input_dataframe_02)[["Age"]][i] <= 16){
            new_variable_values[i] <- 'Young'
            break
        }
        if(get(input_dataframe_02)[["Age"]][i] <= 60){
            new_variable_values[i] <- 'Mid-Age'
            break
        }
        if(get(input_dataframe_02)[["Age"]][i] > 60){
            new_variable_values[i] <- 'Old'
            break
        }
    }
}
data_train[['Age_Group']] <- new_variable_values
rm(input_dataframe_02, old_variable_name)
# .....................................................................
# Avail_CabinNr .......................................................
data_train$Avail_CabinNr <- character(length = nrow(data_train))

for(i in 1:nrow(data_train)){
    if(is.na(data_train$Cabin[i])){
        data_train$Avail_CabinNr[i] <- "yes"
    } else{
        data_train$Avail_CabinNr[i] <- "no"
    }
}

data_train$Avail_CabinNr <- as.factor(data_train$Avail_CabinNr)
# .....................................................................
# Alone ...............................................................
data_train$Alone <- character(length = nrow(data_train))

for(i in 1:nrow(data_train)){
    if( (data_train$SibSp[i] == 0) & (data_train$Parch[i] == 0)){
        data_train$Alone[i] <- "Alone"
    } else{
        data_train$Alone[i] <- "Together"
    }
}
data_train$Alone <- as.factor(data_train$Alone)
# .....................................................................
# Titles ..............................................................
# Grab the title of each passenger
new_variable_name <- "Passenger_Title"
name_string_variable <- "Name"
new_variable_values <- gsub("^.*, (.*?)\\..*$", "\\1",
                            data_train[[name_string_variable]])
data_train[[new_variable_name]] <- new_variable_values

# Frequency of each Title by Sex
table(data_train[["Sex"]],
      data_train[["Passenger_Title"]])

### Create Title_Group
old_variable_name <- "Passenger_Title"
new_variable_name <- "Title_Group"
new_variable_values <- numeric(nrow(data_train))
data_train[[new_variable_name]] <- data_train[[old_variable_name]]
# assign Mlle -> Miss (french to engl equivalent)
# assign Ms -> Miss
data_train[[new_variable_name]][data_train[[old_variable_name]] == 'Mlle' | data_train[[old_variable_name]] == 'Ms'] <- 'Miss'
# assign Mme -> Mrs (french to engl equivalent)
data_train[[new_variable_name]][data_train[[old_variable_name]] == 'Mme'] <- 'Mrs'

# Some Titles occur only few times -> Category: Other
otherTitle <- c('Capt', 'Col', 'Don', 'Dona', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir', 'the Countess')
data_train[[new_variable_name]][data_train[[old_variable_name]] %in% otherTitle] <- 'OtherTitle' 

data_train$Title_Group <- factor(data_train$Title_Group)
# .....................................................................



### Pre-Processing - Test Set -----------------------------------------
# set_VariableType ----------------------------------------------------
data_test[['PassengerId']] <- as.numeric(data_test[['PassengerId']])
data_test[['Pclass']] <- as.factor(data_test[['Pclass']])
data_test[['Name']] <- as.character(data_test[['Name']])
data_test[['Sex']] <- as.factor(data_test[['Sex']])
data_test[['Age']] <- as.numeric(data_test[['Age']])
data_test[['SibSp']] <- as.factor(data_test[['SibSp']])
data_test[['Parch']] <- as.factor(data_test[['Parch']])
data_test[['Ticket']] <- as.character(data_test[['Ticket']])
data_test[['Fare']] <- as.numeric(data_test[['Fare']])
data_test[['Cabin']] <- as.character(data_test[['Cabin']])
data_test[['Embarked']] <- as.factor(data_test[['Embarked']])
# ---------------------------------------------------------------------

input_dataframe_02 <- 'data_test'
old_variable_name <- 'Age'
                                                    
new_variable_values <- numeric(nrow(data_test))
for(i in 1:nrow(data_test)){
    for(j in 1:4){
        if(is.na(get(input_dataframe_02)[[old_variable_name]][i])){
            new_variable_values[i] <- 'is.na'
            break
        }
        if(get(input_dataframe_02)[["Age"]][i] <= 16){
            new_variable_values[i] <- 'Young'
            break
        }
        if(get(input_dataframe_02)[["Age"]][i] <= 60){
            new_variable_values[i] <- 'Mid-Age'
            break
        }
        if(get(input_dataframe_02)[["Age"]][i] > 60){
            new_variable_values[i] <- 'Old'
            break
        }
    }
}
data_test[['Age_Group']] <- new_variable_values
rm(input_dataframe_02, old_variable_name)
# .....................................................................
# Avail_CabinNr .......................................................
data_test$Avail_CabinNr <- character(length = nrow(data_test))
for(i in 1:nrow(data_test)){
    if(is.na(data_test$Cabin[i])){
        data_test$Avail_CabinNr[i] <- "yes"
    } else{
        data_test$Avail_CabinNr[i] <- "no"
    }
}
data_test$Avail_CabinNr <- as.factor(data_test$Avail_CabinNr)
# .....................................................................
# Alone ...............................................................
data_test$Alone <- character(length = nrow(data_test))
for(i in 1:nrow(data_test)){
    if( (data_test$SibSp[i] == 0) & (data_test$Parch[i] == 0)){
        data_test$Alone[i] <- "Alone"
    } else{
        data_test$Alone[i] <- "Together"
    }
}

data_test$Alone <- as.factor(data_test$Alone)
data_test$Age_Group <- as.factor(data_test$Age_Group)
# .....................................................................
# Titles ..............................................................
# Grab the title of each passenger
new_variable_name <- "Passenger_Title"
name_string_variable <- "Name"
new_variable_values <- gsub("^.*, (.*?)\\..*$", "\\1",
                            data_test[[name_string_variable]])
data_test[[new_variable_name]] <- new_variable_values

# Frequency of each Title by Sex
table(data_test[["Sex"]],
      data_test[["Passenger_Title"]])

### Create Title_Group
old_variable_name <- "Passenger_Title"
new_variable_name <- "Title_Group"
new_variable_values <- numeric(nrow(data_test))
data_test[[new_variable_name]] <- data_test[[old_variable_name]]
# assign Mlle -> Miss (french to engl equivalent)
# assign Ms -> Miss
data_test[[new_variable_name]][data_test[[old_variable_name]] == 'Mlle' | data_test[[old_variable_name]] == 'Ms'] <- 'Miss'
# assign Mme -> Mrs (french to engl equivalent)
data_test[[new_variable_name]][data_test[[old_variable_name]] == 'Mme'] <- 'Mrs'

# Some Titles occur only few times -> Category: Other
otherTitle <- c('Capt', 'Col', 'Don', 'Dona', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir', 'the Countess')
data_test[[new_variable_name]][data_test[[old_variable_name]] %in% otherTitle] <- 'OtherTitle' 

data_test$Title_Group <- as.factor(data_test$Title_Group)
# .....................................................................



### Predictive Modeling -----------------------------------------------
# Define Response level names - "0", "1" can cause errors
levels(data_train$Survived) <- c("Deceased", "Survived")

# Create a Data Frame for Train prediction results (FITTED VALUES)
data_preds_train <- data.frame(Survived = data_train$Survived)

# Create a Data Frame for Test prediction results (PREDICTION VALUES)
data_preds_test <- data.frame(PassengerId = data_test$PassengerId)


### Define my_control -------------------------------------------------
my_control <- trainControl(method = 'cv',                       # Cross-Validation
                           number = 10,                         # Nr Folds
                           classProb = T)                       # get Class Probability
                           #sampling = "up")                    # try Oversampling Minority class("up"); Downsampling Majority Class ("down")
# ---------------------------------------------------------------------

# Define Vector with variables
vec_var_selection_08 <- c("Pclass", "Sex", "Age_Group", "Avail_CabinNr", "Alone", "Title_Group")

# Define Age_Group as factor
data_train$Age_Group <- as.factor(data_train$Age_Group)
data_test$Age_Group <- as.factor(data_test$Age_Group)

### LDA - Linear Discriminant Analysis Modeling .......................
set.seed(1)
lda_model <- train(as.formula(paste("Survived ~", paste(vec_var_selection_08, collapse = "+ "))),
                   data = data_train,                               # Dataset
                   method = "lda",                                  # Model
                   metric = "Accuracy",                             # Evaluation Metric
                   maximize = T,                                    # Eval Metric to maximize or to reduce
                   trControl = my_control)                          # Cross-Validation Control Parameters

# Print the results
print(lda_model)
print(lda_model$resample)
message("CV Mean: ", round(mean(lda_model$resample[[1]]), digits = 4)) 
message("CV SD: ", round(sd(lda_model$resample[[1]]), digits = 4))

# Get Prediction Values for the Train Set (FITTED VALUES) & Test Set (PREDICTION VALUES)
vec_empty <- predict(lda_model, data_train[, -which(names(data_train) == "Survived")], type = "prob")[[1]]
#print(vec_empty)

vec_empty_test <- predict(lda_model, data_test[, -which(names(data_test) == "PassengerId")], type = "prob")[[1]]
#print(vec_empty_test)

# Write LDA fitted values to the preds_train & preds_test
data_preds_train[["LDA"]] <- vec_empty
data_preds_test[["LDA"]] <- vec_empty_test
# .....................................................................

# Logit - Logistic Regression Model ...................................
set.seed(1)
log_reg_model <- train(as.formula(paste("Survived ~", paste(vec_var_selection_08, collapse = "+ "))),
                       data = data_train,                       
                       method = "glm",                          # Use Generalized Linear Model (i.e., Logistic Regression)
                       family = 'binomial',                     # Select Logit
                       metric = "Accuracy",                     
                       maximize = T,                            
                       trControl = my_control)                  

# View the summary of the model
print(log_reg_model)
print(log_reg_model$resample)
message("CV Mean: ", round(mean(log_reg_model$resample[[1]]), digits = 4)) 
message("CV SD: ", round(sd(log_reg_model$resample[[1]]), digits = 4))

# View the performance metrics
print(log_reg_model$results)

# Get Prediction Values for the Train Set (FITTED VALUES) & Test Set (PREDICTION VALUES)
vec_empty <- predict(log_reg_model, data_train[, -which(names(data_train) == "Survived")], type = "prob")[[1]]
#print(vec_empty)

vec_empty_test <- predict(log_reg_model, data_test[, -which(names(data_test) == "PassengerId")], type = "prob")[[1]]
#print(vec_empty_test)

# Write LDA fitted values to the preds_train & preds_test
data_preds_train[["Logit"]] <- vec_empty
data_preds_test[["Logit"]] <- vec_empty_test
# .....................................................................
### kNN - k-Nearest Neighbors Modeling ................................
set.seed(1)
knn_model <- train(as.formula(paste("Survived ~", paste(vec_var_selection_08, collapse = "+ "))),
                   data = data_train,
                   method = "knn",
                   metric = "Accuracy",
                   maximize = T,
                   trControl = my_control,
                   tuneGrid = expand.grid(k = 2))                       # k = 2

print(knn_model)
print(knn_model$resample)
message("CV Mean: ", round(mean(knn_model$resample[[1]]), digits = 4)) 
message("CV SD: ", round(sd(knn_model$resample[[1]]), digits = 4))

# Get Prediction Values for the Train Set (FITTED VALUES) & Test Set (PREDICTION VALUES)
vec_empty <- predict(knn_model, data_train[, -which(names(data_train) == "Survived")], type = "prob")[[1]]
#print(vec_empty)

vec_empty_test <- predict(knn_model, data_test[, -which(names(data_test) == "PassengerId")], type = "prob")[[1]]
#print(vec_empty_test)

# Write LDA fitted values to the preds_train & preds_test
data_preds_train[["kNN"]] <- vec_empty
data_preds_test[["kNN"]] <- vec_empty_test
# .....................................................................
# Random Forest Modeling ..............................................
set.seed(1)
random_forest_model <- train(as.formula(paste("Survived ~", paste(vec_var_selection_08, collapse = "+ "))),
                             data = data_train,
                             method = 'rf',
                             metric = "Accuracy",
                             maximize = T,
                             trControl = my_control,
                             tuneGrid = expand.grid(.mtry = 3),                 # Nr of Splits
                             ntree = 1000,                                      # Nr or Trees
                             importance = TRUE)

print(random_forest_model)
print(random_forest_model$resample)
message("CV Mean: ", round(mean(random_forest_model$resample[[1]]), digits = 4)) 
message("CV SD: ", round(sd(random_forest_model$resample[[1]]), digits = 4))

# Get Prediction Values for the Train Set (FITTED VALUES) & Test Set (PREDICTION VALUES)
vec_empty <- predict(random_forest_model, data_train[, -which(names(data_train) == "Survived")], type = "prob")[[1]]
#print(vec_empty)

vec_empty_test <- predict(random_forest_model, data_test[, -which(names(data_test) == "PassengerId")], type = "prob")[[1]]
#print(vec_empty_test)

# Write LDA fitted values to the preds_train & preds_test
data_preds_train[["RF"]] <- vec_empty
data_preds_test[["RF"]] <- vec_empty_test
# .....................................................................
# SVM - Support Vector Machine ........................................
set.seed(1)
svmrad_model <- train(as.formula(paste("Survived ~", paste(vec_var_selection_08, collapse = "+ "))),
                      data = data_train,
                      method = "svmRadial",
                      metric = "Accuracy",
                      maximize = T,
                      trControl = my_control,
                      tuneGrid = expand.grid(list(C = 1, sigma = 0.1)))

print(svmrad_model)
print(svmrad_model$resample)
message("CV Mean: ", round(mean(svmrad_model$resample[[1]]), digits = 4)) 
message("CV SD: ", round(sd(svmrad_model$resample[[1]]), digits = 4))

# Get Prediction Values for the Train Set (FITTED VALUES) & Test Set (PREDICTION VALUES)
vec_empty <- predict(svmrad_model, data_train[, -which(names(data_train) == "Survived")], type = "prob")[[1]]
#print(vec_empty)

vec_empty_test <- predict(svmrad_model, data_test[, -which(names(data_test) == "PassengerId")], type = "prob")[[1]]
#print(vec_empty_test)

# Write LDA fitted values to the preds_train & preds_test
data_preds_train[["SVM_lin"]] <- vec_empty
data_preds_test[["SVM_lin"]] <- vec_empty_test
# .....................................................................

# GBM - (GBoost [Gradient Boost]) Modeling ............................
# tune_grid <- expand.grid(.n.trees = c(50, 100, 150, 200, 250, 300),
#                          .interaction.depth = c(2, 3, 4),
#                          .shrinkage = c(0.01, 0.05, 0.1, 0.15, 0.2),
#                          .n.minobsinnode = c(30, 35, 40))

## Train the Gradient Boosting model with hyperparameter tuning
# set.seed(1)
# gbm_model <- train(as.formula(paste("Survived ~", paste(vec_var_selection_08, collapse = "+ "))),
#                    data = data_train,
#                    method = "gbm",
#                    metric = "Accuracy",
#                    maximize = T,
#                    trControl = my_ctrl,
#                    tuneGrid = tune_grid)

# print(gbm_model)

set.seed(1)
gbm_model <- train(as.formula(paste("Survived ~", paste(vec_var_selection_08, collapse = "+ "))),
                   data = data_train,
                   method = "gbm",
                   metric = "Accuracy",
                   maximize = T,
                   trControl = my_control,
                   tuneGrid = expand.grid(.n.trees = 100,
                                          .interaction.depth = 3,
                                          .shrinkage = 0.1,
                                          .n.minobsinnode = 40))

print(gbm_model)
print(gbm_model$resample)
message("CV Mean: ", round(mean(gbm_model$resample[[1]]), digits = 4)) 
message("CV SD: ", round(sd(gbm_model$resample[[1]]), digits = 4))

# Get Prediction Values for the Train Set (FITTED VALUES) & Test Set (PREDICTION VALUES)
vec_empty <- predict(gbm_model, data_train[, -which(names(data_train) == "Survived")], type = "prob")[[1]]
#print(vec_empty)

vec_empty_test <- predict(gbm_model, data_test[, -which(names(data_test) == "PassengerId")], type = "prob")[[1]]
#print(vec_empty_test)

# Write LDA fitted values to the preds_train & preds_test
data_preds_train[["GBM"]] <- vec_empty
data_preds_test[["GBM"]] <- vec_empty_test
# .....................................................................
# ---------------------------------------------------------------------



# Ensembler -----------------------------------------------------------
### Set my_ctrl -------------------------------------------------------
my_ctrl <- trainControl(method = "cv",
                        number = 10,
                        classProb = T)
# ---------------------------------------------------------------------
### RF Ensembler ......................................................
# set.seed(1)
# random_forest_model <- train(Survived ~ .,
#                              data = data_preds_train,
#                              method = 'rf',
#                              metric = 'Accuracy',
#                              maximize = T,
#                              trControl = my_ctrl,
#                              tuneGrid = expand.grid(.mtry = c(2, 3)),
#                              ntree = 50,
#                              importance = TRUE)

# print(random_forest_model)

set.seed(1)
random_forest_model <- train(Survived ~ .,
                             data = data_preds_train,
                             method = 'rf',
                             metric = 'Accuracy',
                             trControl = my_ctrl,
                             tuneGrid = expand.grid(.mtry = 2),
                             ntree = 5000,
                             importance = TRUE)

print(random_forest_model)
print(random_forest_model$resample)
message("CV Mean: ", round(mean(random_forest_model$resample[[1]]), digits = 4))
message("CV SD: ", round(sd(random_forest_model$resample[[1]]), digits = 4))


vec_feature_importance <- importance(random_forest_model$finalModel)[,1]
# print(vec_feature_importance)
for(i in 1:length(vec_feature_importance)){
    message(round(vec_feature_importance[i]/sum(vec_feature_importance), digits = 2), ": ", names(vec_feature_importance)[i])
}
# .....................................................................

### GBM Ensembler .....................................................
# tune_grid <- expand.grid(.n.trees = c(50, 100),
#                          .interaction.depth = c(2, 3, 4),
#                          .shrinkage = c(0.01, 0.02, 0.03, 0.04),
#                          .n.minobsinnode = c(30, 35, 40))

## Train the Gradient Boosting model with hyperparameter tuning
# set.seed(1)
# gbm_model <- train(Survived ~ .,
#                    data = data_preds_train,
#                    method = "gbm",
#                    metric = "Accuracy",
#                    maximize = T,
#                    trControl = my_ctrl,
#                    tuneGrid = tune_grid)

# print(gbm_model)

# Best
# n.trees = 50, interaction.depth = 3, shrinkage = 0.03 and n.minobsinnode = 40
# n.trees = 50, interaction.depth = 2, shrinkage = 0.02 and n.minobsinnode = 30
# n.trees = 1000, interaction.depth = 3, shrinkage = 0.01 and n.minobsinnode = 40

# take: n.trees = 50, interaction.depth = 3, shrinkage = 0.02 and n.minobsinnode = 30

set.seed(1)
gbm_model <- train(Survived ~ .,
                   data = data_preds_train,
                   method = "gbm",
                   metric = "Accuracy",
                   maximize = T,
                   trControl = my_ctrl,
                   tuneGrid = expand.grid(.n.trees = 50,
                                          .interaction.depth = 3,
                                          .shrinkage = 0.02,
                                          .n.minobsinnode = 30))

print(gbm_model)
print(gbm_model$resample)
message("CV Mean: ", round(mean(gbm_model$resample[[1]]), digits = 4))
message("CV SD: ", round(sd(gbm_model$resample[[1]]), digits = 4))
# .....................................................................
# ---------------------------------------------------------------------


### Submission File ---------------------------------------------------
# Get Prediction Values for the Ensemble Model on the Test Set --------
vec_empty <- predict(gbm_model, data_preds_test[, 2:length(data_preds_test)], type = "raw")
#vec_empty_probs <- predict(gbm_model, data_preds_test[, 2:length(data_preds_test)], type = "prob")

levels(vec_empty) <- c("0", "1")

data_submission <- data.frame(PassengerId = data_test$PassengerId,
                              Survived = vec_empty)
# Check
head(data_submission)

# Save the result as the submission
write.csv(data_submission,
          file = "submission.csv",
          row.names = FALSE)
# ---------------------------------------------------------------------

### Sources -----------------------------------------------------------
# Thilaksha Silva - Predicting Titanic Survival using Five Algorithms
#   - Great EDA
#   - Great Model Discussion
#   https://www.kaggle.com/code/thilakshasilva/predicting-titanic-survival-using-five-algorithms
# ---------------------------------------------------------------------

# -------- THANKS FOR READING! Comments & Recommendations appreciated! --------
# ----- Don't forget to upvote, if you like the kernel! -----
