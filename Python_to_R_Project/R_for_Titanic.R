# load packages
library('tidyverse') # data manipulation
library('ggplot2') # visualization
library('ggthemes') # visualization
library('scales') # visualization
library('mice') # imputation
library('randomForest') # classification algorithm

# read csv file
train <- read.csv('titanic/train.csv', na.strings = "")

# inspect data
str(train)
head(train)
summary(train)

# clean data
# drop columns with >40% null values
null_col <- colnames(train)[sapply(train, function(x) sum(is.na(x))) > nrow(train)*0.4]
train <- train[, !(names(train) %in% null_col)]

# drop passengerid and name column
train <- subset(train, select = -c(PassengerId, Name))

# fill null values in age with mean
train$Age[is.na(train$Age)] <- mean(train$Age, na.rm = TRUE)

# define function to find mode
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# fill null values in embarked with mode
train$Embarked[is.na(train$Embarked)] <- Mode(train$Embarked)

# check train
summary(train)

