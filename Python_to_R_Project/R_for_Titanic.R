# load packages
library('tidyverse') # data manipulation
library('ggplot2') # visualization
library('ggthemes') # visualization
library('scales') # visualization
library('mice') # imputation
library('FNN') # classification algorithm
library('dummies') # get dummies
library('reshape2') # reshape matrix
library('gridExtra') # arrange plots
library('caret') # ML library
library('caTools') # for train test split


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

# drop passengerid, name and ticket column
train <- subset(train, select = -c(PassengerId, Name, Ticket))

# fill null values in age with mean
train$Age[is.na(train$Age)] <- mean(train$Age, na.rm = TRUE)

# define function to find mode
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# fill null values in embarked with mode
train$Embarked[is.na(train$Embarked)] <- Mode(train$Embarked)

# create a copy
train_clean = data.frame(train)

# check train
summary(train)

# get dummies for sex
train$Male <- ifelse(train$Sex == 'male', 1, 0)

# get dummies for embarked
train <- cbind(train, dummy(train$Embarked, sep='_'))

# drop original/extra columns
train <- subset(train, select = -c(Sex, Embarked, train_C))

# check train
head(train)

# data visualisation

# plot heatmap
train_cor <- round(cor(train, use="all.obs"), 2)
head(train_cor)

ggplot(data = melt(train_cor), aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                     midpoint = 0, limit = c(-1,1), space = "Lab", 
                     name="Pearson\nCorrelation")

# plot
p1 <- qplot(Pclass, data = train, geom = "histogram", group = Survived, fill = Survived)
p2 <- qplot(Male, data = train, geom = "histogram", group = Survived, fill = Survived)
p3 <- qplot(Fare, data = train, geom = "histogram", group = Survived, fill = Survived)
grid.arrange(p1, p2, p3, nrow=3)

# modelling

# train test split
set.seed(42)
sample <- sample.split(train, SplitRatio = 0.67)
train <- subset(train, sample == TRUE)
test <- subset(train, sample == FALSE)

# create X and y
X_train <- subset(train, select = -c(Survived))
y_train <- train$Survived

X_test <- subset(test, select = -c(Survived))
y_test <- test$Survived

# scale
Xs_train <- scale(X_train)

# logistic regression
to_train <- cbind(X_train, y_train)
lr <- glm(y_train ~ ., data = to_train, family='binomial')
summary(lr)

# knn
cl <- factor(y_train)
fit <- knn(X_train, X_test, cl, k=1)
summary(fit)


