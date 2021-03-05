# loading libraries
library(readr)
library(tidyr)
library(dplyr)
library(stringr)
library(outliers)
library(data.table)
library(modeldata)
library(discretization)

# Importing the dataset
setwd('C:/Users/user/Downloads/ITS61504 Data Mining')
dataset = read.csv('Dataset Used in DM Assignment V1.01.csv')
print(dataset)

# remove whitespace
dataset$Work_Class <- trimws(dataset$Work_Class)
dataset$Education <- trimws(dataset$Education)
dataset$Marital_Status <- trimws(dataset$Marital_Status)
dataset$Sex <- trimws(dataset$Sex)
dataset$Income <- trimws(dataset$Income)

# smooth outliers in Age using z-score normalisation and the winsorisation formula
dataset$Age <- ifelse(((dataset$Age - mean(dataset$Age)) / sd(dataset$Age)) < -1.96,
                      -1.95*sd(dataset$Age)+mean(dataset$Age), dataset$Age)
dataset$Age <- ifelse(((dataset$Age - mean(dataset$Age)) / sd(dataset$Age)) > 1.96, 
                      round(1.95*sd(dataset$Age)+mean(dataset$Age)), dataset$Age) 
max(dataset$Age)
min(dataset$Age)

# smooth outliers in Hours_Per_week using z-score normalisation and the winsorisation formula
dataset$Hours_Per_week <- ifelse(((dataset$Hours_Per_week - mean(dataset$Hours_Per_week)) / sd(dataset$Hours_Per_week)) < -1.96, 
                                 round(-1.95*sd(dataset$Hours_Per_week)+mean(dataset$Hours_Per_week)), dataset$Hours_Per_week) 
dataset$Hours_Per_week <- ifelse(((dataset$Hours_Per_week - mean(dataset$Hours_Per_week)) / sd(dataset$Hours_Per_week)) > 1.96, 
                                 round(1.95*sd(dataset$Hours_Per_week)+mean(dataset$Hours_Per_week)), dataset$Hours_Per_week) 
max(dataset$Hours_Per_week)
min(dataset$Hours_Per_week)

# fill in missing values of Work_Class with mode
library(tidyverse)
val <- unique(dataset$Work_Class[!(dataset$Work_Class == "?")])
mode <- val[which.max(tabulate(match(dataset$Work_Class, val)))]
print(mode)
dataset$Work_Class <- replace(dataset$Work_Class, dataset$Work_Class == "?", mode)

# make a copy of original Education values for later use in Naive Bayes
oriEdu <- dataset$Education

# re-levelling and generalisation
dataset$Marital_Status <- replace(dataset$Marital_Status, dataset$Marital_Status == "Separated", "Not-married")
dataset$Marital_Status <- replace(dataset$Marital_Status, dataset$Marital_Status == "Divorced", "Not-married")
dataset$Marital_Status <- replace(dataset$Marital_Status, dataset$Marital_Status == "Married-civ-spouse", "Married")
dataset$Marital_Status <- replace(dataset$Marital_Status, dataset$Marital_Status == "Married-AF-spouse", "Married")
dataset$Marital_Status <- replace(dataset$Marital_Status, dataset$Marital_Status == "Married-spouse-absent", "Married")
dataset$Education <- replace(dataset$Education, dataset$Education == "5th-6th", "Lower")
dataset$Education <- replace(dataset$Education, dataset$Education == "7th-8th", "Lower")
dataset$Education <- replace(dataset$Education, dataset$Education == "9th", "Lower")
dataset$Education <- replace(dataset$Education, dataset$Education == "10th", "Lower")
dataset$Education <- replace(dataset$Education, dataset$Education == "11th", "Lower")
dataset$Education <- replace(dataset$Education, dataset$Education == "HS-grad", "Lower")
dataset$Education <- replace(dataset$Education, dataset$Education == "Some-college", "Higher")
dataset$Education <- replace(dataset$Education, dataset$Education == "Assoc-acdm", "Higher")
dataset$Education <- replace(dataset$Education, dataset$Education == "Assoc-voc", "Higher")
dataset$Education <- replace(dataset$Education, dataset$Education == "Bachelors", "Higher")
dataset$Education <- replace(dataset$Education, dataset$Education == "Masters", "Higher")
dataset$Education <- replace(dataset$Education, dataset$Education == "Doctorate", "Higher")
dataset$Education <- replace(dataset$Education, dataset$Education == "Prof-school", "Higher")
dataset$Work_Class <- replace(dataset$Work_Class, dataset$Work_Class == "Local-gov", "Governments")
dataset$Work_Class <- replace(dataset$Work_Class, dataset$Work_Class == "Federal-gov", "Governments")
dataset$Work_Class <- replace(dataset$Work_Class, dataset$Work_Class == "State-gov", "Governments")
dataset$Work_Class <- replace(dataset$Work_Class, dataset$Work_Class == "Self-emp-not-inc", "Self-emp")
dataset$Work_Class <- replace(dataset$Work_Class, dataset$Work_Class == "Self-emp-inc", "Self-emp")

# Encode and change levels of target feature/attribute Income to a binary numeric value of 0 or 1 for classification modelling
dataset$Income <- ifelse(dataset$Income == ">50K", 1, 0)

# discretise Age into three discrete levels
dataset$Age <- ifelse(dataset$Age < 33, "<33", 
                      ifelse(dataset$Age > 47, ">47", "33-47"))

# duplicate to create new datasets since Decision Tree will have four Hours_Per_week levels
# and Naive Bayes exercise will not discretise Hours_Per_week
datasetDT <- dataset # for Decision Tree Section 2
datasetNB <- dataset # for Naive Bayes Section 3

# discretise Hours_Per_week into three discrete levels
dataset$Hours_Per_week <- ifelse(dataset$Hours_Per_week < 34, "<34", 
                                 ifelse(dataset$Hours_Per_week > 48, ">48", "34-48"))
print(dataset)

# chi-squared test to obtain p-values on target attribute/feature Income are 
# 0.0036 for Education, 0.0301 for Marital_Status, 1 for Sex, 0.0142 for Work_Class, 
# 0.0020 for Hours_Per_week, 0.0163 for Age
chisq.test(dataset$Age, dataset$Income)
chisq.test(dataset$Hours_Per_week, dataset$Income)
chisq.test(dataset$Sex, dataset$Income)
chisq.test(dataset$Education, dataset$Income)
chisq.test(dataset$Work_Class, dataset$Income)
chisq.test(dataset$Marital_Status, dataset$Income)

# frequent itemsets with min support of 50% using the Apriori arules package
library(arules)
itemsets <- apriori(dataset, parameter = list(minlen=1, support=0.5, target="frequent"), 
                    appearance = list(none = c("Income=[0,1]")))
inspect(sort(itemsets, by="support"))

# concept hierarchy generation of Income into >50k and <=50k
dataset$Income <- ifelse(dataset$Income == 1, ">50k", "<=50k")

# strong association rules with min confidence of 75%
rules <- apriori(dataset, parameter = list(supp = 0.5, conf = 0.75, minlen=1, target="rules"))
inspect(sort(rules, by="confidence"))

# concept hierarchy generation of Age into three higher level concepts
datasetDT$Age <- ifelse(datasetDT$Age == "<33", "Young",
                        ifelse(datasetDT$Age == ">47", "Senior", "Middle-aged"))

# concept hierarchy generation and segmentation by natural partitioning of Hours_Per_week into 
# four higher level concepts
datasetDT$Hours_Per_week <- ifelse(datasetDT$Hours_Per_week > 51, "Too-much", 
                                   ifelse(datasetDT$Hours_Per_week < 52 & datasetDT$Hours_Per_week > 40, "Over-time",
                                          ifelse(datasetDT$Hours_Per_week < 41 & datasetDT$Hours_Per_week >29, "Full-time", "Part-time")))

# concept hierarchy generation of Income into High and Low
datasetDT$Income <- ifelse(datasetDT$Income == 1, "High", "Low")
print(datasetDT)

# Splitting the dataset into the Training set and Test set
library(caTools)
set.seed(246) #list of random numbers starting from position 111
#987, 1011, 8
split = sample.split(datasetDT$Income, SplitRatio = 0.75)
training_set = subset(datasetDT, split == TRUE) #75% of dataset used as training subset
test_set = subset(datasetDT, split == FALSE) #25% of dataset used as training subset

# Fitting Decision Tree Classification model through information gain to the Training set
# Specify model formula (Income is identified as the class) & model data
library(rpart)
classifier <- rpart(formula = Income ~., data = training_set, method="class",
                    control = rpart.control(minsplit = 2, minbucket = 1, usesurrogate = 2, 
                                            maxdepth = 6, xval=15, cp=0.0000001))
print(classifier)

# Predicting the Test set results, return prediction for each class
y_pred <- predict(classifier, newdata = test_set[-7], type = 'class')
print(y_pred)

# confusion Matrix to compare original (test set) against prediction (y_pred)
cm = table(test_set[, 7], y_pred)
print(cm)
# 24/25 = 92% accuracy 
# 6/7 = 85.71% precision
# 6/6 = 100% recall

# Plot tree
plot(classifier)

#add text inside plotting area
text(classifier)

# Beautify tree font size, and positioning and colour 
library(rattle)
library(rpart.plot)
library(RColorBrewer)
prp(classifier, cex = 0.55, faclen = 0, extra = 1, border.col = 'maroon') 

# classification rules
rpart.rules(classifier, roundint=FALSE, clip.facs=TRUE)

# concept hierarchy generation of Income into High and Low
datasetNB$Income <- ifelse(datasetNB$Income == 1, "High", "Low")
print(datasetNB)

# use original Education values
datasetNB$Education <- oriEdu

print(datasetNB)

# fitting naive bayes
library(e1071)
classifierNB = naiveBayes(x = datasetNB[-7],
                          y = datasetNB$Income)
print(classifierNB)

# predict posterior probabilities based on specified characteristics
q1 <- data.frame(Age = "Middle-aged" , Education = "Bachelors", Hours_Per_week = 43)
predict(classifierNB, q1, type = "raw") 
q2 <- data.frame(Age = "Middle-aged" , Work_Class = "Private", Education = "Bachelors", 
                 Work_Class = "Governments", Hours_Per_week = 43)
predict(classifierNB, q2, type = "raw")
q3a <- data.frame(Marital_Status="Not-married")
predict(classifierNB, q3a, type = "raw") 
q3b <- data.frame(Marital_Status="Never-married")
predict(classifierNB, q3b, type = "raw") 
q3c <- data.frame(Marital_Status="Married")
predict(classifierNB, q3c, type = "raw") 
q4 <- data.frame(Age = "Senior" , Work_Class = "Private", Education = "5th-6th", 
                 Education = "7th-8th", Education = "9th", Education = "10th", 
                 Education = "11th", Education = "HS-grad", Education = "Some-college",
                 Education = "Assoc-acdm", Education = "Assoc-voc", Hours_Per_week = 50, 
                 Marital_Status="Married")
predict(classifierNB, q4, type = "raw")