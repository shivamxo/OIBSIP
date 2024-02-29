#importing the data set
credit_card <- read.csv("creditcard.csv")

# Glance at the structure of our data set
str(credit_card)

# Convert class to a factor variable
credit_card$Class <- factor(credit_card$Class, levels = c(0,1))

# Get the summary of the data
summary(credit_card)

# Count the missing values
sum(is.na(credit_card))

# Distribution of legit and fraud transaction of the data set
table(credit_card$Class)

# Find percentage of legit and fraud transaction of data set
prop.table(table(credit_card$Class))

# pie chart of transactions
labels <- c("legit","fraud")
labels <- paste(labels,round(100*prop.table(table(credit_card$Class)),2))
labels <- paste0(labels,"%")

pie(table(credit_card$Class),labels, col = c("blue","yellow"),main = "Pie chart of Transactions")

#------------------------------------------------------------------------#

# no model Prediction
predictions <- rep.int(0,nrow(credit_card))
predictions <- factor(predictions, levels = c(0,1))

library(caret)
confusionMatrix(data = predictions,reference = credit_card$Class)

#----------------------------------------------------------------------#
library(dplyr)

set.seed(1)
credit_card <- credit_card %>% sample_frac(0.1)

table(credit_card$Class)

# Create scatter plot for data set
library(ggplot2)

ggplot(data = credit_card, aes(x = V1, y = V2, col= Class)) + 
  geom_point() + 
  theme_light() + 
  scale_colour_manual(values = c('dodgerblue2','red'))
#-----------------------------------------------------------------------#

# Create training and test set for our Fraud Detection Model
library(caTools)

set.seed(123)

data_sample = sample.split(credit_card$Class, SplitRatio = 0.80)

train_data = subset(credit_card,data_sample== TRUE)

test_data = subset(credit_card, data_sample== FALSE)

dim(train_data)
dim(test_data)

# Random Over Sampling
table(train_data$Class)

n_legit <- 22750
new_frac_legit <- 0.50
new_n_total <- n_legit/new_frac_legit

library(ROSE)
oversampling_result <- ovun.sample(Class ~ .,
                                   data = train_data,
                                   method = "over",
                                   N = new_n_total,
                                   seed = 2019)

oversampled_credit <- oversampling_result$data

table(oversampled_credit$Class)

ggplot(data = oversampled_credit, aes(x = V1,y = V2, col= Class)) +
  geom_point() +
  theme_bw() +
  scale_color_manual(values = c('dodgerblue','red'))
#---------------------------------------------------------------------#
# Random under sampling

table(train_data$Class)

n_fraud <- 35
new_frac_fraud <- 0.50
new_n_total <- n_fraud/new_frac_fraud

undersampling_result <- ovun.sample(Class ~ .,
                                    data = train_data,
                                    method = "under",
                                    N = new_n_total,
                                    seed = 2019)
undersampled_credit <- undersampling_result$data

table(undersampled_credit$Class)

ggplot(data = undersampled_credit,aes(x = V1, y = V2,col = Class)) +
  geom_point() +
  theme_bw() +
  scale_color_manual(values = c('dodgerblue','red'))
#------------------------------------------------------------------------#
# ROS and RUS
n_new <- nrow(train_data)
fraction_fraud_new <- 0.50

sampling_result <- ovun.sample(Class ~ .,
                               data = train_data,
                               method = "both",
                               N = n_new,
                               p = fraction_fraud_new,
                               seed = 2019)

sampled_credit <- sampling_result$data

table(sampled_credit$Class)

prop.table(table(sampled_credit$Class))

ggplot(data = sampled_credit, aes(x = V1, y = V2, col = Class)) +
  geom_point() +
  theme_bw() +
  scale_colour_manual(values = c ('dodgerblue2','red'))

# Using SMOTE To balance data
library(smotefamily)

table(train_data$Class)

# Set the number of fraud and legitimate cases, and the desired percentage of legitimate cases

n0 <- 22750
n1 <- 35
r0 <- 0.6

# Calculate the value for the dup_size parameter of SMOTE

ntimes <- ((1 - r0) / r0) * (n0 / n1) - 1

smote_output = SMOTE(X = train_data[, -c(1, 31)],
                     target = train_data$Class,
                     K = 5,
                     dup_size = ntimes)

credit_smote <- smote_output$data

colnames(credit_smote)[30] <- "Class"

prop.table(table(credit_smote$Class))

# Class distribution for original data set
ggplot(train_data, aes(x = V1, y = V2, color = Class)) +
  geom_point() +
  scale_color_manual(values = c('dodgerblue2','red'))

# Class distribution for over-sampled data set using SMOTE
ggplot(credit_smote, aes(x = V1, y = V2, color = Class)) +
  geom_point() +
    scale_color_manual(values = c('dodgerblue2','red'))

library(rpart)
library(rpart.plot)

CART_model <- rpart(Class ~ . , credit_smote)

rpart.plot(CART_model, extra = 0, type = 5, tweak = 1.2 )

# Predict fraud classes
predicted_val <- predict(CART_model, test_data, type = 'class')

# Build confusion matrix
library(caret)
confusionMatrix(predicted_val, test_data$Class)

# Decision tree without SMOTE

CART_model <- rpart(Class ~ ., train_data[,-1])

rpart.plot(CART_model, extra = 0, type = 5, tweak = 1.2)

# Predict fraud classes
predicted_val <- predict(CART_model, test_data[,-1], type = 'class')

library(caret)
confusionMatrix(predicted_val, test_data$Class)

#----------------------------------------------------------------------------#

predicted_val <- predict(CART_model, credit_card[,-1], type = 'class')
confusionMatrix(predicted_val, credit_card$Class)
