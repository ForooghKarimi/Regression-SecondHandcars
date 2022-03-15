#importing libraries
library(psych)  # general functions
library(ggplot2)  # data visualization
library(car)  # regression diagnostics

#loading dataset
df=read.csv('D:/Statistics-Eyvazian/Project/project- Regression- Second Hand Car Price/car.csv')

head(df)   #first 6 rows

dim(df)  #dimension of dataset

str(df)  #structure of predictors

df$Car_Name=factor(df$Car_Name)
df$Fuel_Type=factor(df$Fuel_Type)
df$Seller_Type=factor(df$Seller_Type)
df$Transmission=factor(df$Transmission)
df$Owner=factor(df$Owner)


summary(df)  #statistical summary of dataset

sum(is.na(df))  #checking for null values

table(df$Car_Name)  #the number of levels in Car_Name and the frequency of each level
table(df$Fuel_Type)   #the number of levels in Fuel_Type and the frequency of each level
#As the frequency of CNG is low, it is better that CNG get combined with another level with which has the most similarity.
table(df$Seller_Type)   #the number of levels in Seller_Type and the frequency of each level
table(df$Transmission )   #the number of levels in Transmission  and the frequency of each level
table(df$Owner)   #the number of levels in Owner  and the frequency of each level
#As the frequency of group "3" is low, it is better that group "3" get combined with another level with which has the most similarity.

#Visualizong Selling_Price against Owner Predictor, in order to understand to which group of Owner,
#group "3" has the most similarity
ggplot(df, aes(x=Owner, y= Selling_Price)) + geom_point()

#As above plot, it is better to categorize group "3" of Owner to group "1"
df$Owner[df$Owner=='3']='1'
df$Owner=factor(df$Owner)
table(df$Owner)

#Visualizong Selling_Price against "Fuel_Type" Predictor, in order to understand to which group of "Fuel_Type",
#"CNG" has the most similarity
ggplot(df, aes(x=Fuel_Type, y= Selling_Price)) + geom_point()

#it seems that CNG has the most similarity with Petrol
df$Fuel_Type[df$Fuel_Type=='CNG']='Petrol'
df$Fuel_Type=factor(df$Fuel_Type)
table(df$Fuel_Type)

str(df)

describe(df)

#plotting the interaction between predictors,in order to detect "multicollinearity"
pairs(~Selling_Price + Year+Present_Price+Kms_Driven+Fuel_Type+Seller_Type+Transmission+Owner ,data = df)

#the effect of "Transmission" and "Owner" on "Selling_Price"
ggplot(df, aes(x=Transmission, y= Selling_Price, color = Owner)) + geom_boxplot()

#the effect of "Seller_Type" and "Owner" on "Selling_Price"
ggplot(df, aes(x=Seller_Type, y=Selling_Price , color = Owner)) + geom_boxplot()

#fitting the model
fit=lm(Selling_Price ~ Year+Present_Price+Kms_Driven+Fuel_Type+Seller_Type+Transmission+Owner , data=df)

library(car)  # for regression diagnostics
vif(fit) #calculating Variance Inflation Factor for  detecting "multicollinearity"

#the __VIF__ for all predictors are below "5", which are satisfactory levels

layout(matrix(c(1,2,3,4),2,2)) # optional 4 graphs/page
plot(fit)
#Plots "Residuals vs Fitted" and "Scale-Location" shows that there is __"Heteroscedasticity"__ in the dataset.
#According to "Residuals vs Leverage" plot, observations "197", "87" and "65" are __"Outliers"__, based on Cook's Distance.

#for solving "Heteroscedasticity" problem we can run "Box-Cox" Transformation
library(EnvStats)
b = boxcox(df$Selling_Price, objective.name = "Log-Likelihood", optimize = TRUE)
df$Selling_Price2 = boxcoxTransform(df$Selling_Price, lambda = b$lambda)

# we run the model with transformed response variable
fit= lm( Selling_Price2 ~ Year+Present_Price+Kms_Driven+Fuel_Type+Seller_Type+Transmission+Owner , data=df)
layout(matrix(c(1,2,3,4),2,2)) # optional 4 graphs/page
plot(fit)

#It seems that the problem of __Heteroscedasticity__ is solved.

#summary of the model
summary(fit)

#in order to detect "Outlier" residuals, we need to save residuals in a variable
library(MASS)
residuals= residuals(fit)

#the, we sort it in descending format
residuals=sort(abs(residuals), decreasing=TRUE)

#calculating "Standardized Residuals"
stan_resid = rstandard(fit)

#calculating "Studentized Residuals"
stud_resid=rstudent(fit)

#in order to detect "oulier" predictors that are influenctial we can use "Cook's Distance" method
cook = cooks.distance(fit)
plot(cook, ylab="Cooks distances")

#Another method for detecting influential outliers is "Leverage", which uses Hat matrix
leverage = hat(model.matrix(fit))
plot(leverage)

#we need to know to which observations,the detected outliers belong
library(olsrr)  
ols_plot_cooksd_chart(fit)
ols_plot_resid_stud(fit)
ols_plot_resid_stand(fit)

pairs(~Selling_Price2 + Year+Present_Price+Kms_Driven+Fuel_Type+Seller_Type+Transmission+Owner ,data = df)
coefficients(fit) # model coefficients
confint(fit, level=0.95) # CIs for model parameters

library(lubridate)
library(caret)
library(tidyverse)

set.seed(123)

#train & test splitting
y_train=df$Selling_Price2 %>%  createDataPartition(p = 0.8, list = FALSE)
train_data=df[y_train,]
test_data=df[-y_train,]

#Building the model
fit2= lm(Selling_Price2 ~  Year+Present_Price+Kms_Driven+Fuel_Type+Seller_Type+Transmission , data = train_data)

#make predictions for test data
pred=fit2 %>% predict(test_data)

#calculating R-Square, RMSE and MAE
data.frame( R2 = R2(pred, test_data$Selling_Price2),
            RMSE = RMSE(pred, test_data$Selling_Price2),
            MAE = MAE(pred, test_data$Selling_Price))

#We choose a model that has a lower rmse/mean(error rate)
RMSE(pred, test_data$Selling_Price2)/mean(test_data$Selling_Price2)

#deleting outlier (we only delete observation 87, which is identified as outlier in all methods)
df=df[-c(87),]

#again we do train test split, build the model and make prediction
library(lubridate)
library(caret)
library(tidyverse)
set.seed(123)
y_train <- df$Selling_Price2 %>%  createDataPartition(p = 0.8, list = FALSE)
train_data  <- df[y_train, ]
test_data <- df[-y_train, ]
# Build the model
fit2 <- lm(Selling_Price2 ~  Year+Present_Price+Kms_Driven+Fuel_Type+Seller_Type+Transmission , data = train_data)
# Make predictions and compute the R2, RMSE and MAE
predictions <- fit2 %>% predict(test_data)
data.frame( R2 = R2(predictions, test_data$Selling_Price2),
            RMSE = RMSE(predictions, test_data$Selling_Price2),
            MAE = MAE(predictions, test_data$Selling_Price))
RMSE(predictions, test_data$Selling_Price2)/mean(test_data$Selling_Price2)

#it is better to check whether there is interaction between variables or not
library(jtools)
library(ggplot2)
library(interactions)

# we use interaction plot for detecting the interaction between a categorica and one numerical variable
interact_plot(fit, pred = Present_Price, modx = Owner)
#two lines are parallel > it shows that there is no interaction between these two variable

interact_plot(fit, pred = Kms_Driven, modx = Owner)
#the same as the previous one

# we use cat plot in order to detect interaction between two categorical variables
cat_plot(fit, pred = Transmission, modx = Owner)
cat_plot(fit, pred = Transmission, modx = Owner, geom = "line", point.shape = TRUE)
#these plots also shows that there is nor interaction between these two variables



###### STEPWISE #####
#now we want to use "stepwise forward regression" method for finding the best fit
library(MASS)

#we remove column 1 (response variable) and column 3 (car name, which is not important in our model,
#because it is accounted in the present price)
df=df[,-c(1,3)]
head(df)

set.seed(123)
y_train=df$Selling_Price2 %>%  createDataPartition(p = 0.8, list = FALSE)
train.data  <- df[y_train, ]
test.data <- d[-y_train, ]
#step wise 
#Forward

nullmod= lm(Selling_Price2 ~ 1, data = train_data)
fullmod=lm(Selling_Price2 ~  Year+Present_Price+Kms_Driven+Fuel_Type+Seller_Type+Transmission+Transmission:Seller_Type+Owner:Seller_Type+Owner:Present_Price+Present_Price:Owner,data=train_data)
reg1A= step(nullmod, scope = list(lower = nullmod, upper = fullmod),
              direction="forward")

reg1A
str(summary(reg1A))
summary(reg1A)


#Backwards
reg2B= step(fullmod, scope = list(lower = nullmod, upper = fullmod),
              direction="backward")
str(summary(reg2B))
summary(reg2B)


#both Stepwise
reg1C <- step(nullmod, scope = list(lower = nullmod, upper = fullmod),
              direction="both")
reg1C
summary(reg1C)


#all of subset 
library(leaps)
#attach(df)
leaps=regsubsets(Selling_Price2 ~ Year+Present_Price+Kms_Driven+Fuel_Type+Seller_Type+Transmission+Owner , data=df,nbest=1)
graphics.off()
par("mar")
par(mar=c(1,1,1,1))
plot(leaps,scale="r2")

res.sum= summary(leaps)
res.sum
data.frame(
  Adj.R2 = which.max(res.sum$adjr2),
  CP = which.min(res.sum$cp),
  BIC = which.min(res.sum$bic)
)
#each index shows how many variable can build the best fit

#plot statistic by subset size, based on r2
library(car)
library(leaps)
par("mar")
par(mar=c(1,1,1,1))
subsets(leaps, statistic="rsq")

#seeing all possible steps
library(olsrr)
fit3=lm(Selling_Price2 ~  Year+Present_Price+Kms_Driven+Fuel_Type+Seller_Type+Transmission+Owner , data=df)
test=ols_step_all_possible(fit3)
View(test)
plot(test)

#Building Polynomial regression
#Imporing required library
library(tidyverse)
library(caret)
# Build the model
model <- lm(Selling_Price2 ~  poly(Year,3, raw = TRUE)+poly(Present_Price, 5, raw = TRUE)+Kms_Driven+Fuel_Type+Seller_Type+Transmission+Owner, 
            data = train_data)

# Make predictions
predictions=model %>% predict(test_data)
data.frame( R2 = R2(predictions, test_data$Selling_Price2),
            RMSE = RMSE(predictions, test_data$Selling_Price2),
            MAE = MAE(predictions, test_data$Selling_Price))

RMSE(predictions, test_data$Selling_Price2)/mean(test_data$Selling_Price2)# Model performance
modelPerfomance = data.frame(
  RMSE = RMSE(predictions, test_data$Selling_Price2),
  R2 = R2(predictions, test_data$Selling_Price2)
)
modelPerfomance
ggplot(train_data, aes(Selling_Price2, Present_Price) ) + geom_point() + 
  stat_smooth(method = lm, formula = y ~ poly(x,3, raw = TRUE))

print(model)
print(modelPerfomance)
RMSE(predictions, test_data$Selling_Price2)/mean(test_data$Selling_Price2)#We choose a model that has a lower rmse/mean(error rate)


#stepwise & poly & interaction
fit2=lm(Selling_Price2 ~  poly(Year,3, raw = TRUE)+ poly(Present_Price, 5, raw = TRUE)+ Kms_Driven+Fuel_Type+Seller_Type+Transmission+Transmission:Seller_Type+Owner:Seller_Type+Owner:Present_Price+Present_Price:Owner, data = train_data)
#Stepwise both
nullmod=lm(Selling_Price2 ~ 1, data = train_data)
fullmod=lm(Selling_Price2 ~  poly(Year,3, raw = TRUE)+poly(Present_Price, 5, raw = TRUE)+Kms_Driven+Fuel_Type+Seller_Type+Transmission+Transmission:Seller_Type+Owner:Seller_Type+Owner:Present_Price+Present_Price:Owner, data = train_data)
reg2C= step(nullmod, scope = list(lower = nullmod, upper = fullmod),
              direction="both" )
reg2B= step(fullmod, scope = list(lower = nullmod, upper = fullmod),
              direction="backward")
reg2A=  step(nullmod, scope = list(lower = nullmod, upper = fullmod),
              direction="forward")

#testing for backward & stepwise & poly
model2=lm(Selling_Price2 ~ poly(Present_Price, 5, raw = TRUE) + poly(Year,3, raw = TRUE) + Fuel_Type + Kms_Driven,train_data)
# Make predictions
predictions2 <- model2 %>% predict(test_data)
# Model performance
modelPerfomance2 = data.frame(
  RMSE = RMSE(predictions2, test_data$Selling_Price2),
  R2 = R2(predictions2, test_data$Selling_Price2))
modelPerfomance2
RMSE(predictions2, test_data$Selling_Price2)/mean(test_data$Selling_Price2)#We choose a model that has a lower rmse/mean(error rate)

#kfold crossvalidation

#Define training control
library(caret)
set.seed(123) 
train_control= trainControl(method = "cv", number = 10)


#Train the model
fitcv= train(Selling_Price2 ~ poly(Present_Price, 5, raw = TRUE) + poly(Year,3, raw = TRUE) + Fuel_Type + Kms_Driven, data = d, method = "lm",
               trControl = train_control)

#Summarize the results
print(fitcv)


