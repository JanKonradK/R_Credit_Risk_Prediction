# Install and load necessary packages
if (!require(dplyr)) install.packages("dplyr")
if (!require(pdp)) install.packages("pdp")
if (!require(randomForest)) install.packages("randomForest")
if (!require(ggplot2)) install.packages("ggplot2")
if (!require(caret)) install.packages("caret")
if (!require(pROC)) install.packages("pROC")

library(dplyr)
library(pdp)
library(randomForest)
library(ggplot2)
library(caret)
library(pROC)

# Load the dataset
data <- read.table("C:/Users/Konra/Documents/Projects/Credit Risk R/statlog+german+credit+data/german.data", header = FALSE, stringsAsFactors = TRUE)

# Assign column names
colnames(data) <- c("Status", "Duration", "CreditHistory", "Purpose", "CreditAmount",
                    "Savings", "Employment", "InstallmentRate", "PersonalStatusSex",
                    "OtherDebtors", "ResidenceSince", "Property", "Age", "OtherInstallmentPlans",
                    "Housing", "ExistingCredits", "Job", "LiablePeople", "Telephone", "ForeignWorker", "CreditRisk")

# Map the "A" codes to descriptive labels
data <- data %>%
  mutate(
    Status = recode(Status,
                    "A11" = "< 0 DM",
                    "A12" = "0 <= ... < 200 DM",
                    "A13" = ">= 200 DM / salary assignments for at least 1 year",
                    "A14" = "no checking account"),
    CreditHistory = recode(CreditHistory,
                           "A30" = "no credits taken / all credits paid back duly",
                           "A31" = "all credits at this bank paid back duly",
                           "A32" = "existing credits paid back duly till now",
                           "A33" = "delay in paying off in the past",
                           "A34" = "critical account / other credits existing (not at this bank)"),
    Purpose = recode(Purpose,
                     "A40" = "car (new)",
                     "A41" = "car (used)",
                     "A42" = "furniture/equipment",
                     "A43" = "radio/television",
                     "A44" = "domestic appliances",
                     "A45" = "repairs",
                     "A46" = "education",
                     "A47" = "vacation",
                     "A48" = "retraining",
                     "A49" = "business",
                     "A410" = "others"),
    Savings = recode(Savings,
                     "A61" = "< 100 DM",
                     "A62" = "100 <= ... < 500 DM",
                     "A63" = "500 <= ... < 1000 DM",
                     "A64" = ">= 1000 DM",
                     "A65" = "unknown / no savings account"),
    Employment = recode(Employment,
                        "A71" = "unemployed",
                        "A72" = "< 1 year",
                        "A73" = "1 <= ... < 4 years",
                        "A74" = "4 <= ... < 7 years",
                        "A75" = ">= 7 years"),
    PersonalStatusSex = recode(PersonalStatusSex,
                               "A91" = "male : divorced/separated",
                               "A92" = "female : divorced/separated/married",
                               "A93" = "male : single",
                               "A94" = "male : married/widowed",
                               "A95" = "female : single"),
    OtherDebtors = recode(OtherDebtors,
                          "A101" = "none",
                          "A102" = "co-applicant",
                          "A103" = "guarantor"),
    Property = recode(Property,
                      "A121" = "real estate",
                      "A122" = "building society savings agreement / life insurance",
                      "A123" = "car or other, not in attribute 6",
                      "A124" = "unknown / no property"),
    OtherInstallmentPlans = recode(OtherInstallmentPlans,
                                   "A141" = "bank",
                                   "A142" = "stores",
                                   "A143" = "none"),
    Housing = recode(Housing,
                     "A151" = "rent",
                     "A152" = "own",
                     "A153" = "for free"),
    Job = recode(Job,
                 "A171" = "unemployed / unskilled - non-resident",
                 "A172" = "unskilled - resident",
                 "A173" = "skilled employee / official",
                 "A174" = "management / self-employed / highly qualified employee / officer"),
    Telephone = recode(Telephone,
                       "A191" = "none",
                       "A192" = "yes, registered under the customer's name"),
    ForeignWorker = recode(ForeignWorker,
                           "A201" = "yes",
                           "A202" = "no"),
    CreditRisk = factor(CreditRisk, levels = c("1", "2"), labels = c("Good", "Bad"))
  )

# Convert categorical variables to factors
categorical_vars <- c("Status", "CreditHistory", "Purpose", "Savings", "Employment",
                      "PersonalStatusSex", "OtherDebtors", "Property", "OtherInstallmentPlans",
                      "Housing", "Job", "Telephone", "ForeignWorker", "CreditRisk")

data[categorical_vars] <- lapply(data[categorical_vars], as.factor)

# Split the data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(data$CreditRisk, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Train the Random Forest model directly
set.seed(123)
rf_model <- randomForest(CreditRisk ~ ., data = trainData, ntree = 50000, mtry = 150, importance = TRUE)

# Feature Importance
varImpPlot(rf_model, main = "Variable Importance")

# Evaluate the model on the test data
predictions <- predict(rf_model, testData, type = "prob")[, "Bad"]  # Get probabilities for the "Bad" class

# Calculate AUC-ROC
roc_curve <- roc(testData$CreditRisk, predictions, levels = c("Good", "Bad"), direction = "<")
auc_value <- auc(roc_curve)
cat("AUC:", auc_value, "\n")

# Plot ROC curve using ggplot2
roc_data <- data.frame(
  TPR = roc_curve$sensitivities,
  FPR = 1 - roc_curve$specificities
)

ggplot(roc_data, aes(x = FPR, y = TPR)) +
  geom_line(color = "blue", size = 1, linetype = "solid") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(title = paste("ROC Curve (AUC =", round(auc_value, 2), ")"),
       x = "False Positive Rate",
       y = "True Positive Rate") +
  theme_minimal() +
  theme(panel.grid.major = element_line(color = "grey", size = 0.5),
        panel.grid.minor = element_line(color = "lightgrey", size = 0.25)) +
  scale_x_continuous(breaks = seq(0, 1, by = 0.1)) +
  scale_y_continuous(breaks = seq(0, 1, by = 0.1)) +
  annotate("text", x = 0.5, y = 0.5, label = "Random Guessing", color = "red", angle = 40, vjust = -1)

# Error Rate Plot with Legend
plot(rf_model, main = "Error Rate vs. Number of Trees")
legend("topright", legend = c("Overall", "Good", "Bad"), col = c(1, 2, 3), lty = 1, cex = 0.8)

# Distribution of Predictions
ggplot(data.frame(Predictions = factor(ifelse(predictions > 0.5, "Bad", "Good"))), aes(x = Predictions)) +
  geom_bar(fill = "steelblue") +
  labs(title = "Distribution of Predictions", x = "Predicted Class", y = "Count") +
  theme_minimal() +
  scale_x_discrete(labels = c("Good", "Bad"))

# Save the model
saveRDS(rf_model, file = "C:/Users/Konra/Documents/Projects/Credit Risk R/rf_model.rds")

# Load the model (example)
# loaded_model <- readRDS("C:/Users/Konra/Documents/Projects/Credit Risk R/rf_model.rds")

# Generate Partial Dependence Data for 'CreditAmount'
partial_data <- pdp::partial(
  object = rf_model,
  pred.var = "CreditAmount",
  train = trainData,
  plot = FALSE  # Return the data for custom plotting
)

# Plot the partial dependence using ggplot2
ggplot(partial_data, aes(x = CreditAmount, y = yhat)) +
  geom_line(color = "blue") +
  geom_point(color = "red") +
  labs(title = "Partial Dependence Plot for Credit Amount",
       x = "Credit Amount",
       y = "Partial Dependence") +
  theme_minimal()