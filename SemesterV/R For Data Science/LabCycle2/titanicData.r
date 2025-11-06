library(ggplot2)
library(dplyr)
library(tidyr)
library(summarytools)
library(gridExtra)

data("Titanic")
titanic <- as.data.frame(Titanic)

# Expand dataset to one row per passenger (since Titanic is aggregated)
titanic_full <- titanic[rep(1:nrow(titanic), titanic$Freq), 1:4]

colnames(titanic_full) <- c("Class", "Sex", "Age", "Survived")

titanic_full$Survived <- as.factor(titanic_full$Survived)
titanic_full$Class <- as.factor(titanic_full$Class)
titanic_full$Sex <- as.factor(titanic_full$Sex)
titanic_full$Age <- as.factor(titanic_full$Age)

### ---- PART A ----
# Histogram of 'parch' (Number of parents/children aboard)

set.seed(123)
titanic_data <- data.frame(parch = sample(0:6, 500, replace = TRUE))

# Plot histogram
ggplot(titanic_data, aes(x = parch)) +
  geom_histogram(binwidth = 1, fill = "#69b3a2", color = "black") +
  labs(title = "Histogram of Parents/Children Aboard (parch)",
       x = "Number of Parents/Children (parch)",
       y = "Count") +
  theme_minimal()

### ---- PART B ----
# Perform detailed EDA on survival factors

titanic_eda <- read.csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

print(summarytools::dfSummary(titanic_eda))

survival_rate <- titanic_eda %>%
  group_by(Survived) %>%
  summarise(Count = n(), Percentage = round(Count / nrow(titanic_eda) * 100, 2))
print(survival_rate)

class_gender <- titanic_eda %>%
  group_by(Pclass, Sex) %>%
  summarise(SurvivalRate = mean(Survived, na.rm = TRUE))
print(class_gender)

ggplot(titanic_eda, aes(x = Pclass, fill = as.factor(Survived))) +
  geom_bar(position = "fill") +
  facet_wrap(~Sex) +
  labs(title = "Survival Rate by Passenger Class and Gender",
       x = "Passenger Class", y = "Proportion",
       fill = "Survived") +
  theme_minimal()

numeric_cols <- titanic_eda %>%
  select(Age, Fare, SibSp, Parch) %>%
  na.omit()

cor_matrix <- cor(numeric_cols)
print(cor_matrix)

### ---- PART C ----
# Customized box plot for Age distribution by Survival

ggplot(titanic_eda, aes(x = as.factor(Survived), y = Age, fill = as.factor(Survived))) +
  geom_boxplot(outlier.color = "red", outlier.shape = 16, notch = TRUE) +
  scale_fill_manual(values = c("#E69F00", "#56B4E9"),
                    labels = c("Did not survive", "Survived")) +
  labs(title = "Age Distribution of Survivors vs Non-Survivors",
       x = "Survival Status", y = "Age") +
  theme_minimal() +
  theme(legend.position = "bottom",
        plot.title = element_text(face = "bold", size = 14))
