library(ggplot2)
library(dplyr)
library(GGally)

data("iris")

### ---- PART A ----
# Statistical test: Are there significant differences in sepal lengths between species?

str(iris)

ggplot(iris, aes(x = Species, y = Sepal.Length, fill = Species)) +
  geom_boxplot() +
  labs(title = "Sepal Length Distribution by Species",
       x = "Species",
       y = "Sepal Length (cm)") +
  theme_minimal()

# 1. Normality (using Shapiro test for each group)
by(iris$Sepal.Length, iris$Species, shapiro.test)

# 2. Homogeneity of variance (Levene’s Test)
library(car)
leveneTest(Sepal.Length ~ Species, data = iris)

# Perform one-way ANOVA
anova_result <- aov(Sepal.Length ~ Species, data = iris)
summary(anova_result)

# If ANOVA is significant, perform post-hoc test (Tukey HSD)
tukey_result <- TukeyHSD(anova_result)
print(tukey_result)

### ---- PART B ----
# Pair plot to visualize relationships among all variables
ggpairs(iris,
        columns = 1:4,
        mapping = aes(color = Species),
        title = "Pair Plot of Iris Dataset Variables")
