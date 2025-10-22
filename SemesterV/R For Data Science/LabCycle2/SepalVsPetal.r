%%R
# Save the plot
png("iris_scatterplot.png", width = 1200, height = 800, res = 150)
plot(iris$Sepal.Length, iris$Petal.Length,
     xlab = "Sepal Length",
     ylab = "Petal Length",
     main = "Scatterplot of Sepal Length vs Petal Length",
     pch = 19, col = "blue")
dev.off()

# Display the saved image inline (works in some notebook environments)
library(png)
library(grid)
img <- readPNG("iris_scatterplot.png")
grid.raster(img)
