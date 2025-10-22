data(mtcars)

mtcars$gear <- as.factor(mtcars$gear)

col_values <- c("3" = "skyblue", "4" = "lightgreen", "5" = "lightpink")
pch_values <- c("3" = 21, "4" = 22, "5" = 24)  

y_min <- min(mtcars$hp) - 10
y_max <- max(mtcars$hp) + 20

bp <- boxplot(hp ~ gear,
              data = mtcars,
              main = "Box Plot of Horsepower by Gear Type",
              xlab = "Number of Gears",
              ylab = "Horsepower (hp)",
              col = col_values,
              border = "gray30",
              ylim = c(y_min, y_max))

set.seed(123)

for (g in levels(mtcars$gear)) {
  idx <- which(mtcars$gear == g)
  x_pos <- rep(which(levels(mtcars$gear) == g), length(idx))  
  points(jitter(x_pos, amount = 0.1),
         mtcars$hp[idx],
         pch = pch_values[g],
         col = "black",
         bg = col_values[g],
         cex = 1.2)
}

outliers_df <- data.frame()
for (g in levels(mtcars$gear)) {
  vals <- mtcars$hp[mtcars$gear == g]
  stats <- boxplot.stats(vals)
  if (length(stats$out) > 0) {
    outliers_df <- rbind(outliers_df,
                         data.frame(gear = g, hp = stats$out))
  }
}

if (nrow(outliers_df) > 0) {
  text(x = sapply(outliers_df$gear, function(g) which(levels(mtcars$gear) == g)) + 0.15,
       y = outliers_df$hp,
       labels = round(outliers_df$hp, 1),
       col = "red",
       pos = 4,
       cex = 0.8)
}

legend("topright",
       legend = paste(levels(mtcars$gear), "gears"),
       pch = pch_values,
       pt.bg = col_values,
       col = "black",
       title = "Gear Type",
       bty = "o")
