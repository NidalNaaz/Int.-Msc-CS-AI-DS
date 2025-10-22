data(mtcars)


mpg_mean <- mean(mtcars$mpg)
mpg_sd <- sd(mtcars$mpg)


breaks <- seq(floor(min(mtcars$mpg)), ceiling(max(mtcars$mpg)), by=2)


hist_info <- hist(mtcars$mpg, breaks = breaks, plot = FALSE)


max_count <- max(hist_info$counts)

colors <- rgb(0, 0, 1, alpha = hist_info$counts / max_count)

hist(mtcars$mpg,
     breaks = breaks,
     col = colors,
     xlab = "Miles per Gallon (mpg)",
     main = "Histogram of MPG with Frequency-based Blue Shades")

text(x = mpg_mean + 2, y = max_count * 0.8,
     labels = paste0("Mean = ", round(mpg_mean, 2)),
     col = "red", cex = 1.2)
text(x = mpg_mean + 2, y = max_count * 0.7,
     labels = paste0("SD = ", round(mpg_sd, 2)),
     col = "red", cex = 1.2)
