data(mtcars)

carb_levels <- sort(unique(mtcars$carb))
colors <- rainbow(length(carb_levels))
sizes <- seq(1, 3, length.out = length(carb_levels))
names(colors) <- carb_levels
names(sizes) <- carb_levels

plot(mtcars$disp, mtcars$wt,
     pch = 19,
     col = colors[as.character(mtcars$carb)],
     cex = sizes[as.character(mtcars$carb)],
     xlab = "Displacement (disp)",
     ylab = "Weight (wt)",
     main = "Displacement vs Weight with Carburetors")

legend("topright",
       legend = paste(carb_levels, "Carburetors"),
       col = colors,
       pt.cex = sizes,
       pch = 19,
       title = "Carburetors")

# Add smooth line (LOESS)
loess_fit <- loess(wt ~ disp, data = mtcars)
disp_seq <- seq(min(mtcars$disp), max(mtcars$disp), length.out = 100)
lines(disp_seq, predict(loess_fit, newdata = data.frame(disp = disp_seq)),
      col = "blue", lwd = 2)
