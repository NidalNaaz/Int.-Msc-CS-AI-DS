data(mtcars)


colors <- c("4" = "red", "6" = "green", "8" = "blue")


plot(mtcars$disp, mtcars$mpg,
     col = colors[as.character(mtcars$cyl)],
     pch = 19,
     xlab = "Displacement (disp)",
     ylab = "Miles per Gallon (mpg)",
     main = "MPG vs Displacement Colored by Cylinders")

lines(
  sort(mtcars$disp),
  predict(loess(mpg ~ disp, data = mtcars, span = 0.75), newdata = data.frame(disp = sort(mtcars$disp))),
  col = "black",
  lwd = 2
)

legend("topright",
       legend = c("4 cylinders", "6 cylinders", "8 cylinders"),
       col = c("red", "green", "blue"),
       pch = 19)
