data(mtcars)


cyl_am_table <- table(mtcars$cyl, mtcars$am)

colors <- c("0" = "orange", "1" = "purple")  # 0 = automatic, 1 = manual

barplot(cyl_am_table,
        col = colors,
        beside = TRUE,
        xlab = "Number of Cylinders",
        ylab = "Count of Cars",
        main = "Number of Cylinders by Transmission Type",
        legend.text = c("Automatic", "Manual"),
        args.legend = list(title = "Transmission", x = "topright", inset = 0.05))
