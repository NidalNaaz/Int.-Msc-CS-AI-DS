n <- as.integer(readline("Enter number of terms: "))
sum <- 0
sign <- 1
for (i in 1:n) {
  term <- i / (2*i - 1)
  sum <- sum + sign * term
  sign <- sign * -1
}
cat("Sum of series:", sum, "\n")


"
Input:
Enter number of terms: 5

Output:
Sum of series: 0.7873016
"
