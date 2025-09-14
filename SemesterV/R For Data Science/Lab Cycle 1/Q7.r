n <- as.integer(readline("Enter number of terms: "))
fib <- c(0,1,1)
if (n <= 3) {
  cat(fib[1:n], "\n")
} else {
  for (i in 4:n) {
    fib[i] <- fib[i-1] + fib[i-2] + fib[i-3]
  }
  cat(fib, "\n")
}

"
Input:
Enter number of terms: 7

Output:
0 1 1 2 4 7 13
"
