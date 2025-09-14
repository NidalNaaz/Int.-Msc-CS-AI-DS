n <- as.integer(readline("Enter number of elements: "))
lst <- c()
for (i in 1:n) {
  lst[i] <- readline("Enter element: ")
}
rev_lst <- c()
for (i in 1:n) {
  rev_lst[i] <- lst[n - i + 1]
}
cat("Reversed list:", rev_lst, "\n")


"
Input:
Enter numbers separated by space: 1 2 3 4 5

Output:
Reversed list: 5 4 3 2 1
"
