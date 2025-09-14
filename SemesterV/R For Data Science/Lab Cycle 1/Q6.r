isPrime <- function(x) {
  if (x < 2) return(FALSE)
  for (i in 2:floor(sqrt(x))) {
    if (x %% i == 0) return(FALSE)
  }
  return(TRUE)
}

num <- as.integer(readline("Enter a number: "))
if (isPrime(num)) cat("Prime\n") else cat("Not Prime\n")

start <- as.integer(readline("Enter start of range: "))
end <- as.integer(readline("Enter end of range: "))
for (i in start:end) {
  if (isPrime(i)) cat(i, " ")
}
cat("\n")


"
Input:
Enter a number: 17

Output:
Prime
"
