n <- as.integer(readline("Enter number of students: "))
ages <- c()
for (i in 1:n) {
  name <- readline("Enter name: ")
  age <- as.integer(readline("Enter age: "))
  grade <- readline("Enter grade (A/B/C/D/F): ")
  if (age > 0 && grade %in% c("A","B","C","D","F")) {
    ages <- c(ages, age)
  }
}
if (length(ages) > 0) {
  cat("Average age:", mean(ages), "\n")
} else {
  cat("No valid records\n")
}


"
Input:
Enter number of students: 3
Enter name: Anna
Enter age: 20
Enter grade (A/B/C/D/F): A
Enter name: Ben
Enter age: -3
Enter grade (A/B/C/D/F): B
Enter name: Carl
Enter age: 21
Enter grade (A/B/C/D/F): C

Output:
Average age: 20.5
"

