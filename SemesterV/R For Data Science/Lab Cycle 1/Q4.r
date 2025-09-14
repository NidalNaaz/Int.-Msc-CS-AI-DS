len <- as.integer(readline("Enter password length: "))
chars <- c(letters, LETTERS, 0:9, strsplit("!@#$%^&*()", "")[[1]])
password <- paste(sample(chars, len, replace = TRUE), collapse = "")
cat("Generated password:", password, "\n")



"
Input:
Enter password length: 8

Output:
Generated password: aG4@dK2!
"
