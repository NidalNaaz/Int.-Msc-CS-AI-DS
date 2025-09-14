sentence <- readline("Enter a sentence: ")
shift <- as.integer(readline("Enter shift value: "))

encrypt <- function(char, shift) {
  if (grepl("[A-Z]", char)) {
    return(intToUtf8((utf8ToInt(char) - 65 + shift) %% 26 + 65))
  } else if (grepl("[a-z]", char)) {
    return(intToUtf8((utf8ToInt(char) - 97 + shift) %% 26 + 97))
  } else {
    return(char)
  }
}

encrypted <- paste(sapply(strsplit(sentence, "")[[1]], encrypt, shift), collapse = "")
cat("Encrypted:", encrypted, "\n")


"
Input:
Enter a sentence: Hello World
Enter shift value: 3

Output:
Encrypted: Khoor Zruog
"
