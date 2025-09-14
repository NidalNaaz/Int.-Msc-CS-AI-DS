text <- readline("Enter a paragraph: ")
words <- unlist(strsplit(text, "\\s+"))
word_count <- length(words)
avg_length <- mean(nchar(words))
longest <- words[which.max(nchar(words))]
cat("Total words:", word_count, "\n")
cat("Average word length:", avg_length, "\n")
cat("Longest word:", longest, "\n")
old_word <- readline("Enter word to replace: ")
new_word <- readline("Enter new word: ")
replaced <- gsub(old_word, new_word, text)
cat("Updated paragraph:", replaced, "\n")



"
Input:
Enter a paragraph: This is a simple test paragraph
Enter word to replace: simple
Enter new word: easy

Output:
Total words: 6 
Average word length: 4.666667 
Longest word: paragraph 
Updated paragraph: This is a easy test paragraph
"
