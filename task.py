"""
Build a RAG System
1.Choose a book: Find a plain text version of a book that you would like to use as your knowledge base. You can find many free books on websites like Project Gutenberg.
2.Utilize:
  2.1.langchain
  2.2.gemini AI
  2.3.chromadb
  2.4.tiktoken
3.Create a knowledge base: Create a file named book.txt and paste the content of the book into it.
4.Build ya RAG system: Write a Python script that:
  4.1.Loads the book.txt file.
  4.2.Splits the book into smaller chunks.
  4.3.Creates a ChromaDB vector store from the chunks.
  4.4.Creates a retriever.
  4.5.Creates a RetrievalQA chain.
  4.6.Prompts the user to ask a question about the book.
  4.7.Prints the answer to the screen.
5.Enable test option for the application: Ask the application a few questions about the book to make sure it is working correctly."""