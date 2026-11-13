# Deep Dive: Tokenization and Data Curation

## 1. The Tokenization Bottleneck
Neural networks cannot read text. They can only process arrays of numbers. **Tokenization** is the process of chopping text into chunks and assigning each chunk a number.

### 1.1 Byte-Pair Encoding (BPE)
The standard algorithm used by GPT-4 and Llama-3 is **BPE**. 
BPE starts by making every individual character (a, b, c) a token. It then scans the dataset and merges the most frequently adjacent characters.
*   "t" + "h" $\rightarrow$ "th"
*   "th" + "e" $\rightarrow$ "the"
This repeats until you have a vocabulary of ~100,000 common word chunks.

### 1.2 Why AI Cannot Spell "Strawberry"
If you ask an LLM "How many 'r's are in strawberry?", it often answers "2". Why?
Because the BPE tokenizer chunked the word into `['straw', 'berry']`. The LLM has *never seen the individual letters*. It only sees the two mathematical embeddings for those two chunks. Asking it to count the letters is like asking a human to count the number of atoms in a brick.

### 1.3 The Future: Byte-Level Models
Tokenizers cause massive problems for coding, multilinguality, and arithmetic. A major area of research is **Byte Latent Transformers (BLT)**, which delete the tokenizer entirely and feed raw raw bytes (0s and 1s) directly into the model, forcing the network to learn the concept of letters and numbers organically.

---

## 2. Data Curation: The Most Important Step
In 2020, researchers believed "more data is better," scraping the entire internet without filtering it. In 2023, the Chinchilla and Llama papers proved that **Data Quality matters far more than Data Quantity**.

If you train a 70B model on 1 Trillion tokens of high-quality textbooks, it will completely crush a 70B model trained on 3 Trillion tokens of Reddit comments.

### 2.1 The Data Pipeline
Modern datasets (like FineWeb or DCLM) go through brutal filtering:
1.  **Deduplication:** Removing repeated text. If a model sees the same sentence 50 times, it memorizes it and overfits.
2.  **MinHash / LSH:** Algorithms to find *fuzzy* duplicates (web pages that are 90% identical).
3.  **Quality Classifiers:** Training a small, smart model to grade the "educational value" of a web page from 1 to 5. Any page scoring below a 4 is deleted before the main model ever sees it. 
4.  **Synthetic Data:** The internet is running out of high-quality human text. The future is using models like GPT-4 to generate perfectly clean, step-by-step reasoning textbooks to train smaller models.

---

## 3. Implementation Details: The MinHash Deduplication Pipeline

We have provided a fully working implementation of the fuzzy-deduplication algorithm required to clean massive datasets:
*   **See Code:** [29_Dataset_Curation_MinHash.py](../../Code/Deep_Dives/29_Dataset_Curation_MinHash.py)

### How the Code Works
If you have 10 Billion documents, you cannot compare each document to every other document (that is $O(N^2)$ and would take hundreds of years). We use **MinHash** to approximate similarity in linear time.
1.  **Shingling:** The code splits a document into overlapping word chunks (e.g., "the cat sat").
2.  **Hashing:** It passes these chunks through 50 randomized hash functions.
3.  **The Signature:** For each hash function, it records only the *minimum* integer value. This reduces a 10,000-word document down to a simple list of 50 integers (the "signature").
4.  **Jaccard Similarity:** By mathematically comparing the two 50-integer signatures, the algorithm calculates the exact overlap (similarity) of the two original documents. If it crosses a threshold (e.g., 70%), the document is permanently deleted from the training set.
