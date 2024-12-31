import re

# ----------------------------------- Loading and Preprocessing of Data ---------------------------------- #
def preprocessing(file_name):
  corpus = open(file_name, "r")
  corpus = corpus.read()
  corpus = corpus.lower()
  corpus = (re.findall(r'\w+', corpus))
  return corpus

# ----------------------------------- Creating N-grams ---------------------------------- #
def n_gram(corpus, n):
  ngram_dict = {}
  n_1gram_dict = {}

  for i in range(len(corpus) - n + 1):
    ngram = tuple(corpus[i:i+n])
    if ngram in ngram_dict:
      ngram_dict[ngram] += 1
    else:
      ngram_dict[ngram] = 1
    
    n_1gram = ngram[:-1]
    if n_1gram in n_1gram_dict:
      n_1gram_dict[n_1gram] += 1
    else:
      n_1gram_dict[n_1gram] = 1

  return (ngram_dict, n_1gram_dict)

# ----------------------------------- Calculating Probabilities ---------------------------------- #
def probabilities(ngram, n_1gram, vocab):
  p_dict = {}
  vocab_size = len(vocab)
  for word in ngram:
    n_gram_count = ngram[word]
    n_1 = word[:-1]
    n_1_count = n_1gram[n_1]
    prob = (n_gram_count+1)/(n_1_count+vocab_size)
    p_dict[word] = prob
  return p_dict

# ----------------------------------- Perplexity Calculation on Test Set ---------------------------------- #
def perplexity(nprob_dict, n_1dict, corpus_test, n, vocab_train):
  p = 1
  v = len(vocab_train)
  N = len(corpus_test) - n + 1
  for i in range(len(corpus_test)-n+1):
    ngram = tuple(corpus_test[i:i+n])
    if ngram in nprob_dict:
      p = p *  nprob_dict[ngram]
    else:
      n_1gram = ngram[:-1]
      if n_1gram in n_1dict:
        p = p * (1 / (n_1dict[n_1gram] + v))
      else:
        p = p * (1/v)
  p = 1/p
  p = p ** (1/N)
  return p

# ----------------------------------- main() ---------------------------------- #
def main():
  file_name = "train.txt"
  corpus_train = preprocessing(file_name)
  vocab_train = set(corpus_train)
  file_name = "test.txt"
  corpus_test = preprocessing(file_name)
  vocab_test = set(corpus_test)

  for n in range(1, 4):
    ngram, n_1gram = n_gram(corpus_train, n)
    prob_dict = probabilities(ngram, n_1gram, vocab_train)
    p = perplexity(prob_dict, n_1gram, corpus_test, n, vocab_train)
    if (n == 1):
      print("Unigram: ", p)
    elif (n == 2):
      print("Bigram: ", p)
    elif (n == 3):
      print("Trigram: ", p)

main()