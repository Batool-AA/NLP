import re

# ----------------------------------- Input Handling ---------------------------------- #
def input_handling(file_name):
  corpus = open(file_name, "r")
  corpus = corpus.read()
  corpus = corpus.lower()
  corpus = corpus.split()
  return corpus

# ------------------- Tokenization based on words + Frequency of each word ------------ #
def calculate_word_freq(corpus):
  word_freq = {}
  for w in corpus:
      if w in word_freq:
          word_freq[w] += 1
      else:
          word_freq[w] = 1
  return word_freq

# ----------------- Tokenization based on letters + Frequency of each letter ---------- #
def letter_tokenization(word_freq):
  tokenized = []
  letter_freq = {}
  for i in word_freq:
      s = [i[0]] 
      if i[0] in letter_freq:
          letter_freq[i[0]] += word_freq[i]
      else:
          letter_freq[i[0]] = word_freq[i]

      for k in i[1:]:
          s.append('##'+k)

          if ('##'+k) in letter_freq:
              letter_freq['##'+k] += word_freq[i]
          else:
              letter_freq['##'+k] = word_freq[i]

      tokenized.append((s, word_freq[i]))
  return ((tokenized, letter_freq))

# ------------------------------------- Merging --------------------------------------- #
def merge(tokenized, letter_freq, merge_limit):
  merges = 0
  vocab = []

  while merges != merge_limit:
      pairs_freq = {}

      # ------------ Finding pairs + Frequency of each pair ------------------ #
      for t in range(len(tokenized)):
          for u in range(len(tokenized[t][0])-1):
              new_pair = tokenized[t][0][u] + tokenized[t][0][u+1]
              if new_pair not in pairs_freq:
                  #[freq_pair, freq_1, freq_2, [(row, col)]]
                  pairs_freq[new_pair] = [tokenized[t][1], letter_freq[tokenized[t][0][u]], letter_freq[tokenized[t][0][u+1]], [(t,u)]]
              else:
                  pairs_freq[new_pair][0] += tokenized[t][1]
                  pairs_freq[new_pair][3] += [(t,u)]

      # ----------------- Calculating score for each pair -------------------- #
      score_arr = []
      for p in pairs_freq:
          score = pairs_freq[p][0] / (pairs_freq[p][1] * pairs_freq[p][2])
          score_arr.append((p, score))

      score_arr.sort(key=lambda x: x[1], reverse=True)


      # ------------------------- Updating data ------------------------------ #

      vocab.append(score_arr[0][0])
      letter_freq[score_arr[0][0]] = pairs_freq[score_arr[0][0]][0]

      change = pairs_freq[score_arr[0][0]][3]
      for c in change:
          tokenized[c[0]][0][c[1]] = tokenized[c[0]][0][c[1]] + tokenized[c[0]][0][c[1]+1]
          tokenized[c[0]][0].remove(tokenized[c[0]][0][c[1]+1])

      merges += 1
  return vocab

#-------------------------------- Main Function ------------------------------------#

def main():
  file_name = "wordpiece_input.txt"
  corpus = input_handling(file_name)
  word_frequency = calculate_word_freq(corpus)
  tokenized, letter_freq = letter_tokenization(word_frequency)
  vocab = merge(tokenized, letter_freq, 50)

  output_file = open("q2_batool_ba07612.txt", "w", encoding='utf-8')

  output_file.write("Merged Vocabulary:\n\n")

  for i in range(len(vocab)):
    if vocab[i][0] == '#':
      vocab[i] = vocab[i][0:2] + re.sub('#', '', vocab[i][2:])
    else:
      vocab[i] = re.sub('#', '', vocab[i])
    output_file.write(vocab[i]+'\n')


main()