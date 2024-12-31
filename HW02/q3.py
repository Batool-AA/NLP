import csv
import numpy as np
import re

# ----------------------------------- Loading and Preprocessing of Data ---------------------------------- #
def preprocessing(file_name):
  vocab = {}
  data = []

  with open(file_name, mode='r', newline='', encoding='utf-8') as file:
    corpus = csv.reader(file)
    next(corpus)

    for row in corpus:
      text = row[0]
      text = text.lower()
      text = (re.sub(r'[^a-z0-9\s]', "", text))
      words = text.split()
      data.append((words, row[1]))
      #indexing
      for w in words:
        if w not in vocab:
          vocab[w] = len(vocab)

  return (vocab, data)

# ----------------------------------- One Hot Encoding ---------------------------------- #
def encoding(vocab, data):
  encoded = []
  labels = []
  v = len(vocab)

  for sentence, label in data:
    t = np.zeros(v)
    for word in sentence:
        if word in vocab:
          t[vocab[word]] = 1
    encoded.append(t)
    labels.append(int(label))
  labels = np.array(labels)
  labels = labels.reshape(-1, 1)
  return (np.array(encoded), labels)

# ----------------------------------- Activation Function ---------------------------------- #
def sigmoid(x):
   return 1 / (1 + np.exp(-x))

# ----------------------------------- Forward Pass ---------------------------------- #
def forward(input, weight, bias):
   z = np.dot(input, weight) + bias
   a = sigmoid(z)
   return a

# ----------------------------------- Back Propagation ---------------------------------- #
def backward(truth, encoding, ih, ho, w_ih, w_ho, b_h, b_o, l):
   output_error = ho * (1 - ho) * (truth - ho)
   hidden_error = ih * (1 - ih) * (output_error * w_ho.T)

   w_ho_updated = w_ho + (l * np.dot(ih.T, output_error))
   w_ih_updated = w_ih + (l * np.dot(encoding.T, hidden_error))

   b_o_updated = b_o + l * np.sum(output_error, axis=0, keepdims=True)  
   b_h_updated = b_h + l * np.sum(hidden_error, axis=0, keepdims=True)  

   return (w_ho_updated, w_ih_updated, b_h_updated, b_o_updated)

# ----------------------------------- Initialize ---------------------------------- #
def initialize(input_size, hidden_size, output_size):
   w_ih = np.random.randn(input_size, hidden_size)
   b_h =  np.zeros((1, hidden_size))
   w_ho = np.random.randn(hidden_size, output_size)
   b_o = np.zeros((1, output_size))
   return (w_ih, b_h, w_ho, b_o)

# ----------------------------------- Training ---------------------------------- #
def train(encoded, labels, epoch, learning_rate, w_ih, b_h, w_ho, b_o):
    for _ in range(epoch):
        ih = forward(encoded, w_ih, b_h)
        ho = forward(ih, w_ho, b_o)
        w_ho, w_ih, b_h, b_o = backward(labels, encoded, ih, ho, w_ih, w_ho, b_h, b_o, learning_rate)
    
    return w_ho, w_ih, b_h, b_o

# ----------------------------------- Testing ---------------------------------- #
def predict(encoding, w_ih, b_h, w_ho, b_o):
   output_ih = forward(encoding, w_ih, b_h)
   output_final = forward(output_ih, w_ho, b_o)
   return output_final

# ----------------------------------- Evaluation ---------------------------------- #
def evaluate(predicted, truth):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range (len(truth)):
        if truth[i] == 1 and predicted[i] >= 0.5:
           tp += 1
        elif truth[i] == 1 and predicted[i] < 0.5:
           fn += 1
        elif truth[i] == 0 and predicted[i] >= 0.5:
           fp += 1
        elif truth[i] == 0 and predicted[i] < 0.5:
           tn += 1

    a = (tp+tn) / (tp+tn+fp+fn)
    a = a * 100

    if (tp + fn) == 0:
       recall = 0
    else:
        recall = tp / (tp+fn)

    if (tp+fp) == 0:
       precision = 0
    else:
        precision = tp / (tp+fp)

    if (precision+recall) == 0:
      f1 = 0
    else:
      f1 = (2*precision*recall) / (precision+recall)

    return (a, recall, precision, f1, tp, tn, fp, fn)

# ----------------------------------- main() ---------------------------------- #
def main():
  file_name = "sentiment_train_dataset.csv"
  vocab_index_train, data_train = preprocessing(file_name)
  encoded_train, labels_train = encoding(vocab_index_train, data_train)

  input_size = encoded_train.shape[1]  
  hidden_size = 128
  output_size = 1
  learning_rate = 0.01
  epoch = 10

  w_ih, b_h, w_ho, b_o = initialize(input_size, hidden_size, output_size)
  w_ho, w_ih, b_h, b_o = train(encoded_train, labels_train, epoch, learning_rate, w_ih, b_h, w_ho, b_o)

  file_name = "sentiment_test_dataset.csv"
  vocab_index_test, data_test = preprocessing(file_name)
  encoded_test, labels_test = encoding(vocab_index_test, data_test)

  predicted = predict(encoded_test, w_ih, b_h, w_ho, b_o)
  accuracy, recall, precision, f1, tp, tn, fp, fn = evaluate(predicted, labels_test)
  print("Accuracy: ", accuracy, "%\nRecall: ", recall, "\nPrecision: ", precision, "\nF-1: ", f1)
  print("tp: ", tp, "\ntn: ", tn, "\nfp: ", fp, "\nfn: ", fn)
  
main()