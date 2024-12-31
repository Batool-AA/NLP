import os
import re

# ----------------------------------- Loading and Preprocessing of Data ---------------------------------- #
def extraction(path, n):
    all_reviews = []
    d = os.listdir(path)

    for review_f in d[:n]:
        review_path = os.path.join(path, review_f)
        review = open(review_path, "r", encoding='utf-8')
        review = review.read()
        review = review.lower()
        review = (re.sub(r'[^a-z\s]', "", review))
        review = (re.findall(r'\w+', review))
        all_reviews.append(review)

    return all_reviews

# ----------------------------------- Counting the frequency of each word ---------------------------------- #
def count(pos, neg):
    c = {}

    for p in pos:
        for w in p:
            if w in c:
                c[w][0] += 1
            else:
                c[w] = [1, 0]
    
    for n in neg:
        for w in n:
            if w in c:
                c[w][1] += 1
            else:
                c[w] = [0, 1]
    return c

# ----------------------------------- Calculating the likelihood ---------------------------------- #
def likelihood(total_pos, total_neg, count_dict):
    l = {}
    v = len(count_dict)

    for i in count_dict:
        l_p = (count_dict[i][0] + 1) / (total_pos + v)
        l_n = (count_dict[i][1] + 1) / (total_neg + v)
        l[i] = [l_p, l_n]
    
    return l

# ----------------------------------- Naive Bayes ---------------------------------- #
def nb(corpus, l, prior_n, prior_p):
    out = []

    for review in corpus:
        prob_pos = prior_p
        prob_neg = prior_n

        for word in review:
            if word in l:
                prob_pos *= (l[word][0])  
                prob_neg *= (l[word][1])  

        if prob_pos > prob_neg:
            out.append('pos')
        else:
            out.append('neg')

    return out

# ----------------------------------- Evaluation ---------------------------------- #
def evaluate(p, n):
    tp = p.count("pos")
    tn = n.count("neg")
    fp = n.count("pos")
    fn = p.count("neg")

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
    train_p = extraction("aclImdb/train/pos", 500)
    train_n = extraction("aclImdb/train/neg", 500)
    test_p = extraction("aclImdb/test/pos", 100)
    test_n = extraction("aclImdb/test/neg", 100)

    count_dict = count(train_p, train_n)

    likelihood_train = likelihood(len(train_p), len(train_n), count_dict)

    prior_p = len(train_p) / (len(train_p) + len(train_n))
    prior_n = len(train_n) / (len(train_p) + len(train_n))

    p = nb(test_p, likelihood_train, prior_p, prior_n)
    n = nb(test_n, likelihood_train, prior_p, prior_n)

    accuracy, recall, precision, f1, tp, tn, fp, fn = evaluate(p,n)

    print("Accuracy: ", accuracy, "%\nRecall: ", recall, "\nPrecision: ", precision, "\nF-1: ", f1)
    print("tp: ", tp, "\ntn: ", tn, "\nfp: ", fp, "\nfn: ", fn)

main()