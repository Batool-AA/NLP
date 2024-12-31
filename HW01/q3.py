
corpus = open("urdu_text_input.txt", "r", encoding='utf-8')
corpus = corpus.read()

#-------------------------------- Using Regular Expressions -----------------------#
import re

word_tokens_re = set(re.findall(r'\w+', corpus)) #word tokenization
letter_tokens_re = set(re.findall(r'\w', corpus)) #letter tokenization

output_file = open("q3_batool_ba07612.txt", "w", encoding='utf-8')

output_file.write("Using Regular Expression\n\n")
word_tokens_str = "\n".join(word_tokens_re)
output_file.write("Word Based\n")
output_file.write(word_tokens_str)
letter_tokens_str = "\n".join(letter_tokens_re)
output_file.write("\n\nLetter Based\n")
output_file.write(letter_tokens_str)
output_file.write("\n\n\n")

#----------------------------- Using nltk Library ----------------------------#
from nltk.tokenize import word_tokenize 
word_tokens_nltk = set(word_tokenize(corpus))
output_file.write("Using nltk Library\n")
output_file.write("\n".join(word_tokens_nltk))
output_file.write("\n\n\n")

#---------------------------- Using split() Method ---------------------------#

word_tokens_split = set(corpus.split())
output_file.write("Using split() Method\n")
output_file.write("\n".join(word_tokens_split))
