import csv, re, string
import nltk
import numpy as np
from gensim.models import Word2Vec


def preprocess(s):
	s = s.replace('\r', ' ')
	s = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', s)
	stack_trace = s.find("Stack trace:")
	if stack_trace != -1:
		s = s[:stack_trace]
	s = re.sub(r'(\w+)0x\w+', '', s)
	s = s.lower()
	s = nltk.word_tokenize(s)
	s = [word.strip(string.punctuation) for word in s]
	return s



data_tokens = []
data_developer = []

with open('train.csv','r') as training_file:
	trainCSV = csv.reader(training_file)
	row_count = 0
	for row in trainCSV:
		if row_count >= 10:
			break
		
		print row
		if row_count != 0:
			data = preprocess(row[1]) + preprocess(row[2])
			data = filter(None, data)
			data_tokens.append(data)
			data_developer.append(row[0])
		row_count += 1


word2vec_model = Word2Vec(data_tokens, min_count = 5, size = 200, window = 5)
vocabulary = word2vec_model.vocab
