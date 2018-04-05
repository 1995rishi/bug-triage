import csv, re, string
import nltk
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
np.random.seed(1337)

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
	# reader = csv.reader(x.replace('\0', '') for x in mycsv)
	trainCSV = csv.reader(x.replace('\0','') for x in training_file)
	row_count = 0
	try:
		for row in trainCSV:
			if row_count >= 1000:
				break
			if row_count != 0:
				data = preprocess(row[1]) + preprocess(row[2])
				data = filter(None, data)
				data_tokens.append(data)
				data_developer.append(row[0])
			row_count += 1
	except:
		print(row_count)

word2vec_model = Word2Vec(data_tokens, min_count = 5, size = 200, window = 5)
vocabulary = word2vec_model.wv.vocab

total = len(data_developer)
train_length = int(0.8*total)
data_developer_train = data_developer[0:train_length+1]
data_tokens_train = data_tokens[0:train_length+1]
data_developer_test = data_developer[train_length:total]
data_tokens_test = data_tokens[train_length:total]

# Remove words outside the vocabulary
updated_train_data = []    
updated_train_data_length = []    
updated_train_owner = []
final_test_data = []
final_test_owner = []
for j, item in enumerate(data_tokens_train):
	current_train_filter = [word for word in item if word in vocabulary]
	if len(current_train_filter)>=20:  
	  updated_train_data.append(current_train_filter)
	  updated_train_owner.append(data_developer_train[j])  
	  
for j, item in enumerate(data_tokens_test):
	current_test_filter = [word for word in item if word in vocabulary]  
	if len(current_test_filter)>=20:
	  final_test_data.append(current_test_filter)    	  
	  final_test_owner.append(data_developer_test[j])    	  

# Remove data from test set that is not there in train set
train_owner_unique = set(updated_train_owner)
test_owner_unique = set(final_test_owner)
unwanted_owner = list(test_owner_unique - train_owner_unique)
updated_test_data = []
updated_test_owner = []
updated_test_data_length = []
for j in range(len(final_test_owner)):
	if final_test_owner[j] not in unwanted_owner:
		updated_test_data.append(final_test_data[j])
		updated_test_owner.append(final_test_owner[j])  

train_data = []
for item in updated_train_data:
	  train_data.append(' '.join(item))
     
test_data = []
for item in updated_test_data:
	  test_data.append(' '.join(item))

vocab_data = []
for item in vocabulary:
	  vocab_data.append(item)

# print len(test_data)

# Extract tf based bag of words representation
tfidf_transformer = TfidfTransformer(use_idf=False)
count_vect = CountVectorizer(min_df=1, vocabulary= vocab_data,dtype=np.int32)

train_counts = count_vect.fit_transform(train_data)       
train_feats = tfidf_transformer.fit_transform(train_counts)
#print train_feats.shape

test_counts = count_vect.transform(test_data)
test_feats = tfidf_transformer.transform(test_counts)
#print train_feats
#print test_feats.shape



################################################################ SVM ########################################################
print "Starting SVM ....."
classifierModel = svm.SVC(probability=True, verbose=False, decision_function_shape='ovr')
classifierModel.fit(train_feats, updated_train_owner)
predict = classifierModel.predict(test_feats)
predict_prob = classifierModel.predict_proba(test_feats)
classes = classifierModel.classes_ 
# print predict

match = 0
for i in range(len(predict_prob)):
	expected = updated_train_owner[i]
	for j in range(len(classes)):
		if predict_prob[i][j]<mx:
			mx = predict_prob[i][j]
			mi = j
	if classes[mi] == updated_train_owner[i]:
		match += 1
print "accuracy = ", float(match)/float(len(predict))*100

match = 0
for i in range(len(predict)):
	if predict[i] == updated_test_owner[i]:
		match += 1
print "accuracy = ", float(match)/float(len(predict))*100


##############################################################  Naive Bayes #################################################
'''print "Starting Naive Bayes....."
classifierModel = MultinomialNB(alpha=0.01)        
classifierModel = OneVsRestClassifier(classifierModel).fit(train_feats, updated_train_owner)
predict = classifierModel.predict_proba(test_feats)  
classes = classifierModel.classes_
match = 0
print predict
for i in range(len(predict)):
	if predict[i] == updated_test_owner[i]:
		match += 1
print "accuracy = ", float(match)/float(len(predict))*100'''

