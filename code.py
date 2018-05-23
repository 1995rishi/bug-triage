import csv, re, string
import nltk
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from nltk.corpus import stopwords

#from sklearn.cluster import MeanShift, estimate_bandwidth, DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import accuracy_score, precision_score, recall_score

#import tensorflow as tf

# np.random.seed(1337)


stop_words = stopwords.words('english')
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
	s = [item.lower() for item in s if item.lower() not in stop_words]
	return s



data_tokens = []
data_developer = []

with open('train.csv','r') as training_file:
	# reader = csv.reader(x.replace('\0', '') for x in mycsv)
	trainCSV = csv.reader(x.replace('\0','') for x in training_file)
	row_count = 0
	try:
		for row in trainCSV:
			if row_count >= 10000:
				break
			if row_count != 0:
				data = preprocess(row[1]) + preprocess(row[2])
				data = filter(None, data)
				data_tokens.append(data)
				data_developer.append(row[0])
			row_count += 1
	except Exception as e:
		print(e,row_count)

word2vec_model = Word2Vec(data_tokens, min_count = 5, size = 200, window = 5)
vocabulary = word2vec_model.wv.vocab

total = len(data_developer)
train_length = int(0.75*total)
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

# print len(set(train_owner_unique))

# Extract tf based bag of words representation
tfidf_transformer = TfidfTransformer(use_idf=True,sublinear_tf=True)
count_vect = CountVectorizer(min_df=5, vocabulary= vocab_data,dtype=np.int32)

train_counts = count_vect.fit_transform(train_data)       
train_feats = tfidf_transformer.fit_transform(train_counts)
print train_feats.shape

test_counts = count_vect.transform(test_data)
test_feats = tfidf_transformer.transform(test_counts)
#print train_feats
#print test_feats.shape



################################################################ SVM ########################################################
print "Starting SVM ....."
y_true=[]
y_predicted=[]
classifierModel = svm.SVC(probability=True, verbose=False, decision_function_shape='ovr')
classifierModel.fit(train_feats, updated_train_owner)
predict_prob = classifierModel.predict_proba(test_feats)
classes = classifierModel.classes_ 
# print predict
print "classes = ",len(classes)
k=int(0.05*len(classes))
match = 0
for j,prob in enumerate(predict_prob):
	expected = updated_test_owner[j]
	prob = [ [i,prob[i]] for i in range(len(prob))]
	prob = sorted(prob, reverse = True, key = lambda x: x[1])
	y_predicted.append(classes[prob[0][0]])
	y_true.append(expected)
	for i in range(k):
		c = prob[i][0]
		if classes[c]==expected:
			match+=1
			break
print "accuracy = ", float(match)/float(len(predict_prob))*100
ps = precision_score(y_true,y_predicted, average=None)
s=0
t=0
for p in ps:
	if p != 0.0:
		t+=1
		s+=p
rs =recall_score(y_true,y_predicted, average=None)
sr=0
tr=0
for p in rs:
	if p != 0.0:
		tr+=1
		sr+=p
print "accuracy (top 5%)= ", float(match)/float(len(predict))*100
print "acuracy = ", accuracy_score(y_true,y_predicted)
print "precision = ",float(s)/t 

print "recall = ", float(sr)/tr



##############################################################  Naive Bayes #################################################
print "Starting Naive Bayes....."
y_true=[]
y_predicted=[]
classifierModel = MultinomialNB(alpha=0.01)        
classifierModel = OneVsRestClassifier(classifierModel).fit(train_feats, updated_train_owner)
predict = classifierModel.predict_proba(test_feats)
classes = classifierModel.classes_
print "classes = ",len(classes)
k=int(0.05*len(classes))
match = 0
for j,prob in enumerate(predict):
	expected = updated_test_owner[j]
	prob = [ [i,prob[i]] for i in range(len(prob))]
	prob = sorted(prob, reverse = True, key = lambda x: x[1])
	y_predicted.append(classes[prob[0][0]])
	y_true.append(expected)
	for i in range(k):
		c = prob[i][0]
		if classes[c]==expected:
			match+=1
			break
ps = precision_score(y_true,y_predicted, average=None)
s=0
t=0
for p in ps:
	if p != 0.0:
		t+=1
		s+=p
rs =recall_score(y_true,y_predicted, average=None)
sr=0
tr=0
for p in rs:
	if p != 0.0:
		tr+=1
		sr+=p
print "accuracy (top 5%)= ", float(match)/float(len(predict))*100
print "acuracy = ", accuracy_score(y_true,y_predicted)
print "precision = ",float(s)/t 

print "recall = ", float(sr)/tr




#############################################################  SGD Classification ###########################################
print "Starting SGD....."
y_predicted=[]
y_true=[]
classifierModel = SGDClassifier(loss='log', alpha=0.01)
classifierModel.fit(train_feats, updated_train_owner)
predict_prob = classifierModel.predict_proba(test_feats)
classes = classifierModel.classes_ 
# print predict
print "classes = ",len(classes)
k=int(0.05*len(classes))
match = 0
for j,prob in enumerate(predict_prob):
	expected = updated_test_owner[j]
	prob = [ [i,prob[i]] for i in range(len(prob))]
	prob = sorted(prob, reverse = True, key = lambda x: x[1])
	y_predicted.append(classes[prob[0][0]])
	y_true.append(expected)
	for i in range(k):
		c = prob[i][0]
		if classes[c]==expected:
			match+=1
			break
print "accuracy = ", float(match)/float(len(predict_prob))*100
ps = precision_score(y_true,y_predicted, average=None)
s=0
t=0
for p in ps:
	if p != 0.0:
		t+=1
		s+=p
rs =recall_score(y_true,y_predicted, average=None)
sr=0
tr=0
for p in rs:
	if p != 0.0:
		tr+=1
		sr+=p
print "accuracy (top 5%)= ", float(match)/float(len(predict))*100
print "acuracy = ", accuracy_score(y_true,y_predicted)
print "precision = ",float(s)/t 

print "recall = ", float(sr)/tr


############################################################   MeanShift Clustering #########################################
# print "Starting Meanshift Clustering...."
# bandwidth = estimate_bandwidth(train_feats.toarray(), quantile=0.2, n_samples=500)
# print "estimated bandwidth = ", bandwidth
# ms = MeanShift(bandwidth=bandwidth)
# ms = ms.fit(train_feats.toarray())
# predict = ms.predict(test_feats)
# labels = ms.labels_
# cluster_devs = {}
# for i in range(len(labels)):
# 	try:
# 		cluster_devs[labels[i]].append(updated_train_owner[i])
# 	except:
# 		cluster_devs[labels[i]] = [updated_train_owner[i]]

# match = 0
# print(max(labels))
# for i in range(max(labels)):
# 	print len(set(cluster_devs[i]))
# for i, label in enumerate(predict):
# 	expected = updated_test_owner[i]
# 	if expected in cluster_devs[label]:
# 		match += 1

# print "accuracy = ", float(match)/float(len(predict))*100
# cluster_centers = ms.cluster_centers_
# print len(cluster_centers)


#############################################################  Kmeans clustering  ############################################
# print "Starting Kmeans Clustering...."
# kmeans = KMeans(n_clusters=200, random_state=0).fit(train_feats)
# labels = kmeans.labels_
# predict = kmeans.predict(test_feats)
# cluster_devs = {}
# for i in range(len(labels)):
# 	try:
# 		cluster_devs[labels[i]].append(updated_train_owner[i])
# 	except:
# 		cluster_devs[labels[i]] = [updated_train_owner[i]]


# match = 0
# print(max(labels))
# for i in range(max(labels)):
# 	print len(set(cluster_devs[i]))
# for i, label in enumerate(predict):
# 	expected = updated_test_owner[i]
# 	if expected in cluster_devs[label]:
# 		match += 1

# print "accuracy = ", float(match)/float(len(predict))*100

###########################################################   AgglomerativeClustering ########################################
# print "Starting Agglomerative Clustering...."
# kmeans = AgglomerativeClustering(n_clusters=150).fit(train_feats.toarray())
# labels = kmeans.labels_
# predict = kmeans.fit_predict(train_feats.toarray(), test_feats.toarray())
# cluster_devs = {}
# for i in range(len(labels)):
# 	try:
# 		cluster_devs[labels[i]].append(updated_train_owner[i])
# 	except:
# 		cluster_devs[labels[i]] = [updated_train_owner[i]]


# match = 0
# print(max(labels))
# for i in range(max(labels)):
# 	print len(set(cluster_devs[i]))
# for i, label in enumerate(predict):
# 	expected = updated_test_owner[i]
# 	if expected in cluster_devs[label]:
# 		match += 1

# print "accuracy = ", float(match)/float(len(predict))*100
