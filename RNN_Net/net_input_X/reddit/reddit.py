import csv
import itertools
import numpy as np
import nltk


f_loc = '/Users/martian2049/Desktop/NN:AI/RNN/RNN_Net/net_input_X/reddit/'
f_names = ['small.csv','test.csv','tiny.csv','reddit-comments-2015-08.csv']
f_name = f_loc + f_names[2]


vocabulary_size = 30
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"



def get_data_info(verbose=False, veryverbose=False):
	with open(f_name, newline='\n') as csvfile:
	    csvreader = csv.reader(csvfile)
	    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in csvreader])
	    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
	    if veryverbose: print('sentences[:3]\n: ',sentences[:3])
	    if verbose: print('# of Parsed sentences: ', (len(sentences)))
	tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
	if veryverbose: print('\ntokenized_sentences[0:2][0:4]: ',tokenized_sentences[0:2][0:4]) 

	word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
	if verbose: print ("\nFound %d unique words tokens." % len(word_freq.items()))

	# Get the most common words and build index_to_word and word_to_index vectors
	vocab = word_freq.most_common(vocabulary_size-1)
	if veryverbose: print("\nExample of vocab is: ",vocab[:3])
	index_to_word = [x[0] for x in vocab]
	index_to_word.append(unknown_token) #print(index_to_word) list
	word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)]) #print(word_to_index) dict
	if veryverbose: print("\nExample of index_to_word is: ",index_to_word[3:6])

	# Replace all words not in our vocabulary with the unknown token
	for i, sent in enumerate(tokenized_sentences):
	    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

	if veryverbose: print ("\nExample sentence: '%s'" % sentences[0])
	if veryverbose: print ("\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0])

	# Create the training data
	X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
	y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

	return X_train,y_train,vocabulary_size,index_to_word,word_to_index
















