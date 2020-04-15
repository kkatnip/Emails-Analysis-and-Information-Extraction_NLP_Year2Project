import nltk, re
from nltk.corpus import treebank
from os import listdir #listdir Return a list containing the names of the entries in the directory given by path
from os.path import isfile, join
import os

from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus.reader import WordListCorpusReader # List of words, one per line.  Blank lines are ignored.
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger, DefaultTagger
from nltk import word_tokenize

from nltk.chunk.regexp import ChunkString, ChunkRule, ChinkRule, RegexpParser
from nltk.tree import Tree

# load training files
def load_training_files(train_path):
	corpus_root = train_path
	trainingfileIDs = [f for f in listdir(corpus_root) if isfile(join(corpus_root, f))]
	#delete .DS file
	try : 
	    del (trainingfileIDs [trainingfileIDs.index( '.DS_Store' )])
	except :
	    pass

	return trainingfileIDs

def read_training_data(train_path, trainingfileIDs):
	corpus = WordListCorpusReader(train_path, trainingfileIDs)
	#print(corpus.raw('0.txt'))#test: get data from a certain file
	#print(word_tokenize(corpus.raw('0.txt')))
	return corpus


def clean_inner_tags(list_of_sents):
	clean_list = []
	for sent in list_of_sents:
		sent = re.sub('<.*?>', '', sent)
		clean_list.append(sent)
	return clean_list

#tokenize a list of sentences 
def tokenize_list_of_sents(list_of_sents):
	list_of_sents_clean = clean_inner_tags(list_of_sents)
	sents = [word_tokenize(sent) for sent in list_of_sents_clean] 
	return sents

"""
layer_tagger uses treebank corpora to train train_sents
use default, unigram, bigram taggers as backoff
too slow. dump
"""
def layer_tagger():
	train_sents = treebank.tagged_sents()[:5000]

	t0 = DefaultTagger('NN')
	layerTagger = UnigramTagger(train_sents, backoff=t0)
	
	return layerTagger

# tag each sentence. take in lists. tokenize then tag with layer tagger.
def tag_training_data(tokenized_sents):
	layerTagger = layer_tagger()
	tagged_list = [layerTagger.tag(sent) for sent in tokenized_sents]
	#print("tagged list", tagged_list)
	return tagged_list


# save names in a list 
def get_name(list_of_names):
	return list_of_names


# pattern matching via given tags. return list containing eneities of each catagory
def pattern_matching_training_data(trainingfileIDs, corpus, paragraph_list, sentence_list, time_list, location_list, speaker_list):
	for f in trainingfileIDs:
		data = corpus.raw('0.txt')
		#print(data)

		#sentence 
		sentences = re.findall( r'<sentence>.*?</sentence>', data, re.M)
		if sentences != []:
			for sent in sentences:
				if sent not in sentence_list:
					sentence_toke = tokenize_list_of_sents([sent])
					sentence_list.extend(tag_training_data(sentence_toke))
			#print ("sentence: ", sentences)
		#time #use regExp
		times = re.findall( r'<stime>.*?</stime>', data, re.M)

		#paragraph  #use regExp
		paragraphs = re.findall( r'<paragraph>.*?</paragraph>', data, re.M)
		if paragraphs != []:
			paragraph_list.extend(paragraphs)


		#location #use regExp and tagger
		locations = re.findall( r'<location>.*?</location>', data, re.M)
		if locations != []:
			for sent in locations:
				if sent not in location_list:
					location_toke = tokenize_list_of_sents([sent])
					location_list.extend(tag_training_data(location_toke))

		#speaker
		speakers = re.findall( r'<speaker>.*?</speaker>', data, re.M)
		if speakers != []:
			speaker_list.extend(get_name(speakers))

		#test first text
		break




"""execute"""
path = '/Users/zhanglingyi/nltk_data/corpora/training/'
trainingIDs = load_training_files(path)
trainingcorpus = read_training_data(path, trainingIDs)

#set paragraph, sentence, time, location, speaker list
paragraph_list = []
sentence_list = []
time_list = []
location_list = []
speaker_list = []

pattern_matching_training_data(trainingIDs, trainingcorpus, paragraph_list, sentence_list, time_list, location_list, speaker_list)

#print("tokenize:  ", tokenize_list_of_sents(sentence_list))
print("the extract lists: ", paragraph_list, "\n\n", sentence_list, "\n\n", time_list, "\n\n", location_list, "\n\n", speaker_list)
layerTagger = layer_tagger()
#print(layerTagger.tag(['Lectures', 'are', 'presented', 'on', 'Tuesdays', 'at', '5:00', 'p.m.', 'in', 'DH', '2315']))

#define grammar
chunker = RegexpParser(r'''
 NP:
 
 ''')

#take sentencelist as an example. 
for sent in sentence_list:
	sent_tree = Tree('SENTENCE', sent)
	st = ChunkString(sent_tree)
	print("st: ", st)
	chunker.parse(sent)




















