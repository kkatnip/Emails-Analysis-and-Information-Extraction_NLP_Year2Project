import nltk, re
from nltk.corpus import treebank, brown
from os import listdir #listdir Return a list containing the names of the entries in the directory given by path
from os.path import isfile, join
import os

import pprint, pickle
from pprint import pprint
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus.reader import WordListCorpusReader # List of words, one per line.  Blank lines are ignored.
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger, DefaultTagger
from nltk.chunk.regexp import RegexpParser # !!!!
from nltk.tag import pos_tag
from nltk.tag.sequential import ClassifierBasedPOSTagger
import sys, http.client, urllib.request, urllib.parse, urllib.error, json





# load training files
def load_files(path):
	corpus_root = path
	fileIDs = [f for f in listdir(corpus_root) if isfile(join(corpus_root, f))]
	#delete .DS file
	try : 
	    del (fileIDs [fileIDs.index( '.DS_Store' )])
	except :
	    pass

	return fileIDs

def read_data(path, fileIDs):
	corpus = PlaintextCorpusReader(path, fileIDs)
	#print(corpus.raw('0.txt'))#test: get data from a certain file
	#print(corpus.raw('0.txt'))
	return corpus


def taggedDict(taggedText, database):
	regExps = { # 是否要包括简单的直接用regExp得到的？
				'paragraph' : '<paragraph>.*?</paragraph>',
				'sentence'	: '<sentence>.*?</sentence>',
				'stime'		: '<stime>.*?</stime>',
				'etime'		: '<etime>.*?</etime>',
				'location'	: '<location>.*?</location>',
				'speaker'	: '<speaker>.*?</speaker>'
	}

	for key in regExps.keys():
		if(key not in database.keys()):
			database[key] = [removetagsText(str) for str in (re.findall(r'' + regExps[key], taggedText, re.DOTALL))]
		else:
			database[key].extend([removetagsText(str) for str in (re.findall(r'' + regExps[key], taggedText, re.DOTALL))])# FLAG? match new lines

		# print(database[key])

	return database

def removetagsDict(taggedDict):
	clean_dict = dict()
	for key in taggedDict.keys():
		clean_dict[key] = [re.sub('<.*?>', '', elem) for elem in taggedDict[key]]
	return clean_dict

def removetagsText(text):
	clean_txt = re.sub('<.*?>', '', text)
	return clean_txt




def updateDic_count(dictionary, word):
	if word in dictionary:
		dictionary[word] += 1
	else:
		dictionary[word] = 1
	return dictionary


def classify_based_tag_train():
	train_sents = treebank.tagged_sents()[:5000]
	#train_sents = brown.tagged_sents(categories='learned', tagset='universal')
	bigram_tagger = BigramTagger(train_sents)
	cbtagger = ClassifierBasedPOSTagger(train=train_sents, backoff = bigram_tagger)
	pickle.dump(cbtagger, open( 'my_tagger.pkl', 'wb' ) )



# input: ['location str not tokenized', '...']

class train_location_ngram:
    def __init__(self):
        self.location_db = set()
        self.uniword_count = dict()
        self.biword_count = dict()
		#self.uniword_prob = dict()
        self.bi_trans_prob = dict()
        self.bigram_db = set() 
        self.total_word = 0 # total number of words
	
    def train_word_count_prob(self, list_of_locs, threshold = 0.5): #需要加上list_of_locs这个变量吗？
        for phrase in list_of_locs:
            tokenized = word_tokenize(phrase)
            length = len(tokenized)
            self.location_db.add(phrase)


            for i in range(0, length - 2):
                self.uniword_count = updateDic_count(self.uniword_count, tokenized[i])
                self.biword_count = updateDic_count(self.biword_count, tokenized[i] + ' ' + tokenized[i + 1])
                self.total_word += 1
            self.uniword_count = updateDic_count(self.uniword_count, tokenized[length - 1])

		# compute maximum likelihood estimate 
		# if the trans probability is larger than a given threshold, store the bigram in database
        for phrase in self.biword_count.keys():
            words = re.split(' ', phrase)
            trans_prob = self.biword_count[phrase] / self.uniword_count[words[0]] # trans probability approximate
            self.bi_trans_prob[phrase] = trans_prob
            if trans_prob > threshold:
                 self.bigram_db.add(phrase)

def store_location(location_trainer, database):
	location_trainer.train_word_count_prob(database['location'], 0.5) # only store the pharse with a trans_prob above threshold
	return location_trainer

class train_name:
	def __init__(self):
		self.name_db = set()
		self.verb_near_speaker = set()

	def get_namedb(self, database):
		for name in database['speaker']:
			self.name_db.update(set(name.split()))

	def get_verb_speaker(self, trainingIDs, trainingcorpus):
		return 1




def isStoredLoc(pron, loc_stored):
	if pron in loc_stored.location_db:
		return 1
	# loc_stored store the bigram of trained location
	lst_pron = re.split(" ", pron)
	#print(pron)
	#print(lst_pron)
	for i in range(1, len(lst_pron)):
		#print("pron loc:  " + lst_pron[i - 1] + " " + lst_pron[i])
		if (lst_pron[i - 1] + " " + lst_pron[i]) in loc_stored.bigram_db:
			return 1
		else:
			continue

	return 0

# bigram
def findLoc(pron, location_trainer):
	if (isStoredLoc(pron, location_trainer)):
		return 1

	return 0

def isStoredName(name, namedb):
	if name in namedb:
		return 1
	return 0

def findName(pron, namedb):
	name_corpus = list()
	# read in the given document of names
	name_path = '/Users/zhanglingyi/nltk_data/corpora/names.family'
	with open(name_path) as f:
		name_corpus.extend([x.strip('\n') for x in f.readlines()])
	name_path = '/Users/zhanglingyi/nltk_data/corpora/names.female'
	with open(name_path) as f:
		name_corpus.extend([x.strip('\n') for x in f.readlines()])
	name_path = '/Users/zhanglingyi/nltk_data/corpora/names.male'
	with open(name_path) as f:
		name_corpus.extend([x.strip('\n') for x in f.readlines()])

	lst_pron = re.split(" ", pron)
	for name in lst_pron:
		if isStoredName(name, namedb) or (name in name_corpus):
			continue
		else:
			return 0
	return 1

# chunk the (CD)NNP/NN(CD) phrase / regexp speaker / who / prof.
# return a list of phrases (without pos tags)
def chunk_location_sent(pos_text, temp_text):
	list_of_locs = list()

	chunk_grammar = r"""

	LOC:   {((<CD>?<NNP>+<CD>?)|(<CD>?<NN>+<CD>?))+}

	"""
	chunker = RegexpParser(chunk_grammar)


	chunked_article = chunker.parse(pos_text)
	for subtree in chunked_article.subtrees(): 
		if subtree.label()=='LOC':
			#print(' '.join((tuples[0] for tuples in list(subtree))))
			#print(subtree.pprint())
			NNPs = ' '.join((tuples[0] for tuples in list(subtree)))
			#print("LOC: " + NNPs)
			list_of_locs.append(NNPs)
	#print("loc list:", list_of_locs)
	return list_of_locs


def chunk_name_sent(pos_text, temp_text):
	list_of_names = list()

	chunk_grammar = r"""

	NAME: 	{<NNP>+}

	"""
	chunker = RegexpParser(chunk_grammar)


	chunked_article = chunker.parse(pos_text)
	#print("chunk:", chunked_article)
	for subtree in chunked_article.subtrees(): 
		if subtree.label()=='NAME':
			#print(' '.join((tuples[0] for tuples in list(subtree))))
			#print(subtree.pprint())
			NNPs = ' '.join((tuples[0] for tuples in list(subtree)))
			#print("..: ", NNPs)
			#print("LOC: " + NNPs)
			list_of_names.append(NNPs)

	#print("namelist: ", list_of_names)
	return list_of_names

def tag_time(text):
  reg_time2 = "((?<![-]\s)(([0-2][1-9]):([0-5][0-9])\s?([AaPp][Mm])?))|((?<![-])(([0-2][1-9]):([0-5][0-9])\s?([AaPp][Mm])?))"
  reg_time3 = ""
  reg_time1 = "(([0-2][1-9]):([0-5][0-9])\s?([AaPp][Mm])?)\s?[-]\s?([0-2][1-9]):([0-5][0-9])\s?([AaPp][Mm])?)"
  temp_text = text

  find_1time = [t[0] for t in re.findall(r"" + reg_time2, text, re.IGNORECASE)]
  
  #print(find_1time)

  for time in find_1time:

    temp_text = re.sub(time, "<stime>" + time + "</stime>", temp_text)

 
  return temp_text

def remove_first_whitespaces(string):
	return ''.join(string.strip())

def tagHeader(header, header_speaker, header_loc, header_topic):

	if header_loc != '':
		header = re.sub(header_loc, "<location>" + header_loc + "</location>", header)

	if header_speaker != '':
		header = re.sub(header_speaker, "<speaker>" + header_speaker + "</speaker>", header)

	if(re.search(r'(?<=time:\s).*', header, re.IGNORECASE) != "None"):
		header_time = re.findall(r'(?<=time:\s).*', header, re.IGNORECASE)
		if header_time != []:
			header_time = header_time[0]
			header_time = remove_first_whitespaces(header_time)
			if header_time != '':
				header = re.sub(header_time, "<stime>" + header_time + "</stime>", header_time)
	#header = tag_time(header)

	# tag topic

	return header




def tag_sent(sent):
  lst_sent = re.split("(\n)", sent)
  flag = 0
  #print(lst_sent)
  lst2_sent = list()
  # tag the one ends with a . or .?!" & \n and not the '\n' or ' '
  # CHUNK?
  for s in lst_sent:
    flag = 0
    #print("lst: ", lst_sent)

    if s != '':
      ls = s.split(" ")
      #print("ls: ", ls)
      # it's not a sentence if the last word is not a proper noun!!
      if (s[-1] in [".", "?", "!"]) and (pos_tag(ls[-1]) != "NNP" or "NN") :
        #print("s-1: ", s[-1])
        flag = 1
        s = s[:-1] + "</sentence>" + s[-1]
    lst2_sent += s

    if(flag == 1):
    	for i in range(0, len(lst2_sent)):
    		if lst2_sent[i] != '' and lst2_sent[i] != "\n" and lst2_sent[i] != " ":
    			lst2_sent[i] = "<sentence>" + lst2_sent[i]
    			break

  #print(''.join(lst2_sent))

  return ''.join(lst2_sent)

def tag_paragraph(text):
  # tag <sentence> </sentence>\n
  tag_para = text
  find_first_para = re.findall(r"\n\n<sentence>.*?</sentence>", text, re.DOTALL)
  find_back_para = re.findall(r".*?</sentence>.\n\n", text)
  
  #print(find_first_para)
  #print("\n")
  #print(find_back_para)

  for para in find_first_para:
    #print("para: ", para)

    tag_para = re.sub(para, para[:2] + "<paragraph>" + para[2:], tag_para)
  #print(tag_para)

  for para in find_back_para:
    tag_para = re.sub(para, para[:-2] + "</paragraph>" + para[-2:], tag_para)

  #print(tag_para)
  return tag_para



def add_dict(item, dit):
	if item in dit.keys():
		dit[item] += 1
	else:
		dit[item] = 1

	return dit

def tagtext(filename, text, location_trainer, database, name_trainer, extracted_untagged_db, i):
	header = list()
	temp_text = list()
	pron_name_list =list()
	pron_loc_list = list()

	#return a dictionary of extracted entities

	# in case the info is included in header
	header_loc = ''
	header_speaker = ''
	header_topic = ''
	header_type = ''

	sentences = list()
	# pos tagged sents
	pos_text = list()
	# POS tagger
	tagger = pickle.load(open('my_tagger.pkl', 'rb' ) )


	# split 'header'
	splitFirstDoubleReturn = text.split("Abstract:", 1)
	if(text == ""):
		return extracted_untagged_db

	header = splitFirstDoubleReturn[0]
	if len(splitFirstDoubleReturn) > 1:
		temp_text = splitFirstDoubleReturn[1]
		#print(temp_text)

	#print(header)
	# find header info
	if(re.search(r'((?<=speaker:).*)|((?<=who:).*)', header, re.IGNORECASE) != "None"):
		speaker = re.findall(r'((?<=speaker:).*)|((?<=who:).*)', header, re.IGNORECASE)
		#print(speaker)
		if speaker != []:
			header_speaker = speaker[0][1]
			#print(header_speaker)
			header_speaker = remove_first_whitespaces(header_speaker)
			#print(header_speaker)
	if(re.search(r'(?<=where:).*', header, re.IGNORECASE) != "None"):
		loc = re.findall(r'(?<=where:).*', header, re.IGNORECASE)
		if loc != []:
			header_loc = loc[0][1]
			header_loc = remove_first_whitespaces(header_loc)
	if(re.search(r'(?<=topic:).*', header, re.IGNORECASE) != "None"):
		header_topic = re.findall(r'(?<=topic:).*', header, re.IGNORECASE)[0]
		#print("topic : ", header_topic)
	if(re.search(r'(?<=type:).*', header, re.IGNORECASE) != "None"):
		header_type = re.findall(r'(?<=type:).*', header, re.IGNORECASE)[0]
		#print("type:", header_type)
	

	# store topic type & files -> class ontology
	extracted_untagged_db['topicAndType'].append(header_topic + " " + header_type)
	extracted_untagged_db['fileName'].append(filename)

	# tag loc/speaker/time in topic/header
	header = tagHeader(header, header_speaker, header_loc, header_topic)


	if temp_text != []:
		pos_text = tagger.tag(word_tokenize(temp_text))
		print("word tokened: \n", pos_text)

		# tag time
		temp_text = tag_time(temp_text)
		print("time tagged: \n", temp_text)

		pron_loc_list = chunk_location_sent(pos_text, temp_text) 
		pron_name_list = chunk_name_sent(pos_text, temp_text)
		


		name_obvious_list = re.findall(r'(?<=speaker:\s).*', temp_text, re.IGNORECASE)
		print("obvious name: ", name_obvious_list)
		loc__obvious_list = re.findall(r'(?<=where:\s).*', temp_text, re.IGNORECASE)
		if name_obvious_list != ['']:
			for name in name_obvious_list:
				name = remove_first_whitespaces(name)
				temp_text = re.sub(name, "<speaker>" + name + "</speaker>", temp_text)
				print("subed speaker: ", temp_text)
		for loc in loc__obvious_list:
			loc = remove_first_whitespaces(loc)
			if(re.search("[*]", loc) != "None"):
				#print("loc: "+ loc)
				loc = re.sub("[*]", "asterisk", loc)
				temp_text = re.sub(loc, "<location>" + loc + "</location>", temp_text)

				loc = re.sub("asterisk", "[*]", loc)
			else:
				#print("loc: "+ loc)
				temp_text = re.sub(loc, "<location>" + loc + "</location>", temp_text)


		for pron in pron_loc_list: 
			if (loc not in loc__obvious_list):
			# tag location
				if header_loc != '' and pron == header_loc: # appeared in the header
					temp_text = re.sub(pron, "<location>" + pron + "</location>", temp_text)
					continue
				elif findLoc(pron, location_trainer):
					temp_text = re.sub(pron, "<location>" + pron + "</location>", temp_text)

		for  pron in pron_name_list:
			if (pron not in pron_loc_list) and (pron not in name_obvious_list):
				# tag speaker
				if header_speaker != '' and pron == header_speaker: # appeared in the header
					temp_text = re.sub(pron, "<speaker>" + pron + "</speaker>", temp_text)
					continue
				if findName(pron, name_trainer.name_db):
					temp_text = re.sub(pron, "<speaker>" + pron + "</speaker>", temp_text)

		# tag sentence
		sentences = re.split(r"((?<!\w\.\w.)(?<![A-Z][a-z]\.)((?<=\.|\?|\!)|(?<=\n\n))\s)|((?<=\n\n)\s)", temp_text, re.DOTALL)
		#print("sent split: ", sentences)

		taggedSents = list()
		#removeReturnSents = [re.sub(r'\n*', '', )]
		#pos_sents = ([tagger.tag(word_tokenize(sent)) for sent in sentences])
		#print("sent: ", sentences)
		#print("pos sent: ", pos_sents)

		for sent in sentences:
			if(sent != None):
				sent = tag_sent(sent)

				taggedSents += sent
			#print("stnL ", sent)

		joinedTextBody = ''.join(taggedSents)
		#print("tagsent: ", taggedSents)
		# tag paragraph
		if(re.search("[*]", joinedTextBody) != "None"):
			#print("loc: "+ loc)
			joinedTextBody = re.sub("[*]", "asterisk", joinedTextBody)
			joinedTextBody = tag_paragraph(joinedTextBody)

			joinedTextBody = re.sub("asterisk", "[*]", joinedTextBody)
		else:
			joinedTextBody = tag_paragraph(joinedTextBody)

		taggedText = header + 'Abstract:' + joinedTextBody

		#print(taggedText)
		# update extracted untag db
		extracted_untagged_db = taggedDict(taggedText, extracted_untagged_db)
		print(taggedText)

		with open('/Users/zhanglingyi/nltk_data/corpora/newdir/' + str(i) + '.txt', 'wt') as f:
			f.write(taggedText)
			f.close()


		# write text


	# header topic, tag time and obvious info (after joined text) because split sentence might affect tagging between 'sents'
		#temp_text = re.	SETIME:	{<CD><:><CD>(pm|PM|am|AM)?}SETIME:	{<CD>\s?<:>\s?<CD>\s?(pm|PM|am|AM)?}
		#temp_text = re.
		#temp_text = re

		# tag paragraph

	#change!!!!!!!!!!!!!!!
	return extracted_untagged_db


def evaluation(extracted_untagged_db, database):

	for key in database.keys():
		fn = 0.0
		tp = 0.0
		fp = 0.0
		for i in extracted_untagged_db[key]:
			if i in database[key]: # tp
				tp += 1.0
			else:
				#fp
				fp += 1.0
		if tp + fp != 0:
			precision = tp / (tp + fp)
		else:
			precision = 0.0
		recall = tp / len(database[key])
		print(key + " precision: ", precision)
		print(key + " recall: ", precision)
	
	return 0

class ontology():

	def __init__(self):
		self.ontology = dict()


	def read_train(self, database):
		return self

def train_keywords():
	return 0

def extract_keywords():
	# stem

	# filter stop words

	return 0


def main():
	
	database = dict()

	# training files 
	# store untagged text in database
	train_path = '/Users/zhanglingyi/nltk_data/corpora/test_tagged/'
	trainingIDs = load_files(train_path)
	trainingcorpus = read_data(train_path, trainingIDs)

	untagged_path = '/Users/zhanglingyi/nltk_data/corpora/test_untagged/'
	untaggedIDs = load_files(untagged_path)
	untaggedcorpus = read_data(untagged_path, untaggedIDs)

	#train POS tagger
	classify_based_tag_train()

	for f in trainingIDs:
		database = taggedDict(trainingcorpus.raw(f), database)
		#train_topic += [removetagsText(str) for str in re.findall(r"Topic.*", trainingcorpus.raw(f))]


	#print(database, "\n\n")

	#print(train_topic)

	location_trainer = train_location_ngram()
	location_trainer = store_location(location_trainer, database)
	print(location_trainer.bigram_db)

	name_trainer = train_name()
	name_trainer.get_namedb(database)
	#print(name_trainer.name_db)


	extracted_untagged_db = {'paragraph' : [], 'sentence' : [], 'stime' : [], 'etime' : [], 'location' :[], 'speaker' : [], 'topicAndType' : [], 'fileName' : []}

	#tagtext(untaggedcorpus.raw('390.txt'), location_trainer, database)
	#extracted_untagged_db = tagtext(untaggedcorpus.raw('390.txt'), location_trainer, database, name_trainer, extracted_untagged_db)
	# tag untagged text
	i = 0
	#for f in untaggedIDs:
	for f in ['481.txt',  '364.txt']:
		extracted_untagged_db =  tagtext(f, untaggedcorpus.raw(f), location_trainer, database, name_trainer, extracted_untagged_db, i)
		i += 1
		print(f)
	#for f in trainingIDs:
		#taggedText = tagtext(trainingcorpus.raw(f), location_trainer, database)
		# store the tagged text into a new file
	#evaluation(extracted_untagged_db, database)
	#print(extracted_untagged_db['topicAndType'], "\n\n")
	#print(extracted_untagged_db['fileName'])






main()




text = """<0.29.11.93.15.37.25.tpn+@andrew.cmu.edu.0>
Type:     cmu.andrew.org.cmu-hci
Topic:    HCI Seminar - Wed. Dec. 1 '93 Tom Neuendorffer ITC/CS/CMU
Dates:    1-Dec-93
Time:     <stime>3:30</stime> - <etime>5:00 PM</etime>
PostedBy: Tom Neuendorffer on 29-Nov-93 at 15:37 from andrew.cmu.edu
Abstract: 

                               HCI Seminar

                       Wednesday, December 1, 1993
                               <stime>3:30</stime>-<etime>5:00pm</etime>
                             <location>Wean Hall 5409</location>
               Extending an Embedded Object Architecture 
                      to Support Collaborative Work

                             <speaker>Tom Neuendorffer</speaker>
                       Carnegie Mellon University
                      Information Technology Center
                       School of Computer Science

<paragraph><sentence>For several years, the notion of User Interfaces that support embedded
objects has been gaining popularity, culminating in the recent OLE work
from Microsoft and the OpenDoc architecture from CIL</sentence>. <sentence>These are systems
where active objects (text , spreadsheets, pictures, etc. ) are not
simply isolated applications, but are components that can include one
another, allowing both the users and application creators to develop
multi-media documents that can present information in a variety of ways</sentence>.
<sentence>This talk will discuss some attempts to extend an embedded object system
to support collaborative editing, focusing on the system features that
lend themselves to producing groupware, and those that make groupware
difficult</sentence>. <sentence>The goals for this system include not only interactive
document creation, but also interactive application creation and the
ability to give remote software demonstrations</sentence>. <sentence>Included in this talk
will be a discussion of early work using a single process model (as in
the Argo system), and more recent work which supports multiple processes
that share data through a  back-end which is both UI and system
independent</sentence>. <sentence>A file-based interface to this back-end has been
implemented, and an interface based on the HyperDesk Distributed Object
Management System is currently in the works at Mitre, who has been
supporting this recent work</sentence>.</paragraph>

<paragraph><sentence>If time is available at the end of the talk, those who are interested
may stick around for a brief postmortem of the ITC,  including opinions
on where it succeeded,  where it failed, and the lessons learned as they
might apply to other projects within SCS</sentence>.</paragraph>
"""




untaggedtxt = """<0.29.11.93.15.37.25.tpn+@andrew.cmu.edu.0>
Type:     cmu.andrew.org.cmu-hci
Topic:    HCI Seminar - Wed. Dec. 1 '93 Tom Neuendorffer ITC/CS/CMU
Dates:    1-Dec-93
Time:     3:30 - 5:00 PM
PostedBy: Tom Neuendorffer on 29-Nov-93 at 15:37 from andrew.cmu.edu
Abstract: 

                               HCI Seminar

                       Wednesday, December 1, 1993
                               3:30-5:00pm
                             Wean Hall 5409
               Extending an Embedded Object Architecture 
                      to Support Collaborative Work

                             Tom Neuendorffer
                       Carnegie Mellon University
                      Information Technology Center
                       School of Computer Science

For several years, the notion of User Interfaces that support embedded
objects has been gaining popularity, culminating in the recent OLE work
from Microsoft and the OpenDoc architecture from CIL. These are systems
where active objects (text , spreadsheets, pictures, etc. ) are not
simply isolated applications, but are components that can include one
another, allowing both the users and application creators to develop
multi-media documents that can present information in a variety of ways.
This talk will discuss some attempts to extend an embedded object system
to support collaborative editing, focusing on the system features that
lend themselves to producing groupware, and those that make groupware
difficult. The goals for this system include not only interactive
document creation, but also interactive application creation and the
ability to give remote software demonstrations. Included in this talk
will be a discussion of early work using a single process model (as in
the Argo system), and more recent work which supports multiple processes
that share data through a  back-end which is both UI and system
independent. A file-based interface to this back-end has been
implemented, and an interface based on the HyperDesk Distributed Object
Management System is currently in the works at Mitre, who has been
supporting this recent work.

If time is available at the end of the talk, those who are interested
may stick around for a brief postmortem of the ITC,  including opinions
on where it succeeded,  where it failed, and the lessons learned as they
might apply to other projects within SCS.
"""















