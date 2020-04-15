import nltk, re
from nltk.corpus import treebank, brown
from os import listdir #listdir Return a list containing the names of the entries in the directory given by path
from os.path import isfile, join
import os
from nltk.corpus import stopwords
import pickle
from pprint import pprint
from nltk.tokenize import word_tokenize
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus.reader import WordListCorpusReader # List of words, one per line.  Blank lines are ignored.
from nltk.tag import UnigramTagger, BigramTagger
from nltk.chunk.regexp import RegexpParser # !!!!
from nltk.tag import pos_tag
from nltk.tag.sequential import ClassifierBasedPOSTagger
import sys, http.client, urllib.request, urllib.parse, urllib.error, json
import gensim
from gensim.models import Word2Vec




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
	return corpus


def removetagsDict(taggedDict):
	clean_dict = dict()
	for key in taggedDict.keys():
		clean_dict[key] = [re.sub('<.*?>', '', elem) for elem in taggedDict[key]]
	return clean_dict

def removetagsText(text):
	clean_txt = re.sub('<.*?>', '', text)
	return clean_txt

def classify_based_tag_train():
	train_sents = treebank.tagged_sents()[:5000]

	bigram_tagger = BigramTagger(train_sents)
	cbtagger = ClassifierBasedPOSTagger(train=train_sents, backoff = bigram_tagger)
	pickle.dump(cbtagger, open( 'my_tagger.pkl', 'wb' ) )

def removetagsDict(taggedDict):
	clean_dict = dict()
	for key in taggedDict.keys():
		clean_dict[key] = [re.sub('<.*?>', '', elem) for elem in taggedDict[key]]
	return clean_dict

def removetagsText(text):
	clean_txt = re.sub('<.*?>', '', text)
	return clean_txt

def remove_first_whitespaces(string):
	return ''.join(string.lstrip().rstrip())



def updateDic_count(dictionary, word):
	if word in dictionary:
		dictionary[word] += 1
	else:
		dictionary[word] = 1
	return dictionary



def taggedDict(taggedText, database):
	regExps = { # 是否要包括简单的直接用regExp得到的？
				'paragraph' : '<paragraph>.*?</paragraph>',
				'sentence'	: '<sentence>.*?</sentence>',
				'stime'		: '<stime>.*?</stime>',
				'etime'		: '<etime>.*?</etime>',
				'location'	: '<location>.*?</location>',
				'speaker'	: '<speaker>.*?</speaker>',
				'wordsBeforeSpeaker' : '(\w*\s?\w+)(?=\s<speaker>)',
				'wordsAfterSpeaker' : '(?<=</speaker>\s)(\w+\s?\w*)'
	}

	for key in regExps.keys():
		if(key not in database.keys()):
			database[key] = [removetagsText(str) for str in (re.findall(r'' + regExps[key], taggedText, re.DOTALL))]
		else:
			database[key].extend([removetagsText(str) for str in (re.findall(r'' + regExps[key], taggedText, re.DOTALL))])

	return database



def classify_based_tag_train():
	train_sents = treebank.tagged_sents()[:5000]
	#train_sents = brown.tagged_sents(categories='learned', tagset='universal')
	bigram_tagger = BigramTagger(train_sents)
	cbtagger = ClassifierBasedPOSTagger(train=train_sents, backoff = bigram_tagger)
	pickle.dump(cbtagger, open( 'my_tagger.pkl', 'wb' ) )




# location database
class train_location_db:
    def __init__(self):
        self.location_db = set()
        self.uniword_count = dict()
        self.biword_count = dict()
        self.uniword_prob = dict()
        self.bi_trans_prob = dict()
        self.bigram_db = set() 
        self.unigram_db = set()
        self.total_word = 0 # total number of words
	
    def train_word_count_prob(self, list_of_locs, threshold = 0.5): 
        for phrase in list_of_locs:
            tokenized = word_tokenize(phrase)
            length = len(tokenized)
            self.location_db.add(phrase)


            for i in range(0, length - 2):
                self.uniword_count = updateDic_count(self.uniword_count, tokenized[i])
                self.biword_count = updateDic_count(self.biword_count, tokenized[i] + ' ' + tokenized[i + 1])
                self.total_word += 1
            self.uniword_count = updateDic_count(self.uniword_count, tokenized[length - 1])

        # collect the unigrams that appear often 
        for word in self.uniword_count.keys():
        	prog = self.uniword_count[word] / self.total_word
        	self.uniword_prob[word] = self.uniword_count[word] / self.total_word
        	if self.uniword_prob[word] > threshold:
        		self.unigram_db.add(word)

		# compute maximum likelihood estimate 
		# if the trans probability is larger than a given threshold, store the bigram in database
        for phrase in self.biword_count.keys():
            words = re.split(' ', phrase)
            trans_prob = self.biword_count[phrase] / self.uniword_count[words[0]] # trans probability approximate
            self.bi_trans_prob[phrase] = trans_prob
            if trans_prob > threshold:
                 self.bigram_db.add(phrase)
# store the location in database                
def store_location(location_trainer, database):
	location_trainer.train_word_count_prob(database['location'], 0.5) # only store the pharse with a trans_prob above threshold
	return location_trainer

# name database
class train_name_db:
	def __init__(self):
		# name from training set
		self.name_db = set()
		# name given by the family/male/female file
		self.name_corpus = set()
		self.words_near_speaker = set()

		self.get_namecorpus()

	def get_namecorpus(self):
		name_path = '/Users/zhanglingyi/nltk_data/corpora/names.family'
		with open(name_path) as f:
			self.name_corpus.update([x.strip('\n') for x in f.readlines()])
		name_path = '/Users/zhanglingyi/nltk_data/corpora/names.female'
		with open(name_path) as f:
			self.name_corpus.update([x.strip('\n') for x in f.readlines()])
		name_path = '/Users/zhanglingyi/nltk_data/corpora/names.male'
		with open(name_path) as f:
			self.name_corpus.update([x.strip('\n') for x in f.readlines()])



	def get_namedb(self, database):
		for name in database['speaker']:
			self.name_db.update(set(name.split()))

	def get_words_near_speaker(self, wordsBeforeSpeaker, wordsAfterSpeaker, tagger):
		lstofword = list()
		total = 0
		words = dict()
		wordsprob = dict()

		for phrase in wordsBeforeSpeaker+wordsAfterSpeaker:
			lstofword = word_tokenize(phrase)

			for word in lstofword:

					# save the verb/pp near speaker's name
				if tagger.tag([word])[0][1] in ['VB', 'VBD','IN', 'VBG', 'VBN', 'VBP', 'VBZ', 'MD']:
						words = updateDic_count(words, word)
						total += 1
			for word in words.keys():
				wordsprob[word] = words[word] / total
				if wordsprob[word] > 0.05: # threshold the probability of appearence
					self.words_near_speaker.add(word)



def chunk_name_sent(pos_text, text):
  list_of_names = set()
  
  chunk_grammar = r"""

  NAME:   {<VB|VBD|IN|VBG|VBN|VBP|VBZ>*<NNP>+<MD|VB|VBD|IN|VBG|VBN|VBP|VBZ>*}
          {<NNP>+}

  """
  chunker = RegexpParser(chunk_grammar)
  chunked_article = chunker.parse(pos_text)

  return chunked_article

def name_is_in_namedb(list_of_name, nametrain):
	for name in list_of_name:
		if name in nametrain.name_db or name in nametrain.name_corpus:
			return 1
	return 0
# if there is a word that appears near speaker
def word_is_near_speaker(list_of_word, nametrain):
	for word in list_of_word:
		if word in nametrain.words_near_speaker:
			return 1
	return 0

# check if it can be determined as speaker's name and tag
def tag_name(pos_text, text, nametrain, header_speaker, isheader):

	chunked_article = chunk_name_sent(pos_text, text)
	NNPs = list()
	Others = list()
	flag = 0

	#special: the header already tells the speaker
	if header_speaker != '':
		text = re.sub(header_speaker, "<speaker>" + header_speaker + "</speaker>", text)

	for subtree in chunked_article.subtrees(): 
		NNPs = list()
		Others = list()
		if subtree.label()=='NAME':
			for tupl in list(subtree):
				if tupl[1] == 'NNP':
					NNPs.append(tupl[0])
				else:
					Others.append(tupl[0])
			name_probable = ' '.join(NNPs)


			if Others == [] and isheader == 0:
				flag = 0 # if there is no verb and it's not a special case, cannot determine whether it's a speaker or other people
			elif isheader == 0:
				flag = name_is_in_namedb(NNPs, nametrain) and word_is_near_speaker(Others, nametrain) and name_probable != header_speaker
			elif isheader == 1: # it's the speaker found in header
				flag = name_is_in_namedb(NNPs, nametrain)

			if flag == 1:
				text = re.sub(name_probable, "<speaker>" + name_probable + "</speaker>", text)

	return text



def chunk_location_sent(pos_text, text):

	chunk_grammar = r"""

	LOC:   {((<CD>?<NNP>+<CD>?)|(<CD>?<NN>+<CD>?))+}

	"""
	chunker = RegexpParser(chunk_grammar)
	chunked_article = chunker.parse(pos_text)
	return chunked_article

def loc_is_in_location_db(loc, locationTrain):

	# the whole phrase can be found in the previous stored database
	if loc in locationTrain.location_db:
		return 1
	lst_loc = re.split(" ", loc)

	# test if bigram has occured before
	for i in range(1, len(lst_loc)):
		if (lst_loc[i - 1] + " " + lst_loc[i]) in locationTrain.bigram_db:
			return 1
		else:
			continue

	# back off: unigram 
	for item in lst_loc:
		if pos_tag(item) != 'CD':
			if item not in locationTrain.unigram_db:
				return 0 

	return 1


def tag_location(pos_text, text, locationTrain, header_loc):
	list_of_locs = list()

	#special: the header already tells the location
	if header_loc != '':
		if(re.search("[*]", header_loc) != "None"): # or error "nothing to repeat at position 19" will occur
			header_loc = re.sub("[*]", "asterisk", header_loc)
			text = re.sub(header_loc, "<location>" + header_loc + "</location>", text)
			header_loc = re.sub("asterisk", "*", header_loc)
			text = re.sub("asterisk", "*", text)
		else:
			text = re.sub(header_loc, "<location>" + header_loc + "</location>", text)

	chunked_article = chunk_location_sent(pos_text, text)

	for subtree in chunked_article.subtrees(): 
		if subtree.label()=='LOC':
			Loc = ' '.join((tuples[0] for tuples in list(subtree)))
			if(loc_is_in_location_db(Loc, locationTrain) and Loc != header_loc):
				text = re.sub(Loc, "<location>" + Loc + "</location>", text)

	return text

# tag time
def tag_time(text):

  reg_stime = "(?<![-])\s((([0-2]?[1-9]):([0-5][0-9]))\s?([AaPp]\.?[Mm]\.?)?)"
  reg_etime = "(?<=[-])\s??((([0-2]?[1-9]):([0-5][0-9]))\s?([AaPp]\.?[Mm]\.?)?)"
  temp_text = text

  find_stime = [t[0] for t in re.findall(r"" + reg_stime, text, re.IGNORECASE)]
  for time in find_stime:
    time = time.lstrip().rstrip()
    temp_text = re.sub(time, "<stime>" + time + "</stime>", temp_text)

  find_etime = [t[0] for t in re.findall(r"" + reg_etime, text, re.IGNORECASE)]
  for time in find_etime:
    time = time.lstrip().rstrip()
    temp_text = re.sub(time, "<etime>" + time + "</etime>", temp_text)

  return temp_text

# tag sentence
def tag_sent(sent):
	lst_sent = re.split("(\n)", sent)
	lst2_sent = list()
	 # tag the one ends with a . or .?!" & \n and not the '\n' or ' '
	for s in lst_sent:
		flag = 0
		if s != '':
			ls = s.split(" ")
			# it's not a sentence if the last word is not a proper noun
			if (s[-1] in [".", "?", "!"]) and (pos_tag(ls[-1]) != "NNP" or "NN") :
				flag = 1
				s = s[:-1] + "</sentence>" + s[-1]
		lst2_sent += s

		if(flag == 1):
			for i in range(0, len(lst2_sent)):
				if lst2_sent[i] != '' and lst2_sent[i] != "\n" and lst2_sent[i] != " ":
					lst2_sent[i] = "<sentence>" + lst2_sent[i]
					break

	return ''.join(lst2_sent)


# tag paragraph
def tag_para(text):
# find tag <sentence> </sentence>\n to make sure that it's not the newline in the middle of the sentence
  tag_para = text
  find_first_para = re.findall(r"\n\n<sentence>.*?</sentence>", text, re.DOTALL)
  find_back_para = re.findall(r".*?</sentence>.\n\n", text)

  for para in find_first_para:
    tag_para = re.sub(para, para[:2] + "<paragraph>" + para[2:], tag_para)

  for para in find_back_para:
    tag_para = re.sub(para, para[:-2] + "</paragraph>" + para[-2:], tag_para)

  return tag_para

# tag header and extract some useful info
def tag_header(header, locationTrain, nametrain, tagger):

	header_loc = ''
	header_speaker = ''
	header_topic = ''
	header_type = ''
	header_time = ''
	
	# find some obvious info
	if(re.search(r'((?<=speaker:).*)|((?<=who:).*)', header, re.IGNORECASE) != "None"):
		speaker = re.findall(r'((?<=speaker:).*)|((?<=who:).*)', header, re.IGNORECASE)
		if speaker != []:
			if speaker[0][1] != '':
				header_speaker = speaker[0][1] 
			elif speaker[0][0] != '':
				header_speaker = speaker[0][0]
			header_speaker = re.findall(r'<speaker>.*?</speaker>', (tag_name(tagger.tag(word_tokenize(header_speaker)), header_speaker, nametrain, '', 1)))
			if(header_speaker != []):
				header_speaker = header_speaker[0][9:-10]
			else:
				header_speaker = ''
	
	if(re.search(r'(?<=where:|place:).*', header, re.IGNORECASE) != "None"):
		loc = re.findall(r'(?<=where:|place:).*', header, re.IGNORECASE)
		if loc != []:
			header_loc = loc[0]
			header_loc = remove_first_whitespaces(header_loc)

	if(re.search(r'(?<=topic:).*', header, re.IGNORECASE) != "None"):
		header_topic = re.findall(r'(?<=topic:).*', header, re.IGNORECASE)[0]
		

	if(re.search(r'(?<=type:).*', header, re.IGNORECASE) != "None"):
		header_type = re.findall(r'(?<=type:).*', header, re.IGNORECASE)[0]




	if header_loc != '':
		if(re.search("[*]", header_loc) != "None"): # or error "nothing to repeat at position 19" will occur
			header_loc = re.sub("[*]", "asterisk", header_loc)
			header = re.sub(header_loc, "<location>" + header_loc + "</location>", header)
			header_loc = re.sub("asterisk", "*", header_loc)
			header = re.sub("asterisk", "*", header)
		else:
			header = re.sub(header_loc, "<location>" + header_loc + "</location>", header)


	if header_speaker != '':
		header = re.sub(header_speaker, "<speaker>" + header_speaker + "</speaker>", header)


	if header_topic != '':
		pos_topic = tagger.tag(word_tokenize(header_topic))
		new_header_topic = tag_location(pos_topic, header_topic, locationTrain, header_loc)

		new_header_topic = tag_name(pos_topic, new_header_topic, nametrain, header_speaker, 0)

		header = re.sub(header_topic, new_header_topic, header)

	if(re.search(r'(?<=time:\s).*', header, re.IGNORECASE) != "None"):
		header_time = re.findall(r'(?<=time:\s).*', header, re.IGNORECASE)
		if header_time != []:
			header_time = header_time[0]	
	
			if header_time != '':
				new_header_time = tag_time(header_time)

				header = re.sub(header_time, new_header_time, header)

	return (header, header_speaker, header_loc, header_topic, header_type)


# main tagging function
def tag_text(filename, text, locationTrain, database, nametrain, extracted_untagged_db, i):

	header = list()
	temp_text = list()
	pron_name_list =list()
	pron_loc_list = list()

	header_loc = ''
	header_speaker = ''
	header_topic = ''
	header_type = ''

	sentences = list()
	# pos tagged sents
	pos_text = list()
	# POS tagger
	tagger = pickle.load(open('my_tagger.pkl', 'rb' ) )
	extracted_untagged_db['fileName'].append(filename)

	#split the text to header and temp_text(body)
	splitFirstDoubleReturn = text.split("Abstract:", 1)
	if(text == ""):
		return extracted_untagged_db

	header = splitFirstDoubleReturn[0]
	if len(splitFirstDoubleReturn) > 1:
		temp_text = splitFirstDoubleReturn[1]

	(header, header_speaker, header_loc, header_topic, header_type) = tag_header(header, locationTrain, nametrain, tagger)

	# store topics
	extracted_untagged_db['topicAndType'].append(header_topic )

	# tag body
	if temp_text != []:
		pos_text = tagger.tag(word_tokenize(temp_text))

		temp_text = tag_time(temp_text)
		temp_text = tag_location(pos_text, temp_text, locationTrain, header_loc) 
		temp_text = tag_name(pos_text, temp_text, nametrain, header_speaker, 0)

		# tag sentence
		sentences = re.split(r"((?<!\w\.\w.)(?<![A-Z][a-z]\.)((?<=\.|\?|\!)|(?<=\n\n))\s)|((?<=\n\n)\s)", temp_text, re.DOTALL)
		taggedSents = list()

		for sent in sentences:
			if(sent != None):
				sent = tag_sent(sent)
				taggedSents += sent

		joinedTextBody = ''.join(taggedSents)
		# tag paragraph
		if(re.search("[*]", joinedTextBody) != "None"):

			joinedTextBody = re.sub("[*]", "asterisk", joinedTextBody)
			joinedTextBody = tag_para(joinedTextBody)

			joinedTextBody = re.sub("asterisk", "[*]", joinedTextBody)
		else:
			joinedTextBody = tag_para(joinedTextBody)

		taggedText = header + 'Abstract:' + joinedTextBody
		# update extracted untag db
		extracted_untagged_db = taggedDict(taggedText, extracted_untagged_db)		

		with open('/Users/zhanglingyi/nltk_data/corpora/newdir/' + str(i) + '.txt', 'wt') as f:
			f.write(taggedText)
			f.close()

		return extracted_untagged_db


# result is the already been tagged test untag set
def evaluation(trainingCorpus, trainingIDs, resultCorpus, resultIDs):
	regExps = { 
				'paragraph' : '<paragraph>.*?</paragraph>',
				'sentence'	: '<sentence>.*?</sentence>',
				'stime'		: '<stime>.*?</stime>',
				'etime'		: '<etime>.*?</etime>',
				'location'	: '<location>.*?</location>',
				'speaker'	: '<speaker>.*?</speaker>'
	}


	tp_sent = 0
	fp_sent = 0
	total_real_sent = 0

	tp_para = 0
	fp_para = 0
	total_real_para = 0

	tp_stime = 0
	fp_stime = 0
	total_real_stime = 0

	tp_etime = 0
	fp_etime = 0
	total_real_etime = 0

	tp_loc = 0
	fp_loc = 0
	total_real_loc = 0

	tp_speaker = 0
	fp_speaker = 0
	total_real_speaker = 0


	for i in range(0, len(trainingIDs) - 1):
		extract1 = dict()
		extract2 = dict()

		# the training file starts from 301.txt but the result file starts from 0.txt
		e1 = trainingCorpus.raw(str(i + 301) + ".txt")
		e2 = resultCorpus.raw(str(i) + ".txt")

		for key in regExps.keys():
			if(key not in extract1.keys()):
				extract1[key] = [removetagsText(str) for str in (re.findall(r'' + regExps[key], e1, re.DOTALL))]
			else:
				extract1[key].extend([removetagsText(str) for str in (re.findall(r'' + regExps[key], e1, re.DOTALL))])# FLAG? match new lines

		total_real_sent += len(extract1['sentence'])
		total_real_para += len(extract1['paragraph'])
		total_real_stime += len(extract1['stime'])
		total_real_etime += len(extract1['etime'])
		total_real_loc += len(extract1['location'])
		total_real_speaker += len(extract1['speaker'])

		for key in regExps.keys():
			if(key not in extract2.keys()):
				extract2[key] = [removetagsText(str) for str in (re.findall(r'' + regExps[key], e2, re.DOTALL))]
			else:
				extract2[key].extend([removetagsText(str) for str in (re.findall(r'' + regExps[key], e2, re.DOTALL))])# FLAG? match new lines



		for i in extract2['sentence']:
			if i in extract1['sentence']:
				tp_sent += 1
			else:
				fp_sent += 1

		for i in extract2['paragraph']:
			if i in extract1['paragraph']:
				tp_para += 1
			else:
				fp_para += 1

		for i in extract2['stime']:
			if i in extract1['stime']:
				tp_stime += 1
			else:
				fp_stime += 1

		for i in extract2['etime']:
			if i in extract1['etime']:
				tp_etime += 1
			else:
				fp_etime += 1

		for i in extract2['location']:
			if i in extract1['location']:
				tp_loc += 1
			else:
				fp_loc += 1

		for i in extract2['speaker']:
			if i in extract1['speaker']:
				tp_speaker += 1
			else:
				fp_speaker += 1

	precision_sent = tp_sent / (tp_sent + fp_sent)
	recall_sent = tp_sent / total_real_sent

	precision_para = tp_para / (tp_para + fp_para)
	recall_para = tp_para / total_real_para

	precision_stime = tp_stime / (tp_stime + fp_stime)
	recall_stime = tp_stime / total_real_stime

	precision_etime = tp_etime / (tp_etime + fp_etime)
	recall_etime = tp_etime / total_real_etime

	precision_loc = tp_loc/ (tp_loc + fp_loc)
	recall_loc = tp_loc / total_real_loc

	precision_speaker = tp_speaker / (tp_speaker + fp_speaker)
	recall_speaker = tp_speaker / total_real_speaker

	print("the precision:\n sentence: ", precision_sent, "\n paragraph: ",precision_para, "\n stime: ", precision_stime, "\n etime: ", precision_etime, "\n location: ", precision_loc, "\n speaker: ", precision_speaker)
	print("\nthe recall:\n sentence: ", recall_sent, "\n paragraph: ",recall_para, "\n stime: ", recall_stime, "\n etime: ", recall_etime, "\n location: ", recall_loc, "\n speaker: ", recall_speaker)

	#print(extract1, "\n\n", extract2)


	return



# second part of the assignment ################################

def add_dict_item(item, dit):
	if item in dit.keys():
		dit[item] += 1
	else:
		dit[item] = 1
	return dit




def tokenize_stem_filter(str, stemmer, stopw, tokenizer):
	nlst = set()

	tlst = tokenizer.tokenize(str.strip())
	for word in tlst:
		word = word.lower()
		if word not in stopw:
			#nlst.add(stemmer.stem(word))
			nlst.add(word)
	return nlst

# sort frequency of topic words
# topics = database['topic&type']
#
def frequency_topic_keywords(topics):
	freq_lst = dict()
	stemmer = PorterStemmer()
	stopw = set(stopwords.words('english'))

	tokenizer = RegexpTokenizer(r'[A-Za-z]+')

	#print("stop: ", stopw)
	for topic in topics:
		wordset = tokenize_stem_filter(topic, stemmer, stopw, tokenizer)
		for word in wordset:
			freq_lst = add_dict_item(word, freq_lst)

	#print(freq_lst)
	return freq_lst



class ontology():
	
	def __init__(self):
		# the ontology was constructed using the created frequency dictionary of topic words and topics content. can't stem words. word2vec cannot use stem words (result not accurate enough)
		self.ontology = {	

		'science' : {
						'computer' : {
										'robotics' : {'keywords' : {'robotics', 'robots' ,'ai', 'intelligence', 'graphics','hci','image','visualization','automated','vision', 'learning,motion', 'interactive'},
														'files'	: list()
													},

										'hardware' : {'keywords' : {},
														'files'	: list()
													},

										'software' : {'keywords' : {'ps', 'psc', 'cs' },
														'files'	: list()
												},

										'theoretic' : {'keywords' : {'algorithm', 'mathematical', 'strategy',' technique','arithmetical', 'logic'},
														'files'	: list()
													}

									},

						'chemistry' : {'keywords' : set(),
										'files'	: list()
									},
						'physics' : {'keywords' : {},
										'files'	: list()
									},									
						'maths' : 	{'keywords' : {},
										'files'	: list()
									},

					},

		'others' : {'keywords' : {'reminder', 'presentation', 'move', 'teaching', 'financial'},
					'files'	: list()
					},


		'arts' : 	{
						'law' : {'keywords' : {},
								'files'	: list()
								},

						'history' : {'keywords' : {},
								'files'	: list()
								},								

							
						'art' : {'keywords' : {},
								'files'	: list()
								}

					}

		}





		self.freq_dict = dict()
		self.bottom_dict = dict()
		self.update_bottom_dict(self.ontology)

	def save_ontology_to_file(self, filename):
		with open(filename + '.pickle', 'wb') as f:
			pickle.dump(self.ontology, f, protocol = pickle.HIGHEST_PROTOCOL)

	def load_ontology_from_file(self, filename):
		with open(filename + '.pickle', 'rb') as f:
			#return pickle.load(f)
			self.ontology = pickle.load(f)
			#print(self.ontology)

	def save_bottom_dict_to_file():
		with open('bottomdict.pickle', 'wb') as f:
			pickle.dump(self.bottom_dict, f, protocol = pickle.HIGHEST_PROTOCOL)

	def load_bottom_dict_from_file():
		with open('bottomdict.pickle', 'rb') as f:
			#return pickle.load(f)
			self.bottom_dict = pickle.load(f)

	# store the value & key into a bottom_dict dictionary
	def update_bottom_dict(self, level, key = None): # level is where we are now. key is the bottom_dict's key needs to be 
		keys = level.keys()

		# reach the bottom level
		if('keywords' in keys):
			self.bottom_dict[key] = level
			return
		# not the bottom level. store the keys of the next level into the bottom_dict[key]
		for k in keys:
			self.update_bottom_dict(level[k], k)

		return




	def create_frequency_list(self, topics):
		stemmer = PorterStemmer()
		stopw = set(stopwords.words('english'))
		#print("stopwords: ", stopw)

		tokenizer = RegexpTokenizer(r'[A-Za-z]+')

		#print("stop: ", stopw)
		for topic in topics:
			wordset = self.tokenize_stem_filter(topic, stemmer, stopw, tokenizer)
			for word in wordset:
				self.freq_dict = add_dict_item(word, self.freq_dict)

		sort_freq = sorted(self.freq_dict, key=(self.freq_dict).get, reverse = True)
		#print(sort_freq)


	def tokenize_stem_filter(self,str, stemmer, stopw, tokenizer):
		nlst = set()

		tlst = tokenizer.tokenize(str.strip())
		for word in tlst:
			word = word.lower()
			if word not in stopw:
				#nlst.add(stemmer.stem(word)) # can't use stemmed word to create ontology. bad performance word2vec
				nlst.add(word)
		return nlst
		

	def input_keywords_into_ontology(self, topics):
		i = 0

		for topic in topics:
			print(i, "file\n")
			print(topic)
			classname = input("Class name: ").rstrip()
			keywords = input("Keywords (whitespace between input words): ").rstrip().split()
			self.bottom_dict[classname]['keywords'].append(keywords)
			i += 1

		print(self.bottom_dict)
		return 0

	
		
	def classify_topic(self, topicAndTypes, filenameList):
		keys = (self.bottom_dict).keys()
		wordlst = list()
		tokenizer = RegexpTokenizer(r'[A-Za-z]+')
		index = 0;
		model = gensim.models.KeyedVectors.load_word2vec_format( '/Users/zhanglingyi/Documents/NLP/GoogleNews-vectors-negative300.bin', binary=True)
		stopw = set(stopwords.words('english'))

		for topic in topicAndTypes:
			flag = 0
			max_similarity = 0
			classification = ""
			
			# tokenize topic and filter out words that are not in word2vec vocabulary words = filter(lambda x: x in model.vocab, doc.words)
			wordlst = tokenizer.tokenize(topic.strip())
			wordlst = list(filter(lambda x: x in model.vocab, wordlst))
		
			# case there is a keyword in topic recorded in the ontology
			for word in wordlst:

				word = word.lower()
				if word not in stopw:
					for key in keys:
						if word in self.bottom_dict[key]:
							self.bottom_dict[key]['files'].append(filenameList[index])
						
							flag = 1
							break
				if flag == 1:
					break
			if flag == 1:
				index += 1
				continue

			# case there isn't a keyword included in the topic
			for word in wordlst:
				for key in keys:
					similarity = model.similarity(word, key)

					if(similarity > max_similarity):
						max_similarity = similarity
						classification = key

			self.bottom_dict[classification]['files'].append(filenameList[index])
			index += 1





def main():
	
	database = dict()

	# train tag and save tagger to file
	# classify_based_tag_train()
	# if not the first time, load the tagger from file
	tagger = pickle.load(open('my_tagger.pkl', 'rb' ) )

	# training files 
	# store useful info in database
	train_path = '/Users/zhanglingyi/nltk_data/corpora/test_tagged/'
	trainingIDs = load_files(train_path)
	trainingcorpus = read_data(train_path, trainingIDs)


	for f in trainingIDs:
		database = taggedDict(trainingcorpus.raw(f), database)

	nametrain = train_name_db()
	nametrain.get_words_near_speaker(database['wordsBeforeSpeaker'], database['wordsAfterSpeaker'],tagger)
	nametrain.get_namedb(database)

	locationtrain= train_location_db()
	locationtrain = store_location(locationtrain, database)

	# tagging
	untagged_path = '/Users/zhanglingyi/nltk_data/corpora/test_untagged/'
	untaggedIDs = load_files(untagged_path)
	untaggedcorpus = read_data(untagged_path, untaggedIDs)

	extracted_untagged_db = {'paragraph' : [], 'sentence' : [], 'stime' : [], 'etime' : [], 'location' :[], 'speaker' : [], 'topicAndType' : [], 'fileName' : []}

	index = 0
	for f in untaggedIDs:
		extracted_untagged_db =  tag_text(f, untaggedcorpus.raw(f), locationtrain, database, nametrain, extracted_untagged_db, index)
		index += 1
		print(f) 
	
	result_path = '/Users/zhanglingyi/nltk_data/corpora/newdir/'
	resultIDs = load_files(result_path)
	resultcorpus = read_data(result_path, resultIDs)

	# evaluation
	evaluation(trainingcorpus, trainingIDs, resultcorpus, resultIDs)

	# second part of the assignment

	ot = ontology()
	#print(ot.bottom_dict)
	ot.create_frequency_list(extracted_untagged_db['topicAndType'])

	ot.classify_topic(extracted_untagged_db['topicAndType'], extracted_untagged_db['fileName'])

	#print(ot.bottom_dict)
	pprint(ot.ontology)

main()





