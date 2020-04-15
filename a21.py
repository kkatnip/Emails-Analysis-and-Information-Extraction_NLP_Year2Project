from pprint import pprint
import pickle
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet as wn
import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.tag import pos_tag
from nltk.tag.sequential import ClassifierBasedPOSTagger
import sys, http.client, urllib.request, urllib.parse, urllib.error, json
from nltk.corpus import stopwords
import json
import nltk, re
from nltk.corpus import treebank, brown
from os import listdir #listdir Return a list containing the names of the entries in the directory given by path
from os.path import isfile, join
import os
from nltk.corpus.reader.plaintext import PlaintextCorpusReader


def add_dict_item(item, dit):
	if item in dit.keys():
		dit[item] += 1
	else:
		dit[item] = 1
	return dit


def save_into_pickle_file(item, name):
    with open('folder/'+ name + '.pkl', 'wb') as f:
        pickle.dump(item, f, pickle.HIGHEST_PROTOCOL)

def load_pickle_file(name ):
    with open('folder/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


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
							self.bottom_dict[key]['files'].append(topicAndTypes[index])
						
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

			self.bottom_dict[classification]['files'].append(topicAndTypes[index])
			index += 1






topics = ['    AN ASYNCHRONOUS TEAM SOLUTION TO SCHEDULING STEEL PLANTS      cmu.cs.robotics', '    Antal Bejczy Lecture Nov. 11      cmu.cs.robotics', '    The D* Algorithm for Real-Time Path Replanning      cmu.cs.robotics', '    Using Two-Dimensional Image Matching to Interpret the Three-Dimensional World      cmu.cs.robotics', '    Dante II and Beyond: Exploration Robots      cmu.cs.robotics', '    pSather 1.0 - A Simpler Second System      Special CS Seminar', '    An Autonomous Walking Robot for Planetary Exploration      cmu.cs.robotics', '    SCS AI Seminar Next Tuesday      cmu.cs.robotics', '    An Autonomous Walking Robot for Planetary Exploration      cmu.cs.robotics', '    3D-TV (and Workstations)      cmu.cs.robotics', '    Exploring the autonomous learning problem with mathematical programming      cmu.cs.robotics', '    Robot Skill Learning Through Intelligent Experimentation      cmu.cs.robotics', '    Robots for the Assistance to the Disabled and the Elderly:      cmu.cs.robotics', '    Learning to Understand Information on the Internet.      AI Seminar', '    Autonomous Interactive Artwork        cmu.cs.robotics', '    AARON: In Living Color      Joint AI/HCI Seminar', "    TODAY *ROOM CHANGE*  Anita Flynn's Talk (Faculty Candidate): Smith 125.      cmu.cs.robotics", '    Upcoming Robotics Seminars: Apr 14 none; Apr 21 Shree Nayar.      cmu.cs.robotics', '    ON CHOOSING THE BEST FEEDBACK CONTROL STRATEGY FOR      cmu.cs.robotics', '    Control and Manufacturing Research and Education Programs      cmu.cs.robotics', '    ON CHOOSING THE BEST FEEDBACK CONTROL STRATEGY FOR      cmu.cs.robotics', '    Programming Complex Mechanical Systems with Applications       cmu.cs.robotics', '    AI Seminar Next Tuesday      cmu.cs.robotics', '    Medical Robotics: A Step Toward Computer Integrated Medicine      cmu.cs.robotics', '    Dexterity and Intelligence in Robotic Systems      cmu.cs.robotics', '    Control Strategies for Intermittent Dynamic Tasks      cmu.cs.robotics', '    AI Seminar This Friday      cmu.cs.robotics', '    apE visualization software talk this Tue.      cmu.cs.robotics', '    Product Design for Environmental Compatibility      cmu.cs.robotics', '    Product Design for Environmental Compatibility      cmu.cs.robotics', '    CIMDS SEMINAR      cmu.cs.robotics', '    DONATH TALK MOVED      cmu.cs.robotics', '    Robotics Seminar, Prof. John Canny, Friday Oct 11, 3:30, Adamson Wing, Baker Hall      cmu.cs.robotics', '    Special Robotics Seminar      cmu.cs.robotics', '    RI Symposium      cmu.cs.robotics', '    REMINDER - Li Talk Today      cmu.cs.robotics', '    The Programmable Automated Welding System      cmu.cs.robotics', '    Robotics Seminar, Prof. Jean-Claude Latombe, Friday Oct 25, 3:45, Doherty Hall 1112      cmu.cs.robotics', '    Goldfarb Seminar      cmu.cs.robotics', '    VISION Seminar TODAY      cmu.cs.robotics', '    Robotics Seminar, Jeff Kerr, Friday Nov 22, 3:30, Adamson Wing, Baker Hall      cmu.cs.robotics', '    Robotics Seminar (Driscoll)      cmu.cs.robotics', '    There is more to life than the FFT      Robotics Seminar', '    Theory Seminar, Feb. 14, 1992      cmu.cs.robotics', '    Learning Control for Improved Performance in       cmu.cs.robotics', '    Neural net models of lightness and color constancy.      cmu.cs.robotics', '    Computer vision and geometric measurement      cmu.cs.robotics', '    Reminder: Manocha Talk today      cmu.cs.robotics', '    Robotics/Planning Talk: Dan Koditschek      cmu.cs.robotics', '    RI SEMINAR CORRECTION      cmu.cs.robotics', '    Antarctic Exploration Robotics      cmu.cs.robotics', '    Antarctic Exploration Robotics      cmu.cs.robotics', '    Techniques for Task-Directed Sensor Data       cmu.cs.robotics', '    Graphics Seminar, Thu Nov 12, Kinematics      cmu.cs.robotics', '    Radargrammetric Processing of Magellan Images of the Planet Venus      cmu.cs.robotics', '    reminder: Elisha Sacks talk TODAY      cmu.cs.robotics', '    Space Robotics Activities at the Johnson Space Center      cmu.cs.robotics', '    FILTERS FOR IMAGE RESAMPLING      cmu.cs.robotics', '    FINE MOTION PLANNING FOR DEXTEROUS MANIPULATION      cmu.cs.robotics', '    SOLVING SCHEDULING PROBLEMS WITH THE HELP OF MULTI-AGENT SYSTEMS      cmu.cs.robotics', '    BEZIER-MINKOWSKI METAMORPHOSIS & INTERACTIVE GEOMETRIC MODELING      cmu.cs.robotics', '    Seminar      cmu.cs.robotics', '    RI SEMINARS      cmu.cs.robotics', '    Talk on object segmentation techniques and constraint based feature indexing      cmu.cs.robotics', '    MULTIPLE-MODEL MACHINE VISION      cmu.cs.robotics', '    TA seminars, UTC Grad support F94      cmu.cs.robotics.students', '    S95 Graduate Seminar Schedule      cmu.cs.robotics.students', '    Financial Aid Seminar this Thursday      cmu.cs.robotics.students', '    Univ. Teaching Ctr. summer seminars      cmu.cs.robotics.students', '    Univ. Teaching Ctr. summer seminars      cmu.cs.robotics.students', '    Univ. Teaching Ctr. summer seminars      cmu.cs.robotics.students', '    Talk: "DTM from SPOT and Aerial Images"      cmu.cs.scs', '    Green Engineering: Rethinking Engineering Design And        cmu.cs.scs', '    Iterative Design of Efficient Parallel Programs      cmu.cs.scs', '    Elemental measurements in vision.      cmu.cs.scs', '    Elemental measurements in vision.      cmu.cs.scs', '    PSC/CS Seminar: additonal speakers      cmu.cs.scs', '    Programming Systems Seminar      cmu.cs.scs', '    Antarctic Exploration Robotics      cmu.cs.scs', '    1ST ANNUAL SCS HOMECOMING LECTURE      cmu.cs.scs', '    Research in Teleoperated and Cooperative Control       cmu.cs.scs', '    THE FUTURE OF COMPUTING-Lecture      cmu.cs.scs', '    GRAPHICS SEMINAR: Nov. 11, "Computer Graphics Snacks"      cmu.cs.scs', '    Space Robotics Activities at the Johnson Space Center      cmu.cs.scs', '    IC Talks rescheduled      cmu.cs.scs', '    reminder: Elisha Sacks talk TODAY      cmu.cs.scs', '    CMT Research Seminar      cmu.cs.scs', '    Autonomous Walking for the Ambler Planetary      cmu.cs.scs', '    Distinguished Lecture December 2      cmu.cs.scs', '    CS/PSC Seminar today @ 4:00 in Mellon Inst. on Cray MPP      cmu.cs.scs', '    Space Robotics Seminar      cmu.cs.scs', '    POP SEMINAR      cmu.cs.scs', '    CS/PSC Seminar 1/22: Miura on Fujitsu VPP 500      cmu.cs.scs', '    A Revision-Theoretic Analysis of the Arithmetical Hierarchy      LOGIC COLLOQUIUM', '    WWC: Picturetel seminar: Patterson, Feb 10, 7-8:30 pm, 4623 WeH      cmu.cs.scs', '    Change!!: Wed WWC PictureTel Seminar      cmu.cs.scs', '    Change!!: Wed WWC PictureTel Seminar      cmu.cs.scs', '    Program Dependence Graph and Static Single Assignment forms      cmu.cs.scs', '    Re: Lecture by Rajiv Gupta, Univ of Pittsburgh, Feb 12, 9 AM      cmu.cs.scs', '    WWC video seminar: Feb 17 7:15-8:15 pm 4623 WeH Christos H. Papadimitriou      cmu.cs.scs', '    Tax Seminar for Non-Residents      cmu.cs.scs', '    PSC/CS Seminar 3/5 4:00 R. Ewing on Reservoir Simulation      cmu.cs.scs', '    March 17 Distinguished Lecture      cmu.cs.scs', '    PSC/CS Seminar 3/12 4:00 WeH 5202 S. Frank of KSR Inc.      cmu.cs.scs', '    "Testing Preorders for Reactive Real-Time Processes"      PS Seminar', '    MSE End-of-Semester Presentation      cmu.cs.scs', '    Healthy Office Seminar      cmu.cs.scs', '    PSC Seminar today @ 4:00: Tera Comp.      cmu.cs.scs', '    Short talk on Data Breakpoints      cmu.cs.scs', '    Special AI Seminar      cmu.cs.scs', '    reminder: grad student teaching seminars      cmu.cs.scs', '    reminder: grad student teaching seminars      cmu.cs.scs', '    SCS Distinguished Lecture Oct. 7      cmu.cs.scs', '    PSC/CS Seminar 10/8 4:00 pm WeH 5409: Marc Snir of IBM      cmu.cs.scs', '    CS/PSC Seminar 10/22 @ 4:00 p.m WeH 5409 A. Nowatzyk of Sun      cmu.cs.scs', '    Best-First Minimax Search      AI Seminar', "    The EPAM Model of Experts' Discrimination and Learning:       AI Seminar", '    Automated Casual Inference: Recent Work and Open Problems      AI Seminar', '    "Teaching Automata Theory by Computer"      cmu.cs.scs', '    Actualized Intelligence: Planning and Acting in Dynamic Environments      AI Seminar', '    MSE End-of-Semester Presentation      cmu.cs.scs', '    CS/PSC Seminar 12/3 4:00pm Kai Li WeH 5409      cmu.cs.scs', '    Speech Seminar      cmu.cs.scs', '    PS Seminar      cmu.cs.scs', '    PS Seminar      cmu.cs.scs', '    HCI Seminar 1/26      cmu.cs.scs', '    Systems Design of An Off Road Autonomous Navigator      cmu.cs.scs', '    PSC/CS Seminar 11/1 4:00 WeH 5403: J. Smith of Cray Research      cmu.cs.scs', '    Planning: Recent Results      AI Seminar', '    Basic Paramodulation and Basic Strict Superposition      cmu.cs.scs', '    High Performance Fortran      cmu.cs.scs', '    HCI Seminar Feb 16      cmu.cs.scs', '    CS/PSC SEMINARS/SPRING 94      cmu.cs.scs', '    CS/PSC Seminar today @ 4:00 G. Miller WeH 5409      cmu.cs.scs', '    Improved Statistical Language Models from Syntactic Parsing      AI Seminar', '    HCI Seminar March 23      cmu.cs.scs', '    March 23 HCI Seminar      cmu.cs.scs', '    Mach Internals Seminar: Physical Memory Management      cmu.cs.scs', "    NASA's Artificial Intelligence Program      AI Seminar", '    Parallel Computing at the Swiss Scientific      cmu.cs.scs', '    A FREE OBJECT TECHNOLOGY TELECAST/SEMINAR      cmu.cs.scs', '    Computer Systems Seminar 9/16      cmu.cs.scs', '    Calendar joint CS-PSC seminar series      cmu.cs.scs', '    Calendar joint CS-PSC seminar series      cmu.cs.scs', '    Calendar joint CS-PSC seminar series      cmu.cs.scs', '    Calendar joint CS-PSC seminar series      cmu.cs.scs', '    9/23 CS/PSC Seminar      cmu.cs.scs', '    REMINDER - KATZ TALK      cmu.cs.scs', '    CS/PSC/PS seminar Steve McGeady on 10/17      cmu.cs.scs', '    Special AI Seminar      cmu.cs.scs', '    PSC/CS Seminar 9/30: Dennis Gannon      cmu.cs.scs', '    Redesigning the Classroom: Teaching and Technology      cmu.cs.scs', '    "Mathematical Theory of Quasicrystals"      cmu.cs.scs', '    POP TALK      cmu.cs.scs', '    CS Seminar 12/2: Rick Stevens      CS Seminar', '    APPLE COMPUTER TALK      cmu.cs.scs', '    THEORY SEMINAR      cmu.cs.scs', '    POP SEMINAR      cmu.cs.scs', '    PSC/CS Seminar 2/24: Risto Miikkulainen      cmu.cs.scs', '    Logical Interpretations and Computational Simulations      cmu.cs.scs', '    HCI seminar, Raj Reddy, 3:30 Friday 5-5, Wean 5409      cmu.cs.scs', '    Special ECE Seminar May 19      cmu.cs.scs', '    Re: Seminar notice (STC)      cmu.cs.scs', '    Kripke Models for Linear Logic      cmu.cs.scs', '    TALK - Jeff Trinkle, Texas A&M Univ      cmu.cs.scs', '    Ron Rivest-Distingished Lecture Series-wed      cmu.cs.scs', '    Algorithms for Square Roots of Graphs      cmu.cs.scs', '    Algorithms for Square Roots of Graphs      cmu.cs.scs', '    PS SEMINAR      cmu.cs.scs', '    LOGIC COLLOQUIUM      cmu.cs.scs', '    PSC/CS Seminar: 2/28 4:00 @ SEI       cmu.cs.scs', '    Distinguished Lecture--Today      cmu.cs.scs', '    Networks class/DQDB talk      cmu.cs.scs', '    CS Seminar 4/10: Leiserson of MIT/TMC @ 4:00 p.m., WeH 5403      cmu.cs.scs', '    DISTINGUISHED LECTURE--TODAY      cmu.cs.scs', '    David Wile to give talk, Thursday April 16      cmu.cs.scs', '    Special PS Seminar 4/26      cmu.cs.scs', '    The Temporal Logic of Actions      cmu.cs.scs', '    The Temporal Logic of Actions      cmu.cs.scs', '    Undergrad Research Presentations      cmu.cs.scs', '    Special Seminar, Monday May 11      cmu.cs.scs', '    SONY TALK: Dr. Seiichi Watanabe      cmu.cs.scs', '    Interesting HCI seminar...      cmu.edrc.ndim', '    EPP TALK: Household Demand for Garbage and Recycling Collection      cmu.misc.environmental-health-and-safety']
filenames = ['301.txt', '302.txt', '303.txt', '304.txt', '305.txt', '306.txt', '307.txt', '308.txt', '309.txt', '310.txt', '311.txt', '312.txt', '313.txt', '314.txt', '315.txt', '316.txt', '317.txt', '318.txt', '319.txt', '320.txt', '321.txt', '322.txt', '323.txt', '324.txt', '325.txt', '326.txt', '327.txt', '328.txt', '329.txt', '330.txt', '331.txt', '332.txt', '333.txt', '334.txt', '335.txt', '336.txt', '337.txt', '338.txt', '339.txt', '340.txt', '341.txt', '342.txt', '343.txt', '344.txt', '345.txt', '346.txt', '347.txt', '348.txt', '349.txt', '350.txt', '351.txt', '352.txt', '353.txt', '354.txt', '355.txt', '356.txt', '357.txt', '358.txt', '359.txt', '360.txt', '361.txt', '362.txt', '363.txt', '364.txt', '365.txt', '366.txt', '367.txt', '368.txt', '369.txt', '370.txt', '371.txt', '372.txt', '373.txt', '374.txt', '375.txt', '376.txt', '377.txt', '378.txt', '379.txt', '380.txt', '381.txt', '382.txt', '383.txt', '384.txt', '385.txt', '386.txt', '387.txt', '388.txt', '389.txt', '390.txt', '391.txt', '392.txt', '393.txt', '394.txt', '395.txt', '396.txt', '397.txt', '398.txt', '399.txt', '400.txt', '401.txt', '402.txt', '403.txt', '404.txt', '405.txt', '406.txt', '407.txt', '408.txt', '409.txt', '410.txt', '411.txt', '412.txt', '413.txt', '414.txt', '415.txt', '416.txt', '417.txt', '418.txt', '419.txt', '420.txt', '421.txt', '422.txt', '423.txt', '424.txt', '425.txt', '426.txt', '427.txt', '428.txt', '429.txt', '430.txt', '431.txt', '432.txt', '433.txt', '434.txt', '435.txt', '436.txt', '437.txt', '438.txt', '439.txt', '440.txt', '441.txt', '442.txt', '443.txt', '444.txt', '445.txt', '446.txt', '447.txt', '448.txt', '449.txt', '450.txt', '451.txt', '452.txt', '453.txt', '454.txt', '455.txt', '456.txt', '457.txt', '458.txt', '459.txt', '460.txt', '461.txt', '462.txt', '463.txt', '464.txt', '465.txt', '466.txt', '467.txt', '468.txt', '469.txt', '470.txt', '471.txt', '472.txt', '473.txt', '474.txt', '475.txt', '476.txt', '477.txt', '478.txt', '479.txt', '480.txt', '481.txt', '482.txt', '483.txt', '484.txt']




train_path = '/Users/zhanglingyi/nltk_data/corpora/test_tagged/'
trainingIDs = load_files(train_path)
trainingcorpus = read_data(train_path, trainingIDs)

untagged_path = '/Users/zhanglingyi/nltk_data/corpora/test_untagged/'
untaggedIDs = load_files(untagged_path)
untaggedcorpus = read_data(untagged_path, untaggedIDs)




ot = ontology()
#print(ot.bottom_dict)
ot.create_frequency_list(topics)

ot.classify_topic(topics, filenames)

print(ot.bottom_dict)
pprint(ot.ontology)




























