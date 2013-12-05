'''
Analyze the data in the PDFs using LDA and TF-IDF
'''

import os
import subprocess
import optparse
import numpy as np
import json
from gensim import corpora, models # Language models
import itertools
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer, ENGLISH_STOP_WORDS
import re
import unicodedata

# CONFERENCE = 'BMVC'
# YEAR = 2013
CONFERENCE = 'ICCV'
YEAR = 2013
# CONFERENCE = 'CVPR'
# YEAR = 2012

# BASE_FOLDER = os.path.expanduser("~/")+"Dropbox/ConferenceGuides/"
BASE_FOLDER = '/Users/colin/Desktop/ConferenceGuides/'
N_TOPICS=12
N_TOP_WORDS=100

base_folder = BASE_FOLDER
conference = CONFERENCE
year = YEAR
n_topics = 12
n_top_words = 200
min_word_len = 4

def verifiy_folder_hierarchy(folder_name):
	folder_name = folder_name.split("/")
	for i in range(2,len(folder_name)):
		uri = "/".join(folder_name[:i])
		if not os.path.isdir(uri):
			os.mkdir(uri)
			print "Created folder:", uri
	
def cluster_papers(base_folder, conference, year, n_topics=5, n_top_words=200, min_word_len=4):

	conf_folder = base_folder + "{}/{}/".format(year, conference)
	output_folder = base_folder + "/web/{}/{}/".format(year, conference)	
	pdf_folder = conf_folder + "pdfs/"
	txt_folder = conf_folder + "txt/"

	verifiy_folder_hierarchy(output_folder)

	# Load previous data
	all_papers = json.load(open(base_folder+"paper_names/"+conference+str(year)+"_papers.json"))

	# Get stop words
	stopwords = open(base_folder+'stopwords.txt').read().split('\n')
	stopwords.append(list(ENGLISH_STOP_WORDS))

	# Read and filter all of the PDFs
	# Note: We must first convert these PDFs to text
	data = []
	all_txt = []
	all_abstracts = []
	pdf_titles = []
	pdf_authors = []
	pdf_urls = []
	for i,_ in enumerate(all_papers):
		paper = all_papers[i]	
		# Check if we have the PDF or not
		# name = paper['Title'] + " - " + paper['Authors'] + ".pdf"
		first_author = paper['Authors'].split(",")[0].split(" ")[-1]
		first_author = first_author.replace("(", "")
		first_author = first_author.replace(")", "")
		# first_author = re.split("(|)")
		# first_author = re.sub(r'[^\w\s]','',first_author)
		title_abrev = "_".join([x for x in re.split(":| |,", paper['Title']) if len(x)>0][:3])
		title_abrev = title_abrev.replace('''"''', "")
		# title_abrev = "_".join([x for x in re.split(":| |-", paper['Title']) if len(x)>0][:3])
		# title_abrev = "_".join(paper['Title'].split(" ")[:3])
		name = "{}_{}_2013_ICCV_paper.pdf".format(first_author, title_abrev)
		txt_name = "{}_{}_2013_ICCV_paper.txt".format(first_author, title_abrev)
		pdf_file = pdf_folder+name
		if os.path.exists(pdf_file):
			# print "Converting", paper['Title'], "{} of {}".format(i,len(all_papers))
			print "Converting {} of {}".format(i,len(all_papers))
			pdf_titles += [paper['Title']]
			pdf_authors += [paper['Authors']]
			pdf_urls += [paper['PDF']]

			# # Run pdf2text. Do in a subprocess so we can check once it's done.
			# p = subprocess.Popen(["pdftotext", pdf_file, base_folder+"out.txt"])
			# p.wait() # Wait til it is done converting (to prevent a collision)			
			# read_filename = base_folder+"out.txt"

			read_filename = txt_folder + txt_name
			# Read the text
			with open(read_filename, 'r') as f:
				txt_raw = f.read()#.decode("utf-16","ignore")
			# txt_raw = txt_raw.encode("ascii","ignore")
			txt_raw = unicodedata.normalize('NFKD', txt_raw.decode('utf-16', 'ignore'))
			
			
			idx_abstract = txt_raw.find("Abstract")
			idx_intro = txt_raw.find("1. Introduction")
			abstract = txt_raw[idx_abstract+9:idx_intro-1]
			abstract = abstract.replace("\n", " ")
			abstract = unicode(abstract)
			# abstract = unicodedata.normalize('NFKD', abstract)
			# Filter the text
			txt = txt_raw.lower()
			txt = txt.replace("\n", " ")
			txt = txt.split(" ")
			txt = [x for x in txt if x not in stopwords] #stopwords
			txt = [x for x in txt if len(x)>=min_word_len] #short words
			txt = [x for x in txt if not (x[0]=='[' and x[-1]==']')] #references
			txt = [x for x in txt if x.isalpha() and len(x) >= min_word_len]
			txt_unique = np.unique(txt)
			word_hist = dict((x, txt.count(x)) for x in txt_unique)
			txt_unique = [x for i,x in enumerate(txt_unique) if word_hist[x] > 2]
			txt = [x for x in txt if x in txt_unique]
			all_txt += [txt]
			all_abstracts += [abstract]
		else:
			print "Error with:", name		


	# # Save temp data
	# import cPickle as pickle
	# tmp_filename = base_folder+"papers_text/"+conference+"_"+str(year)+"_papers.pkl"
	
	# with open(tmp_filename, 'w') as f:
	# 	pickle.dump({'titles':pdf_titles, 'authors':pdf_authors, 'urls':pdf_urls,
	# 				 'text':all_txt, 'abstract':all_abstracts}, f)
	
	# with open(tmp_filename, 'r') as f:
	# 	data = pickle.load(f)
	# pdf_titles = data['titles']
	# all_txt = data['text']
	# pdf_authors = data['authors']
	# pdf_urls = data['urls']

	n_papers = len(pdf_titles)

	# LDA
	if 0:
		# Build dictionary and corpus
		dictionary = corpora.Dictionary(all_txt)
		corpus_id = [dictionary.doc2bow(x) for x in all_txt]
		# Get top 100 words for each document
		corpus_id = [np.array(c)[np.argsort([x[1] for x in c])[::-1][:n_top_words]] for c in corpus_id]
		corpus_word = [[(dictionary[x[0]], x[1]) for x in c] for c in corpus_id]
		# Update the dictionary to remove infrequent words
		# dictionary = corpora.Dictionary(np.concatenate([[word[0] for word in doc] for doc in corpus_word], 2).tolist())
		
		tmp  = [[word[0] for word in doc] for doc in corpus_word]
		tmp2 = list(itertools.chain(*tmp))
		dictionary = corpora.Dictionary(tmp)
		# dictionary = corpora.Dictionary([word[0] for word in doc for doc in corpus_word])
		corpus_id = [[(dictionary.token2id[w[0]], w[1]) for w in doc] for doc in corpus_word]
		# corpora.MmCorpus.serialize("corpus.mm", corpus_word)

		# Run Latent Dirichlet Analysis to find a set of topics in the documents
		lda = models.ldamodel.LdaModel(corpus_id, num_topics=n_topics, passes=500)

		# Find the dominant topic for each paper
		paper_topics = []
		for i in range(len(corpus_id)):
			paper_topics += [lda[corpus_id[i]][0]]

		# Find the keywords and associated paper for each topic
		n_keywords = 10
		for i in range(n_topics):
			topic_keyword_ids = lda.show_topic(i, n_keywords)
			keywords = [dictionary[int(x[1])] for x in topic_keyword_ids]
			print "Topic {}: {:}".format(i, " ".join(keywords))
			print "----------"
			for ii,p in enumerate(paper_topics):
				if i == p[0]:
					print "--",pdf_titles[ii]
			print


	all_txt_str = [" ".join(x) for x in all_txt]
	# dictionary = np.unique(np.hstack(all_txt))
	# count_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 2), min_df=1)
	# count_vectorizer = CountVectorizer(stop_words=stopwords, vocabulary=dictionary, max_df=.3)
	count_vectorizer = CountVectorizer(stop_words=stopwords, max_df=.3, min_df=5)
	counts = count_vectorizer.fit_transform(all_txt_str)
	tfidf = TfidfTransformer()
	tfidf_vectors = tfidf.fit_transform(counts).todense()
	# tfidf = TfidfVectorizer(stop_words=stopwords, max_df=.1)
	# tfidf.fit(all_txt_str)
	# tfidf_vectors = tfidf.transform(all_txt_str).todense()
	
	# Compute pairwise correlation
	tfidf_pairwise = np.zeros([n_papers, n_papers])
	for i in range(n_papers):
		for j in range(n_papers):
			# tfidf_pairwise[i,j] = ( np.dot(tfidf_vectors[i]*tfidf_vectors[j])/ (np.linalg.norm(tfidf_vectors[i])*np.linalg.norm(tfidf_vectors[j])) )
			# if i!=j:
			tfidf_pairwise[i,j] = ( tfidf_vectors[i]*tfidf_vectors[j].T )#/ (np.linalg.norm(tfidf_vectors[i])*np.linalg.norm(tfidf_vectors[j]))

	if 0:
		import networkx
		graph = networkx.Graph()
		for i in range(len(corpus_id)):
			for j in np.argsort(tfidf_pairwise[i])[-1:-5:-1]:
				graph.add_weighted_edges_from([[pdf_titles[i],pdf_titles[j],tfidf_pairwise[i,j]]])
		networkx.draw_spectral(graph)

	from sklearn.cluster import SpectralClustering, DBSCAN, AffinityPropagation, KMeans, Ward
	# from sklearn.decomposition import NMF
	# topic_model = SpectralClustering(n_clusters=n_topics, affinity='precomputed')
	# topic_model = AffinityPropagation(affinity='precomputed')
	# topic_model = Ward(20)
	# paper_topics2 = topic_model.fit_predict(tfidf_pairwise)
	# topic_model = AffinityPropagation()
	# topic_model = SpectralClustering(n_clusters=n_topics, n_init=100)
	# paper_topics2 = topic_model.fit_predict(tfidf_vectors)
	# print paper_topics2.max()+1
	# print np.histogram(paper_topics2, paper_topics2.max()+1,(0,paper_topics2.max()+1))[0]


	topic_model = KMeans(n_clusters=n_topics, init='k-means++')
	# paper_topics2 = topic_model.fit_predict(tfidf_pairwise)
	paper_topics2 = topic_model.fit_predict(tfidf_vectors)
	np.histogram(paper_topics2, n_topics,(0,n_topics))[0]

	idx_to_word = {x:y for y,x in count_vectorizer.vocabulary_.items()}
	all_keywords = []
	n_words = 10
	for i in range(n_topics):
		keywords = []
		topic_keywords = np.argsort(topic_model.cluster_centers_[i])[-n_words:]
		for j in range(n_words):
			keywords += [idx_to_word[topic_keywords[j]]]
		print "Topic {}:".format(i), " ".join(keywords)
		all_keywords += [" ".join(keywords)]



	# Output to JSON
	top_nodes = 3
	nodes = []
	links = []
	for i in range(n_papers):
		nodes += [{"name":pdf_titles[i], "authors":pdf_authors[i], 'abstract':all_abstracts[i], "url":pdf_urls[i], "group":int(paper_topics2[i])}]
		top_idxs = np.argsort(tfidf_pairwise[i])[-top_nodes-1:]
		for j in range(n_papers):
			if j in top_idxs:
				links += [{"source":i, "target":j, "value":(tfidf_pairwise[i,j])}]
	json.dump({"nodes":nodes, "links":links, 'keywords':all_keywords, "pairwise":tfidf_pairwise.tolist()}, open(output_folder+"graph_data.json", 'w'), indent=4)



if __name__ == "__main__":

	parser = optparse.OptionParser()
	parser.add_option('-f', '--basefolder', dest='base_folder', default=BASE_FOLDER, help='base folder')
	parser.add_option('-c', '--conference', dest='conference', default=CONFERENCE, help='conference acronym')
	parser.add_option('-y', '--year', dest='year', default=YEAR, help='year')
	parser.add_option('-t', '--topics', dest='topics', default=N_TOPICS, help='# of topics')
	parser.add_option('-w', '--words', dest='words', default=N_TOP_WORDS, help='# of top words')
	(opt, args) = parser.parse_args()


	cluster_papers(opt.base_folder, opt.conference, opt.year, opt.topics, opt.words)

