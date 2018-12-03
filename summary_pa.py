import nltk
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import numpy as np
import pprint
import string
import re
import copy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import operator
from networkx.algorithms import community
from networkx.algorithms.community import kernighan_lin_bisection

stop_words = set(stopwords.words('english'))

def mmr_(pr_sorted_, tf_idf_matrix):
    most_sim = pr_sorted_[-1]
    pr_sorted_ = pr_sorted_[:-1]
    
    most_sim_node = most_sim[0]
    
    left_node = []
    for i in pr_sorted_:
        left_node.append(i[0])
    
    for i,j in enumerate(left_node):
        sim = cosine_similarity(tf_idf_matrix.todense()[j], tf_idf_matrix.todense()[most_sim_node])
        lambd = 0.04
        rel_weight = lambd*float(sim)
        pr_sorted_[i][1] = pr_sorted_[i][1] - rel_weight
    
    return pr_sorted_, most_sim_node


def summary(text, no_sent):

	text_orig = copy.deepcopy(text)
	text_lst_init =  re.split('; |\!|\?|\.',text_orig)

	temp_in = []
	for i in text_lst_init:
	    if len(i) > 5:
	        temp_in.append(i)

	text_lst_init = temp_in


	text = re.sub("\d+?\d*(?=\s|,|s|.)\s?", "", text)

	printable = set(string.printable)
	text = filter(lambda x: x in printable, text)
	text = text.replace(',','')
	text = text.replace('  ',' ')

	text_lst = re.split('; |\!|\?|\.',text)

	temp = []
	for i in text_lst:
	    if len(i) > 5:
	        temp.append(i)

	text_lst = temp

	count_vectorizer = CountVectorizer(stop_words = stop_words)
	count_vectorizer.fit_transform(text_lst)

	count_vectorizer.get_feature_names()
	
	freq_term_matrix = count_vectorizer.transform(text_lst)
	tfidf = TfidfTransformer(norm="l2")
	tfidf.fit(freq_term_matrix)

	global tf_idf_matrix
	tf_idf_matrix = tfidf.transform(freq_term_matrix)

	G = nx.Graph()
	nodes = [i for i in range(len(text_lst))]
	G.add_nodes_from(nodes)

	adj_matrix = []
	for i in range((len(text_lst))):
	    adj_matrix.append([])

	for i in nodes:
	    for j in nodes:
	        sim = cosine_similarity(tf_idf_matrix.todense()[i], tf_idf_matrix.todense()[j])
	        adj_matrix[i].append(float(sim))

	for i in nodes:
	    for j in nodes:
	        if i != j:
	            G.add_edge(i,j,weight = adj_matrix[i][j])

	pr=nx.pagerank(G)

	summary_para = []
	summary_para_no = []

	pr_sorted = sorted(pr.items(), key=operator.itemgetter(1))
	for i,w in enumerate(pr_sorted):
		pr_sorted[i] = list(w)

   	pr_new, most_sim_node = mmr_(pr_sorted, tf_idf_matrix)

   	text_lst_init =  re.split('; |\!|\?|\.',text_orig)
	summary_para_no.append(most_sim_node)

	for i,w in enumerate(pr_new):
	    pr_new[i] = tuple(w)

	pr_sorted_new = sorted(pr_new, key=lambda tup: tup[1])


	for i in range(1,no_sent):
		for i,w in enumerate(pr_sorted_new):
		    pr_sorted_new[i] = list(w)
		    
		pr_new, most_sim_node = mmr_(pr_sorted_new, tf_idf_matrix)
		summary_para_no.append(most_sim_node)

		for i,w in enumerate(pr_new):
		    pr_new[i] = tuple(w)

		pr_sorted_new = sorted(pr_new, key=lambda tup: tup[1])

	# summary_para.append(text_lst_init[pr_sorted_new[-1][0]])
	summary_para_no.sort()

  	for i in summary_para_no:
  		summary_para.append(text_lst_init[i])

	return '.'.join(summary_para), text_lst_init, adj_matrix, G

def simple_community(text_lst_init, adj_matrix, G):
	nodes = list(G.nodes)

	G_c = nx.Graph()

	G_c.add_nodes_from(nodes)

	for i in nodes:
	    for j in nodes:
	        if i != j:
	            if adj_matrix[i][j] > 0.17:
	                G_c.add_edge(i,j,weight = adj_matrix[i][j])

	communities_generator = community.girvan_newman(G_c)
	next_level_communities = next(communities_generator)
	comms = sorted(map(sorted, next_level_communities))

	summ_ind = []

	comm_sent = []
	for i in comms:
	    if len(i) > 2:
	        new_G = nx.Graph()
	        for j in i:
	            new_G.add_node(j)
	        
	        for j in i:
	            for k in i:
	                new_G.add_edge(k,j,weight = adj_matrix[j][k])
	                
	        pr = nx.pagerank(new_G)
	        pr_sorted = sorted(pr.items(), key=operator.itemgetter(1))
	        	        
	        summ_ind.append(pr_sorted[-1][0])

	summ_ind.sort()

	for i in summ_ind:
		comm_sent.append(text_lst_init[i])

	return '.'.join(comm_sent)

def bipartite_community(text_lst_init, adj_matrix, G):

	comm_sent = []
	summ_ind = []

	bipart_list = kernighan_lin_bisection(G)
	for i in bipart_list:
	    i = list(i)
	    
	    new_G = nx.Graph()
	    for j in i:
	        new_G.add_node(j)

	    for j in i:
	        for k in i:
	            new_G.add_edge(k,j,weight = adj_matrix[j][k])

	    pr = nx.pagerank(new_G)
	    pr_sorted = sorted(pr.items(), key=operator.itemgetter(1))
	    

	    summ_ind.append(pr_sorted[-1][0])
	    summ_ind.append(pr_sorted[-2][0])

	summ_ind.sort()

	for i in summ_ind:
		comm_sent.append(text_lst_init[i])

	return '.'.join(comm_sent)
