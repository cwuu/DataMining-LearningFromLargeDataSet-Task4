###################################################################################################
##
## Data Mining: Learning from Large Datasets
## Lecture Project - Task 4 (Large Scale Bandit Optimization)
## 
## Team: 	Datasloths
## Authors: Raphael S. (rasuter@student.ethz.ch)
## 			Hwang S. (hwangse@student.ethz.ch)
## 			Wuu Cheng-Hsin (wch@student.ethz.ch)
## 	
## Approach:			
## 			Apply Linear UCB as discussed in lecture
## 			
###################################################################################################

import numpy as np

### SETTINGS ######################################################################################

delta = 0.2  # 1-delta ~ coverage
reward_p = 3.0  # rescaling reward
reward_n = -0.1

delta = np.load('delta.npy')
reward_p = np.load('reward_p.npy')
reward_n = np.load('reward_n.npy')

### GLOBAL VARIABLES ##############################################################################

alpha = 1 + np.sqrt(np.log(2.0/delta)/2.0)  # follows from delta
M = dict()  # matrix M for every article
Minv = dict()  # inverses of M
b = dict()  # vector b for every article
w = dict()  # weight vectors
article_features = dict()
num_articles = 271
num_art_features = 6
num_user_features = 6

current_user_features = None  # save the current user in order to make update in case of made impression
current_article = None  # save which article was recommended last

### FUNCTIONS #####################################################################################

def set_articles(articles):
	""" 
	articles: is a dictionary, key's are article id's, values the corresponding feature vectors

	do initializations of global variables 
	"""
	global M, Minv, b, w, article_features, num_articles, num_art_features, num_user_features

	article_features = articles

	for article_id in articles.keys():
		M[article_id] = np.identity(num_user_features)
		Minv[article_id] = np.identity(num_user_features)
		b[article_id] = np.zeros([num_user_features])
		w[article_id] = np.zeros([num_user_features])

def update(reward):
    """ 
	reward: is either -1 or 0

	update matrix M and vector b for the propsed article (if impression was made)
    M <- M + z*z^T, b <- b + y*z
    """
    global M, Minv, b, w, current_user_features, current_article

    if reward != -1:  # if we had a match with the log
    	reward = reward_p if reward == 1 else reward_n
    	M[current_article] += np.outer(current_user_features, current_user_features)
    	Minv[current_article] = np.linalg.inv(M[current_article])
    	b[current_article] += reward * np.array(current_user_features)
    	w[current_article] = np.dot(Minv[current_article], b[current_article])
    

def recommend(time, user_features, choices):
	""" 
	time: integer timestep
	user_features: list of floats
	choices: list of ints

	recommend argmax UCB(x)
	"""
	global M, Minv, b, alpha, current_user_features, current_article

	user_features = np.array(user_features)
	ucb = dict()
	for article_id in choices:
		temp = np.dot(Minv[article_id], user_features)
		ucb[article_id] = np.inner(user_features, w[article_id]) + alpha*np.sqrt(np.inner(user_features, temp))

	current_article = max(ucb, key=ucb.get)  # get article_id with lowest UCB
	current_user_features = user_features

	return current_article