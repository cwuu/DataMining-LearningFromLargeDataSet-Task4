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
## 			Apply Hybrid UCB as discussed in lecture (L12 P.33)
## 			Naming of variables is taken as on the lecture slides
## 			
###################################################################################################

import numpy as np

### SETTINGS ######################################################################################

alpha = 0.1
reward_pos = 3.0  # rescaled positive reward
reward_neg = -0.1  # rescaled negative reward

### GLOBAL VARIABLES ##############################################################################

num_articles = 271
num_art_features = 6
num_user_features = 6

M = dict()  # matrix M for every article
Minv = dict()  # inverses of M
B = dict()  
b = dict() #weighted
zA0z = dict()
zA0BA = dict()
ABA0BA = dict()
theta = dict()

article_features = dict()
A0 = np.identity(num_art_features)
A0inv = np.identity(num_art_features)
b0 = np.zeros(num_art_features)
beta = np.zeros(num_art_features)

current_user_features = None  # save the current user in order to make update in case of made impression
current_article = None  # save which article was recommended last

### FUNCTIONS #####################################################################################

def set_articles(articles):
	""" 
	articles: is a dictionary, key's are article id's, values the corresponding feature vectors
	"""
	global article_features

	# Initialize global variables
	for article_id, features in articles.iteritems():
		article_features[article_id] = np.array(features)

		M[article_id] = np.identity(num_user_features)
		Minv[article_id] = np.identity(num_user_features)
		B[article_id] = np.zeros((num_art_features,num_user_features))
		b[article_id] = np.zeros(num_user_features) #w

		zA0z[article_id] = np.dot(np.dot(np.array(features), A0inv), np.array(features))
		zA0BA[article_id] = np.zeros(num_user_features)
		ABA0BA[article_id] = np.zeros((num_user_features,num_art_features))
		theta[article_id] = np.zeros(num_user_features)


def update(reward):
	""" 
	reward: -1 if recommendation not matched with log, 0 if wrong recommended, 1 successful
	"""
	global M, Minv, b, B, current_user_features, current_article, A0, A0inv, b0, beta, theta, zA0z, zA0BA, ABA0BA, reward_pos, reward_neg

	if reward != -1:  # if we had a match with the log
		if reward == 1:
			reward = reward_pos
		else:
			reward = reward_neg
		
		A0 += np.dot(np.dot(np.transpose(B[current_article]),Minv[current_article]), B[current_article])
		b0 += np.dot(np.dot(np.transpose(B[current_article]),Minv[current_article]), b[current_article])
		M[current_article] += np.outer(current_user_features, current_user_features)
		Minv[current_article] = np.linalg.inv(M[current_article])
		B[current_article] += np.outer(current_user_features,article_features[current_article])
		b[current_article] += reward * current_user_features #w
		A0 += np.outer(article_features[current_article],article_features[current_article]) - np.dot(np.dot(np.transpose(B[current_article]),Minv[current_article]), B[current_article])
		b0 += reward*article_features[current_article] - np.dot(np.dot(np.transpose(B[current_article]),Minv[current_article]), b[current_article])

		A0inv = np.linalg.inv(A0)
		beta = np.dot(A0inv,b0)

		A0BA = np.dot(np.dot(A0inv, np.transpose(B[current_article])), Minv[current_article])
		zA0z[current_article] = np.dot(np.dot(article_features[current_article], A0inv),article_features[current_article])
		zA0BA[current_article] = np.dot(article_features[current_article], A0BA)
		ABA0BA[current_article] = np.dot(np.dot(Minv[current_article],B[current_article]),A0BA)

		theta[current_article] = np.dot(Minv[current_article],b[current_article] - np.dot(B[current_article],beta))
 
def recommend(time, user_features, choices):
	""" 
	time: integer timestep
	user_features: list of floats
	choices: list of ints

	recommend argmax UCB(x)
	"""
	global M, Minv, b, B, alpha, current_user_features, current_article, zA0z, zA0BA, ABA0BA, theta

	user_features = np.array(user_features)
	max_pt = -np.inf

	#Hybrid LinUCB
	for article_id in choices:
		article = article_features[article_id]

		st = zA0z[article_id] 
		st += -2.0 * np.dot(zA0BA[article_id], user_features)
		st += np.dot(np.dot(user_features,Minv[article_id]),user_features) 
		st += np.dot(np.dot(user_features,ABA0BA[article_id]),user_features)

		pt = np.dot(article, beta) + np.dot(user_features,theta[article_id]) + alpha * np.sqrt(st)

		if pt > max_pt:
			current_article = article_id
			max_pt = pt

	current_user_features = user_features
	return current_article