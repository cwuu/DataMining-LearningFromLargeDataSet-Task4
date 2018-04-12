import numpy as np


k = 6
d = 6

A0 = np.identity(k)
A0_inv = np.identity(k)
b0 = np.zeros(k)
beta = np.zeros(k)

A = {}
A_inv = {}
B = {}
b = {}

# stuff for ucb calculation that is faster when cached
# notation refers to HybridUCB algorithm on the slides
zA0z = {}
zA0BA = {}
ABA0BA = {}
theta = {}


article_features = {}

# Learning rate
r0 = -0.1
r1 = 3.0
alpha = 0.1
#user and article at round t
user_t = None
article_t = None


# articles is a python dictionary
def set_articles(articles):
    for key, value in articles.iteritems():
        article_features[key] = np.array(value)
    pass



def update(reward):
    global A0, b0, A, B, b, A_inv, A0_inv, theta, beta, zA0z, zA0BA, ABA0BA

    #
    if reward != -1:
        scaled_reward = r1 if reward == 1 else r0

        A0 += np.dot(np.dot(np.transpose(B[article_t]),A_inv[article_t]), B[article_t])
        b0 += np.dot(np.dot(np.transpose(B[article_t]),A_inv[article_t]), b[article_t])

        A[article_t] += np.outer(user_t,user_t)
        B[article_t] += np.outer(user_t,article_features[article_t])
        b[article_t] += scaled_reward*user_t
        A_inv[article_t] = np.linalg.inv(A[article_t])

        A0 += np.outer(article_features[article_t],article_features[article_t]) - np.dot(np.dot(np.transpose(B[article_t]),A_inv[article_t]), B[article_t])
        b0 += scaled_reward*article_features[article_t] - np.dot(np.dot(np.transpose(B[article_t]),A_inv[article_t]), b[article_t])

        A0_inv = np.linalg.inv(A0)
        beta = np.dot(A0_inv,b0)

        theta[article_t] = np.dot(A_inv[article_t],b[article_t] - np.dot(B[article_t],beta))

        temp = np.dot(np.dot(A0_inv, B[article_t].T), A_inv[article_t])

        zA0z[article_t] = np.dot(np.dot(article_features[article_t], A0_inv),article_features[article_t])

        zA0BA[article_t] = np.dot(article_features[article_t], temp)

        ABA0BA[article_t] = np.dot(np.dot(A_inv[article_t],B[article_t]),temp)
    pass


def recommend(time, user_features, choices):
    global user_t, article_t, A, A_inv, B, b, zA0z, zA0BA, ABA0BA, theta

    user = np.asarray(user_features)
    article_t = 0
    max_ucb = 0


    for art_id in choices:
        art = article_features[art_id]
        if not art_id in A:
            A[art_id] = np.identity(d)
            A_inv[art_id] = np.identity(d)
            B[art_id] = np.zeros((d,k))
            b[art_id] = np.zeros(d)

            zA0z[art_id] = np.dot(np.dot(art, A0_inv), art)
            zA0BA[art_id] = np.zeros(d)
            ABA0BA[art_id] = np.zeros((d,d))
            theta[art_id] = np.zeros(d)

        # calculation of s
        term1 = zA0z[art_id]
        term2 = -2.0 * np.dot(zA0BA[art_id], user)
        term3 = np.dot(np.dot(user,A_inv[art_id]),user)
        term4 = np.dot(np.dot(user,ABA0BA[art_id]),user)

        s = term1 + term2 + term3 + term4

        ucb = np.dot(art, beta)
        ucb += np.dot(user,theta[art_id])
        ucb += alpha * np.sqrt(s)

        if ucb > max_ucb:
            article_t = art_id
            max_ucb = ucb


    user_t = user
    return article_t
