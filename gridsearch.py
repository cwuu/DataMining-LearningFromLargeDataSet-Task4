## Do Cross-Validation
## 
## 1) set list parameters to test
## 2) make sure that the parameters in the coresets_1.py script are read from the .npy files
## 2) run in shell: python2.7 gridsearch.py
## 
import os
import numpy as np


deltas = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
reward_ps = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
reward_ns = [-0.5, -0.2, 0.0, 0.2]

#TODO be careful which script is called (1 or 2)
call = 'python2.7 runner.py data/webscope-logs.txt data/webscope-articles.txt linUCB_1.py'

for delta in deltas:
	for reward_p in reward_ps:
	    for reward_n in reward_ns:
			np.save('delta.npy', delta)
			np.save('reward_p.npy', reward_p)
			np.save('reward_n.npy', reward_n)

			print 'delta', 'reward_p' 'reward_n'
			print delta, reward_p, reward_n
			os.system(call)