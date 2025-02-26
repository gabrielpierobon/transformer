import numpy as np

X = np.load('data/processed/X_train_M1_M48000_sampled1001.npy')
print('Shape:', X.shape)
print('Dtype:', X.dtype)
print('Memory (MB):', X.nbytes/1024/1024)
print('Zero %:', (X==0).mean()*100)
print('Min:', X.min())
print('Max:', X.max())
print('Unique values:', len(np.unique(X)))
print('First few unique non-zero values:', sorted(list(set(X.flatten())))[1:6]) 