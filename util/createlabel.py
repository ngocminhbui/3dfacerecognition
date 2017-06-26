import numpy as np
f = open('./lists/eval_mini_1.lst', 'r')
lines = f.read().split()
labels = [x.split('/')[0] for x in lines]

print labels
np.save('./lists/eval_mini_1_labels', labels)

np.load('./lists/eval_mini_1_labels.npy')