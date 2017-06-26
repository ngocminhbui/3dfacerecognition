import sklearn.svm as svm
import  numpy as np
from sklearn.naive_bayes import BernoulliNB
import sklearn

import matplotlib.pyplot as plt

train_data_file = '/media/ngocminh/DATA/1412314/Source/tf-rgbd/log/eval_mini_1-1/features20001.txt.npy'
train_label_file = '/media/ngocminh/DATA/1412314/Source/tf-rgbd/log/eval_mini_1-1/features_label20001.txt.npy'

eval_data_file = '/media/ngocminh/DATA/1412314/Source/tf-rgbd/log/eval_mini_1-1/EVALfeatures20001.txt.npy'
eval_label_file = '/media/ngocminh/DATA/1412314/Source/tf-rgbd/log/eval_mini_1-1/EVALfeatures_label20001.txt.npy'

train_data = np.load(train_data_file)
train_label = np.load(train_label_file)

eval_data = np.load(eval_data_file)
eval_label = np.load(eval_label_file)


C = []
for i in range (-10,11):
    C.append(pow(2,i))

for c in C:
    clf = svm.SVC(C=c,  shrinking=True)
    clf.fit(train_data, train_label)

    predicts = clf.predict(eval_data)

    k = np.sum(predicts == eval_label)
    print 'With C=',c,'Precision=',k*1./len(eval_data)


#best alpha: With alpha =  24.0 Precision= 0.884672619048
#best binarize: 0.2
#both: 89%
alpha = np.arange(20,28,0.2)
for a in alpha:
    clf = BernoulliNB(alpha=a, binarize=0.2)
    clf.fit(train_data, train_label)

    predicts = clf.predict(eval_data)

    k = np.sum(predicts == eval_label)
    print 'With alpha, binarize = ', a,0.2,'Precision=',k*1./len(eval_data)



alpha = np.arange(0,30,0.2)
precisions = []
for a in alpha:
    clf = BernoulliNB(alpha=a)
    clf.fit(train_data, train_label)

    predicts = clf.predict(eval_data)

    k = np.sum(predicts == eval_label)
    precisions.append(k*1./len(eval_data))
    print 'With alpha = ', a,'Precision=',k*1./len(eval_data)


plt.plot(alpha[1:], precisions[1:], 'r', label='train err')
plt.xlabel('alpha')
plt.ylabel('precision')
plt.plot()


binarize = np.arange(0,10,0.2)

precisions = []
for b in binarize:
    clf = BernoulliNB(binarize=b)
    clf.fit(train_data, train_label)

    predicts = clf.predict(eval_data)

    k = np.sum(predicts == eval_label)
    precisions.append(k*1./len(eval_data))
    print 'With binarize = ', b,'Precision=',k*1./len(eval_data)


plt.plot(binarize, precisions, 'r', label='train err')
plt.xlabel('binarize')
plt.ylabel('precision')
plt.plot()


# C doesnt affect..
C=np.arange(1,10,1)
for c in C:
    clf = svm.LinearSVC(C=c)
    clf.fit(train_data, train_label)

    predicts = clf.predict(eval_data)

    k = np.sum(predicts == eval_label)
    print 'With C = ', c,'Precision=',k*1./len(eval_data)


alpha = np.arange(0,10,0.2)
for a in alpha:
    clf =  sklearn.naive_bayes.MultinomialNB(alpha=a)
    clf.fit(train_data, train_label)

    predicts = clf.predict(eval_data)

    k = np.sum(predicts == eval_label)
    print 'With alpha = ', a, 'Precision=', k * 1. / len(eval_data)

neighbor = np.arange(1,10,1)
for n in neighbor:
    clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n)
    clf.fit(train_data, train_label)

    predicts = clf.predict(eval_data)

    k = np.sum(predicts == eval_label)
    print 'With neighbor = ', n, 'Precision=', k * 1. / len(eval_data)
