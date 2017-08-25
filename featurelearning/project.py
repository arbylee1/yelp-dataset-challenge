from time import time

from scipy import io
from scipy import sparse
import numpy as np
import pickle
from matplotlib import pyplot


from scipy.stats import chi2
from scipy.stats.mstats_basic import trim
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.svm import LinearSVC
from sklearn.utils.extmath import density
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import Perceptron
from settings import small
from settings import feature_selected as fs

#INPUT
train_y_path = "train_y"
train_matrix_path = "train_matrix"
test_y_path = "test_y"
test_matrix_path = "test_matrix"
feature_list_path = "feature_list"
if small:
    train_y_path += "_s"
    train_matrix_path += "_s"
    test_y_path += "_s"
    test_matrix_path += "_s"
    feature_list_path += "_s"
if fs:
    train_y_path += "_fs"
    train_matrix_path += "_fs"
    test_y_path += "_fs"
    test_matrix_path += "_fs"
    feature_list_path += "_fs"

train_matrix_path += ".mtx"
test_matrix_path += ".mtx"

t0 = time()
stars = [1,2,3,4,5]

print("Loading training data into memory")
train_y = pickle.load(open(train_y_path, 'rb'))
train_matrix = io.mmread(train_matrix_path)
print("\nDone. Took %0.4fs" % (time()-t0))

print("\nLoading testing data into memory")
test_y = pickle.load(open(test_y_path, 'rb'))
test_matrix = io.mmread(test_matrix_path)
print("\nDone. Total time: %0.4fs" % (time()-t0))
print("Training on %d samples with %d features" % train_matrix.shape)
feature_list = pickle.load(open(feature_list_path,'rb'))

# print(train_y)
# k = 50
# print("Extracting %d best features by a chi-squared" % k)
# t0 = time()
# ch2 = SelectKBest(chi2, k)
# train_matrix = ch2.fit_transform(train_matrix, train_y)
# test_matrix = ch2.transform(test_matrix)
#
#
# feature_list = [feature_list[i] for i in ch2.get_support(indices=True)]
# print("Time: %fs" % (time() - t0))
# print()
#
# feature_list = np.asarray(feature_list)

def test(classifier):
    print('\n\n')
    print("Training: ")
    print(classifier)
    t0 = time()
    classifier.fit(train_matrix, train_y)
    train_time = time() - t0
    print("train time: %0.4fs" % train_time)

    t0 = time()
    pred = classifier.predict(test_matrix)
    test_time = time() - t0
    print("test time:  %0.4fs" % test_time)

    score = metrics.accuracy_score(test_y, pred)
    print("accuracy:   %0.4f" % score)

    if hasattr(classifier, 'coef_'):
        print("dimensionality: %d" % classifier.coef_.shape[1])
        print("density: %f" % density(classifier.coef_))

        print("top 50 keywords per rating:")
        for i in stars:
            top50 = np.argsort(classifier.coef_[i-1])[-50:]
            print(trim("%d: %s" % (i, " ".join(feature_list[top50]))))
        print()

        print("Classification report:")
        print(metrics.classification_report(test_y, pred))

        print("Confusion matrix:")
        print(metrics.confusion_matrix(test_y, pred))

    classifier_name = str(classifier).split('(')[0]
    return classifier_name, score, train_time, test_time

results = []
results.append(test(BernoulliNB()))
results.append(test(MultinomialNB()))
results.append(test(LinearSVC()))
results.append(test(Perceptron(n_iter=50)))

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

classifier_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

pyplot.figure(figsize=(12, 8))
pyplot.title("Score")
pyplot.barh(indices, score, .2, label="score", color='r')
pyplot.barh(indices + .3, training_time, .2, label="training time", color='g')
pyplot.barh(indices + .6, test_time, .2, label="test time", color='b')
pyplot.yticks(())
pyplot.legend(loc='best')
pyplot.subplots_adjust(left=.25)
pyplot.subplots_adjust(top=.95)
pyplot.subplots_adjust(bottom=.05)

for i, c in zip(indices, classifier_names):
    pyplot.text(-.3, i, c)

pyplot.show()
