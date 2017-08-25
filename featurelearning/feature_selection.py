from scipy import io
from time import time

import pickle

from scipy.stats import chi2
from sklearn.feature_selection import SelectKBest

from settings import small

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

t0 = time()
stars = [1,2,3,4,5]

print("Loading training data into memory")
train_y = pickle.load(open(train_y_path, 'rb'))
train_matrix = io.mmread(train_matrix_path + ".mtx")
print("\nDone. Took %0.4fs" % (time()-t0))

print("\nLoading testing data into memory")
test_y = pickle.load(open(test_y_path, 'rb'))
test_matrix = io.mmread(test_matrix_path + ".mtx")
print("\nDone. Total time: %0.4fs" % (time()-t0))

print(" %d samples with %d features" % train_matrix.shape)
feature_list = pickle.load(open(feature_list_path,'rb'))

k = train_matrix.shape[1]/100

print("Selecting %d best features using chi-square" % k)
t0 = time()
ch2 = SelectKBest(chi2, k)
train_matrix = ch2.fit(train_matrix, train_y)
train_matrix = ch2.transform(train_matrix)
test_matrix = ch2.transform(test_matrix)

feature_list = [feature_list[i] for i in ch2.get_support(indices=True)]
print("Time: %fs" % (time() - t0))
print()

train_y_path += "_fs"
train_matrix_path += "_fs"
test_y_path += "_fs"
test_matrix_path += "_fs"
feature_list_path += "_fs"

pickle.dump(train_y, open(train_y_path, 'wb'))
io.mmwrite(train_matrix, train_matrix_path)
pickle.dump(test_y, open(test_y_path, 'wb'))
io.mmwrite(test_matrix, test_matrix_path)
pickle.dump(feature_list, open(feature_list_path, 'wb'))

