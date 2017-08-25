import gc
import scipy.io as io
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from settings import small
from settings import size


#INPUT
train_data_path = "train_data.json"
test_data_path = "test_data.json"

#OUTPUT
train_y_path = "../featurelearning/train_y"
train_matrix_path = "../featurelearning/train_matrix"
test_y_path = "../featurelearning/test_y"
test_matrix_path = "../featurelearning/test_matrix"
feature_list_path = "../featurelearning/feature_list"
if small:
    train_y_path += "_s"
    train_matrix_path += "_s"
    test_y_path += "_s"
    test_matrix_path += "_s"
    feature_list_path += "_s"

train_file = open(train_data_path)

train_review_list = []
train_y = []
iter = size

train_data_string = train_file.read()
train_data_json = json.loads(train_data_string)
for entry in train_data_json:
    if small:
        if iter < 0:
            break
        iter -= 1
    train_review_list.append(entry["text"])
    train_y.append(entry["stars"])

pickle.dump(train_y, open(train_y_path, 'wb'))

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english', ngram_range=(1,2))
train_matrix = vectorizer.fit_transform(train_review_list)
io.mmwrite(train_matrix_path, train_matrix)

train_review_list, train_y, train_data_string, train_matrix = None,None,None,None
gc.collect()

test_file = open(test_data_path)
test_review_list = []
test_y = []

test_data_string = test_file.read()
test_data_json = json.loads(test_data_string)
iter = size/2
for entry in test_data_json:
    if small:
        if iter < 0:
            break
        iter -= 1
    test_review_list.append(entry["text"])
    test_y.append(entry["stars"])

pickle.dump(test_y, open(test_y_path, 'wb'))

test_matrix = vectorizer.transform(test_review_list)
io.mmwrite(test_matrix_path, test_matrix)


features = vectorizer.get_feature_names()
pickle.dump(features, open(feature_list_path, 'wb'))