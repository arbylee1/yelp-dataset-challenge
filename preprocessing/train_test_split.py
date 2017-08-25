import csv
import json
import random

random.seed(42)

instance_file = open("yelp_stars_text.json")
instances = instance_file.read()

train_instances = []
test_instances = []

json_arr = json.loads(instances)
for instance in json_arr:
    if random.randint(0, 2) == 0:
        test_instances.append(instance)
    else:
        train_instances.append(instance)

train_output = open("../textprocessing/train_data.json", 'w')
test_output = open("../textprocessing/test_data.json", 'w')

json.dump(train_instances, train_output)
json.dump(test_instances, test_output)
