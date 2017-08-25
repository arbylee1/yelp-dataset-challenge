import csv
import json

restaurants = []
restaurants_file = open("yelp_english_restaurants.csv")
restaurants_reader = csv.reader(restaurants_file)
for id in restaurants_reader:
    restaurants.append(id)

source_file = open("yelp_academic_dataset_review.json")
stars_text_file = open("yelp_stars_text.json", 'w')

reviews = []

for line in source_file:
    review = json.loads(line)
    if restaurants[0].__contains__(review["business_id"]):
        reviews.append({"stars":review["stars"],"text":review["text"]})

json.dump(reviews, stars_text_file)