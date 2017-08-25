import json

file = open('yelp_academic_dataset_business.json')
output = open('yelp_english_restaurants.csv','w')
first = True
banned_cities = {"Karlsruhe", "Montreal"}
for line in file:
    restaurant = json.loads(line)
    if not banned_cities.__contains__(restaurant["city"]):
        if restaurant["categories"].__contains__("Restaurants"):
            if first:
                output.write(restaurant["business_id"])
                first = False
            else:
                output.write((',' + restaurant["business_id"]))

file.close()
output.close()