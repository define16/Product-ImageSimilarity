import json
import os.path

with open(os.path.join("..", "data", "same_name.json")) as f:
    data = json.load(f)

data_set = []
for product_name, value in data.items():
    data_set.append((product_name, value[0].get("대표이미지")))

print(data_set)
