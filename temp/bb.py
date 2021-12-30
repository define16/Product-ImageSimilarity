import json

with open("result.json", "r") as f:
    data = json.load(f)

# with open("product_names.csv", "w") as f:
#     for name, value in data.items():
#         for v in value:
#             v = v.replace('<b>', '')
#             v = v.replace('</b>', '')
#             f.write(name + ',' + v + '\n')

result_word_counts = {}
for name, value in data.items():
    word_counts = {}
    for v in value:
        v = v.replace('<b>', '')
        v = v.replace('</b>', '')
        v = v.replace('[', '')
        v = v.replace(']', '')
        words = v.split()
        for word in words:
            count = word_counts.get(word, 0)
            count += 1
            word_counts[word] = count
    result_word_counts[name] = word_counts

with open("result_word_counts.csv", "w") as f:
    for name, value in result_word_counts.items():
        for k, v in value.items():
            if v > 1:
                f.write(name + ',' + k + ',' + str(v) + '\n')
