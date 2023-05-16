# 将tmpnn_v8.jsonl里的数据随机分成train,validation,test三个部分，validation和test的数量为2000，其余为train数据集
# 将所划分蛋白质的名字记录到split.json文件中，如果名字不是以_A结尾，则加上_A
import utils
dataset = utils.load_jsonl(json_file="/pubhome/bozhang/data/tmpnn_v8.jsonl")
import random
random.shuffle(dataset)
train_dataset = dataset[4000:]
validation_dataset = dataset[:2000]
test_dataset = dataset[2000:4000]
train_names = []
validation_names = []
test_names = []
for data in train_dataset:
    name = data["name"]
    if not name.endswith("_A"):
        name = name+"_A"
    train_names.append(name)
for data in validation_dataset:
    name = data["name"]
    if not name.endswith("_A"):
        name = name+"_A"
    validation_names.append(name)
for data in test_dataset:
    name = data["name"]
    if not name.endswith("_A"):
        name = name+"_A"
    test_names.append(name)

import json
with open("/pubhome/xtzhang/data/random_split.json","w") as f:
    json.dump({"train":train_names,"validation":validation_names,"test":test_names},f)
