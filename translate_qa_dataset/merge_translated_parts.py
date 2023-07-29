import os
from os import listdir
import json

if __name__ == "__main__":

    translated_dir = "../generated_data/translated/"
    translated_file_names = [name for name in listdir(translated_dir) if name.endswith('.json')]
    translated_train_names = [name for name in translated_file_names if "translated_train" in name]
    translated_dev_names = [name for name in translated_file_names if "translated_dev" in name]

    train_data = []
    for f_name in translated_train_names:
        train_data.extend(json.load(open(f"{translated_dir}/{f_name}", "r", encoding="utf-8")))
    with open(f"{translated_dir}/translated_train.json", "w", encoding="utf-8") as f:
        json_object = json.dumps(train_data, indent=4, ensure_ascii=False)
        f.write(json_object)
    
    dev_data = []
    for f_name in translated_dev_names:
        dev_data.extend(json.load(open(f"{translated_dir}/{f_name}", "r", encoding="utf-8")))
    with open(f"{translated_dir}/translated_dev.json", "w", encoding="utf-8") as f:
        json_object = json.dumps(dev_data, indent=4, ensure_ascii=False)
        f.write(json_object)

    print (f"Total train items: {len(train_data)}")
    print (f"Total dev items: {len(dev_data)}")