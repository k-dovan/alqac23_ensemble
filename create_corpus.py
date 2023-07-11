import json
import os
import re
from tqdm import tqdm
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="alqac23_data", type=str, help="path to training data")
    parser.add_argument("--save_dir", default="generated_data", type=str, help="path to training data")
    args = parser.parse_args()
    os.makedirs(args.save_dir,exist_ok=True)

    cp = open(os.path.join(args.save_dir, "corpus.txt"), "w")
    alqac23_corpus_path = os.path.join(args.data_dir, "law.json")
    alqac22_corpus_path = os.path.join(args.data_dir, "additional_data/ALQAC_2022_training_data/law.json")
    zalo_corpus_path = os.path.join(args.data_dir, "additional_data/zalo/zalo_corpus.json")
        
    corpus_paths = [
        alqac23_corpus_path,
        alqac22_corpus_path,
        # zalo_corpus_path
    ]
    
    data = []
    for corpus_path in corpus_paths:
        data.extend(json.load(open(corpus_path)))

    save_dict = {}
    co_f = open(os.path.join(args.save_dir, "cocondenser_data.json"), "w")
    count = 0
    for law_article in tqdm(data):
        law_id = law_article["id"]
        law_articles = law_article["articles"]
        
        for sub_article in law_articles:
            article_id = sub_article["id"]
            article_text = sub_article["text"]
            article_full = article_text.replace("\n", " ")
            cp.write(article_full + "\n")
            
            # Save data for cocondenser 
            spans = []
            passages = re.split(r"\n[0-9]+\. |1\. ", article_text)
            for idx, p in enumerate(passages):
                if p != "":
                    spans.append(p)
            co_f.write("#".join(spans) + "\n")
            
            concat_id = law_id + "_" + article_id
            if concat_id not in save_dict:
                count += 1
                save_dict[concat_id] = {"text": article_text}
    
    co_f.close()
    print(count)
        
    print("Create legal dict from raw data")
    with open(os.path.join(args.save_dir, "legal_dict.json"), "w") as outfile:
        json.dump(save_dict, outfile)
    print("Finish")

    # enrich corpus from train files
    alqac23_corpus_path_train = os.path.join(args.data_dir, "train.json")
    alqac22_corpus_path_train = os.path.join(args.data_dir, "additional_data/ALQAC_2022_training_data/question.json")
    zalo_corpus_path_train = os.path.join(args.data_dir, "additional_data/zalo/zalo_question.json")
     
    train_corpus_paths = [
        alqac23_corpus_path_train,
        alqac22_corpus_path_train,
        zalo_corpus_path_train
    ]
    
    train_items = []
    for train_corpus_path in train_corpus_paths:
        train_items.extend(json.load(open(train_corpus_path)))

    for item in tqdm(train_items):
        question = item["text"]
        cp.write(question + "\n")

    # enrich corpus from public test file
    alqac23_corpus_path_test = os.path.join(args.data_dir, "public_test.json")
    public_items = json.load(open(alqac23_corpus_path_test))

    for item in tqdm(public_items):
        question = item["text"]
        cp.write(question + "\n")

    cp.close()
