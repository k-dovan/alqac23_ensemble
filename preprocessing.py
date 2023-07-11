import os
import json
from tqdm import tqdm
import pickle
from utils import bm25_tokenizer
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", default="alqac23_data", type=str, help="path to raw data")
    parser.add_argument("--save_path", default="generated_data", type=str, help="path to save doc refer.")
    args = parser.parse_args()

    alqac23_corpus_path = os.path.join(args.raw_data, "law.json")
    alqac22_corpus_path = os.path.join(args.raw_data, "additional_data/ALQAC_2022_training_data/law.json")
    zalo_corpus_path = os.path.join(args.raw_data, "additional_data/zalo/zalo_corpus.json")
        
    corpus_paths = [
        alqac23_corpus_path,
        alqac22_corpus_path,
        zalo_corpus_path
    ]
    
    data = []
    for corpus_path in corpus_paths:
        data.extend(json.load(open(corpus_path)))
        
    print("=======================")
    print("Start create flattened corpus.")
    flattened_corpus = []
    for law_article in tqdm(data):
        law_id = law_article["id"]
        law_articles = law_article["articles"]
        
        for sub_article in law_articles:
            article_id = sub_article["id"]
            article_text = sub_article["text"]
            flattened_corpus.append([law_id, article_id, article_text])
    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path,"flattened_corpus.pkl"), "wb") as flat_corpus_file:
        pickle.dump(flattened_corpus, flat_corpus_file)
    print("Created Doc Data.")