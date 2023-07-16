import os
import json
from tqdm import tqdm
import pickle
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", default="alqac23_data", type=str, help="directory to raw data")
    parser.add_argument("--corpus_name", default="alqac23", type=str, choices=["alqac23", "alqac22", "zalo", "all"], help="corpus name")
    parser.add_argument("--save_dir", default="generated_data", type=str, help="directory to save flat corpus.")
    args = parser.parse_args()

    alqac23_corpus_path = os.path.join(args.raw_data_dir, "law.json")
    alqac22_corpus_path = os.path.join(args.raw_data_dir, "additional_data/ALQAC_2022_training_data/law.json")
    zalo_corpus_path = os.path.join(args.raw_data_dir, "additional_data/zalo/zalo_corpus.json")
        
    corpus_paths = {
        "alqac23": alqac23_corpus_path,
        "alqac22": alqac22_corpus_path,
        "zalo": zalo_corpus_path
    }

    data = []
    if args.corpus_name == "all":
        data_paths = list(corpus_paths.values())
        for data_path in data_paths:
            data.extend(json.load(open(data_path)))
    else:
        data_path = corpus_paths[args.corpus_name]
        data = json.load(open(data_path))
        
    print("=======================")
    print(f"Start create flat corpus for {args.corpus_name}")
    flat_corpus = []
    for law_article in tqdm(data):
        law_id = law_article["id"]
        law_articles = law_article["articles"]
        
        for sub_article in law_articles:
            article_id = sub_article["id"]
            article_text = sub_article["text"]
            flat_corpus.append([law_id, article_id, article_text])
    
    os.makedirs(args.save_dir, exist_ok=True)    
    with open(os.path.join(args.save_dir, f"{args.corpus_name}_flat_corpus.pkl"), "wb") as flat_corpus_file:
        pickle.dump(flat_corpus, flat_corpus_file)
    print("Flat corpus created")