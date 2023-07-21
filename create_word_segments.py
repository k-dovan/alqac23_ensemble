
# ========================================================================================================
# This file is for mining stop words and useless characters on specific data corpus.
# ========================================================================================================

import os
import argparse
import json
from tqdm import tqdm
from underthesea import word_tokenize

from utils import bm25_tokenizer

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", default="alqac23_data", type=str, help="path to raw corpus")
    parser.add_argument("--corpus_name", default="alqac23", type=str, choices=["alqac23", "alqac22", "zalo"], help="corpus to segment words")
    parser.add_argument("--save_path", default="generated_data/experiments", type=str, help="path to save segmented words")
    args = parser.parse_args()

    # articles' paths
    alqac23_corpus_path = os.path.join(args.raw_data_dir, "law.json")
    alqac22_corpus_path = os.path.join(args.raw_data_dir, "additional_data/ALQAC_2022_training_data/law.json")
    zalo_corpus_path = os.path.join(args.raw_data_dir, "additional_data/zalo/zalo_corpus.json")
 
    corpus_paths = {
        "alqac23": alqac23_corpus_path,
        "alqac22": alqac22_corpus_path,
        "zalo": zalo_corpus_path
    }

    corpus_name = args.corpus_name
    corpus_path = corpus_paths[corpus_name]    
    data = json.load(open(corpus_path))
            
    segmented_words = []
    for law_article in tqdm(data):
        law_articles = law_article["articles"]
        
        for sub_article in law_articles:
            text = sub_article["text"]
            words = bm25_tokenizer(text)
            segmented_words.append(words)

    with open(f"{args.save_path}/{corpus_name}_segmented_words.txt", 'w', encoding="utf-8") as f:
        for w in tqdm(segmented_words):
            f.write("%s\n" % w)


        