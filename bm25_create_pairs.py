import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from rank_bm25 import *
from utils import bm25_tokenizer
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_pair", default=20, type=int)
    parser.add_argument("--model_path", default="saved_model/bm25plus_k1.5_b0.75_F2_79_P_68_R_85", type=str)
    parser.add_argument("--data_path", default="alqac23_data", type=str, help="path to input data")
    parser.add_argument("--save_pair_path", default="generated_data/", type=str, help="path to save pair sentence directory")
    args = parser.parse_args()
   
    alqac23_corpus_path_train = os.path.join(args.data_path, "train.json")
    alqac22_corpus_path_train = os.path.join(args.data_path, "additional_data/ALQAC_2022_training_data/question.json")
    zalo_corpus_path_train = os.path.join(args.data_path, "additional_data/zalo/zalo_question.json")
     
    train_corpus_paths = [
        alqac23_corpus_path_train,
        alqac22_corpus_path_train,
        # zalo_corpus_path_train
    ]
    
    train_items = []    
    for train_corpus_path in train_corpus_paths:
        train_items.extend(json.load(open(train_corpus_path)))

    with open(args.model_path, "rb") as bm_file:
        bm25 = pickle.load(bm_file)
    with open("generated_data/flattened_corpus.pkl", "rb") as flat_corpus_file:
        flattened_copus = pickle.load(flat_corpus_file)

    doc_data = json.load(open("generated_data/legal_dict.json"))

    save_pairs = []
    top_n = args.top_pair
    for idx, item in tqdm(enumerate(train_items)):        
        question = item["text"]
        relevant_articles = item["relevant_articles"]
        actual_positive = len(relevant_articles)
        
        tokenized_query = bm25_tokenizer(question)
        doc_scores = bm25.get_scores(tokenized_query)

        predictions = np.argpartition(doc_scores, len(doc_scores) - top_n)[-top_n:]

        # Save positive pairs
        for article in relevant_articles:
            save_dict = {}
            save_dict["question"] = question
            concat_id = article["law_id"] + "_" + article["article_id"]
            save_dict["document"] = doc_data[concat_id]["text"]
            save_dict["relevant"] = 1
            save_pairs.append(save_dict)

        # Save negative pairs
        for idx, idx_pred in enumerate(predictions):
            pred = flattened_copus[idx_pred]

            check = 0
            for article in relevant_articles:
                if pred[0] == article["law_id"] and pred[1] == article["article_id"]:
                    check += 1
 
            if check == 0:
                save_dict = {}
                save_dict["question"] = question
                concat_id = pred[0] + "_" + pred[1]
                save_dict["document"] = doc_data[concat_id]["text"]
                save_dict["relevant"] = 0
                save_pairs.append(save_dict)
                    
    save_path = args.save_pair_path
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, f"qrel_pairs_bm25_top{top_n}"), "wb") as pair_file:
        pickle.dump(save_pairs, pair_file)
    print (f"Number of pairs: {len(save_pairs)}")