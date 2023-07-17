import os
import json
import pickle
from re import S
from unicodedata import name
import numpy as np
from tqdm import tqdm
from rank_bm25 import *
from utils import bm25_tokenizer, calculate_f2
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", default="alqac23_data", type=str, help="directory to raw data")
    parser.add_argument("--model_path", default="saved_model/bm25/all_bm25plus_k1.5_b0.75", type=str)
    parser.add_argument("--corpus_name", default="alqac23", type=str, choices=["alqac23", "alqac22", "zalo", "all"], help="corpus for bm25")
    parser.add_argument("--top_k", default=20, type=int)
    parser.add_argument("--save_dir", default="generated_data", type=str, help="path to save pair sentences")
    args = parser.parse_args()
   
    alqac23_corpus_path_train = os.path.join(args.raw_data_dir, "train.json")
    alqac22_corpus_path_train = os.path.join(args.raw_data_dir, "additional_data/ALQAC_2022_training_data/question.json")
    zalo_corpus_path_train = os.path.join(args.raw_data_dir, "additional_data/zalo/zalo_question.json")
    
    train_corpus_paths = {
        "alqac23": alqac23_corpus_path_train,
        "alqac22": alqac22_corpus_path_train,
        "zalo": zalo_corpus_path_train
    }    
    
    train_items = []
    if args.corpus_name == "all":
        train_paths = list(train_corpus_paths.values())
        for train_path in train_paths:
            train_items.extend(json.load(open(train_path)))
    else:
        train_path = train_corpus_paths[args.corpus_name]
        train_items = json.load(open(train_path))

    with open(args.model_path, "rb") as bm_file:
        bm25 = pickle.load(bm_file)
    with open(f"{args.save_dir}/{args.corpus_name}_flat_corpus.pkl", "rb") as flat_corpus_file:
        flat_corpus_data = pickle.load(flat_corpus_file)

    doc_data = json.load(open(f"{args.save_dir}/{args.corpus_name}_legal_dict.json"))

    save_pairs = []

    total_f2 = 0
    total_precision = 0
    total_recall = 0
    k = len(train_items)
    top_n = args.top_k
    for idx, item in tqdm(enumerate(train_items)):
        question = item["text"]
        relevant_articles = item["relevant_articles"]
        actual_positive = len(relevant_articles)
        
        tokenized_query = bm25_tokenizer(question)
        doc_scores = bm25.get_scores(tokenized_query)
        # top_pred = np.sort(doc_scores)[-3:]
        # #top_pred = np.unique(top_pred, axis=0)
        # predictions = []
        # for top in top_pred:
        #     predictions.append(np.where(doc_scores == top)[0][0])
        predictions = np.argpartition(doc_scores, -top_n)[-top_n:]
        # if doc_scores[predictions[1]] - doc_scores[predictions[0]] >= 2.6:
        #      predictions = [predictions[1]]
        
        for article in relevant_articles:
            save_dict = {}
            save_dict["question"] = question
            concat_id = article["law_id"] + "_" + article["article_id"]
            save_dict["document"] = doc_data[concat_id]["text"]
            save_dict["relevant"] = 1
            save_pairs.append(save_dict)
        # print(question)
        # print(relevant_articles)

        true_positive = 0
        false_positive = 0
        for idx, idx_pred in enumerate(predictions):
            pred = flat_corpus_data[idx_pred]
                
            #print(pred, doc_scores[idx_pred])
            #if doc_scores[idx_pred] >= 20:
            check = 0
            for article in relevant_articles:
                if pred[0] == article["law_id"] and pred[1] == article["article_id"]:
                    true_positive += 1
                    check += 1
                    #print(doc_data[pred[0] + "_" + pred[1]])
                else:
                    false_positive += 1
            
            if check == 0:
                save_dict = {}
                save_dict["question"] = question
                concat_id = pred[0] + "_" + pred[1]
                save_dict["document"] = doc_data[concat_id]["text"]
                save_dict["relevant"] = 0
                save_pairs.append(save_dict)
                    
        if true_positive + false_positive == 0:
            precision = 0
        else:
            precision = true_positive/(true_positive + false_positive)
        recall = true_positive/actual_positive
        f2 = calculate_f2(precision, recall)
        total_precision += precision
        total_recall += recall
        total_f2 += f2
        
    print(f"Average F2: {total_f2/k:0.3f}")
    print(f"Average Precision: {total_precision/k:0.3f}")
    print(f"Average Recall: {total_recall/k:0.3f}")

    save_path = args.save_dir
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, f"{args.corpus_name}_qrel_pairs_bm25_top{top_n}"), "wb") as pair_file:
        pickle.dump(save_pairs, pair_file)
    print (f"Number of pairs saved: {len(save_pairs)}")