import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from rank_bm25 import *
import argparse
from utils import bm25_tokenizer, calculate_f2
# from config import Config

class Config:
    raw_data_dir = "alqac23_data"
    save_bm25 = "saved_model/bm25"
    top_k_bm25 = 2
    bm25_k1 = 1.5
    bm25_b = 0.75

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # load document to save running time, 
    # must re-run if we change pre-process step
    parser.add_argument("--corpus_name", default="alqac23", type=str, choices=["alqac23", "alqac22", "zalo"], help="corpus for bm25")
    parser.add_argument("--load_docs", action="store_false")
    parser.add_argument("--num_eval", default=500, type=str)
    args = parser.parse_args()

    cfg = Config()
    
    save_path = cfg.save_bm25
    os.makedirs(save_path, exist_ok=True)

    raw_data = cfg.raw_data_dir
    alqac23_corpus_path = os.path.join(raw_data, "law.json")
    alqac22_corpus_path = os.path.join(raw_data, "additional_data/ALQAC_2022_training_data/law.json")
    zalo_corpus_path = os.path.join(raw_data, "additional_data/zalo/zalo_corpus.json")

    corpus_paths = {
        "alqac23": alqac23_corpus_path,
        "alqac22": alqac22_corpus_path,
        "zalo": zalo_corpus_path
    }    
    
    data_path = corpus_paths[args.corpus_name]
    data = json.load(open(data_path))

    if args.load_docs:
        print("Process documents")
        documents = []
        flat_corpus_data = []
        for law_article in tqdm(data):
            law_id = law_article["id"]
            law_articles = law_article["articles"]
            
            for sub_article in law_articles:
                article_id = sub_article["id"]
                article_text = sub_article["text"]
                    
                tokens = bm25_tokenizer(article_text)
                documents.append(tokens)
                flat_corpus_data.append([law_id, article_id, article_text])
        
        with open(f"generated_data/{args.corpus_name}_bm25_tokenized_corpus.pkl", "wb") as documents_file:
            pickle.dump(documents, documents_file)
        with open(f"generated_data/{args.corpus_name}_flat_corpus.pkl", "wb") as flat_corpus_file:
            pickle.dump(flat_corpus_data, flat_corpus_file)
    else:
        with open(f"generated_data/{args.corpus_name}_bm25_tokenized_corpus.pkl", "rb") as documents_file:
            documents = pickle.load(documents_file)
        with open(f"generated_data/{args.corpus_name}_flat_corpus.pkl", "rb") as flat_corpus_file:
            flat_corpus_data = pickle.load(flat_corpus_file)
            

    # Grid_search, evaluate on training questions
    alqac23_corpus_path_train = os.path.join(raw_data, "train.json")
    alqac22_corpus_path_train = os.path.join(raw_data, "additional_data/ALQAC_2022_training_data/question.json")
    zalo_corpus_path_train = os.path.join(raw_data, "additional_data/zalo/zalo_question.json")
    
    train_corpus_paths = {
        "alqac23": alqac23_corpus_path_train,
        "alqac22": alqac22_corpus_path_train,
        "zalo": zalo_corpus_path_train
    }
    
    train_items = []
    train_corpus_path = train_corpus_paths[args.corpus_name]
    train_items = json.load(open(train_corpus_path))

    print (f"Number of training questions: {len(train_items)}")

    bm25 = BM25Plus(documents, k1=cfg.bm25_k1, b=cfg.bm25_b)
    # bm25 = BM25Plus(documents) # with default {k1,b} params    
        
    total_f2 = 0
    total_precision = 0
    total_recall = 0
    
    k = args.num_eval if args.num_eval < len(train_items) else len(train_items)
    for idx, item in tqdm(enumerate(train_items)):
        if idx >= k:
            break        
        question = item["text"]
        relevant_articles = item["relevant_articles"]
        actual_positive = len(relevant_articles)

        # print(question)
        
        tokenized_query = bm25_tokenizer(question)
        doc_scores = bm25.get_scores(tokenized_query)
        
        # Get top N
        # N large -> reduce precision, increase recall
        # N small -> increase precision, reduce recall
        predictions = np.argpartition(doc_scores, -cfg.top_k_bm25)[-cfg.top_k_bm25:]

        # suggest to investigate delta score for tricking
        #TODO: calculate histogram of the delta_scores to cutoff at right place

        # Trick to balance precision and recall
        if doc_scores[predictions[1]] - doc_scores[predictions[0]] >= 2.7:
            predictions = [predictions[1]]

        true_positive = 0
        false_positive = 0
        for idx, idx_pred in enumerate(predictions):
            pred = flat_corpus_data[idx_pred]            
            
            # suggest to investigate/pick thresh_score
            #TODO: calculate histogram of the thresh_score to cutoff at right place

            thresh_score = 20
            # if doc_scores[idx_pred] < thresh_score:
            #     # print(pred)
            #     print(doc_scores[idx_pred])
            
            # print(doc_scores[idx_pred])
            # print(pred[0], pred[1])

            # Remove prediction with too low score: 20
            if doc_scores[idx_pred] >= thresh_score:
                for article in relevant_articles:
                    # print(article["law_id"], article["article_id"])

                    if pred[0] == article["law_id"] and pred[1] == article["article_id"]:
                        true_positive += 1
                    else:
                        false_positive += 1
                    
        precision = true_positive/(true_positive + false_positive + 1e-20)
        recall = true_positive/actual_positive
        f2 = calculate_f2(precision, recall)
        total_precision += precision
        total_recall += recall
        total_f2 += f2
    
    print(f"Average F2: {total_f2/k:0.3f}")
    print(f"Average Precision: {total_precision/k:0.3f}")
    print(f"Average Recall: {total_recall/k:0.3f}")

    # save the model with its performance
    with open(os.path.join(save_path, 
                           f"{args.corpus_name}_bm25plus_k{cfg.bm25_k1}_b{cfg.bm25_b}"), "wb"
                           ) as bm_file:
        pickle.dump(bm25, bm_file)