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
    data_path = "alqac23_data"
    save_bm25 = "saved_model"
    top_k_bm25 = 2
    bm25_k1 = 2.0
    bm25_b = 0.75

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # load document to save running time, 
    # must re-run if we change pre-process step
    parser.add_argument("--load_docs", action="store_false")
    parser.add_argument("--num_eval", default=500, type=str)
    args = parser.parse_args()
    cfg = Config()
    
    save_path = cfg.save_bm25
    os.makedirs(save_path, exist_ok=True)

    raw_data = cfg.data_path
    alqac23_corpus_path = os.path.join(raw_data, "law.json")
    alqac22_corpus_path = os.path.join(raw_data, "additional_data/ALQAC_2022_training_data/law.json")
    zalo_corpus_path = os.path.join(raw_data, "additional_data/zalo/zalo_corpus.json")

    corpus_paths = [
        alqac23_corpus_path,
        alqac22_corpus_path,
        # zalo_corpus_path
    ]
    
    data = []
    for corpus_path in corpus_paths:
        data.extend(json.load(open(corpus_path)))

    if args.load_docs:
        print("Process documents")
        documents = []
        doc_refers = []
        for law_article in tqdm(data):
            law_id = law_article["id"]
            law_articles = law_article["articles"]
            
            for sub_article in law_articles:
                article_id = sub_article["id"]
                article_text = sub_article["text"]
                    
                tokens = bm25_tokenizer(article_text)
                documents.append(tokens)
                doc_refers.append([law_id, article_id, article_text])
        
        with open(os.path.join(save_path, "documents_manual"), "wb") as documents_file:
            pickle.dump(documents, documents_file)
        with open(os.path.join(save_path,"doc_refers_saved"), "wb") as doc_refer_file:
            pickle.dump(doc_refers, doc_refer_file)
    else:
        with open(os.path.join(save_path, "documents_manual"), "rb") as documents_file:
            documents = pickle.load(documents_file)
        with open(os.path.join(save_path,"doc_refers_saved"), "rb") as doc_refer_file:
            doc_refers = pickle.load(doc_refer_file)
            

    # Grid_search, evaluate on training questions
    alqac23_corpus_path_train = os.path.join(raw_data, "train.json")
    alqac22_corpus_path_train = os.path.join(raw_data, "additional_data/ALQAC_2022_training_data/question.json")
    zalo_corpus_path_train = os.path.join(raw_data, "additional_data/zalo/zalo_question.json")
     
    train_corpus_paths = [
        alqac23_corpus_path_train,
        alqac22_corpus_path_train,
        # zalo_corpus_path_train
    ]
    
    train_items = []
    for train_corpus_path in train_corpus_paths:
        train_items.extend(json.load(open(train_corpus_path)))

    print (f"Number of training questions: {len(train_items)}")

    bm25 = BM25Plus(documents, k1=cfg.bm25_k1, b=cfg.bm25_b)
    # bm25 = BM25Plus(documents) # with default {k1,b} params    
        
    total_f2 = 0
    total_precision = 0
    total_recall = 0
    
    k = args.num_eval
    for idx, item in tqdm(enumerate(train_items)):
        if idx >= k:
            continue
        qid = "question_id" if "question_id" in item.keys() else "id"
        question_id = item[qid]
        question = item["text"]
        relevant_articles = item["relevant_articles"]
        actual_positive = len(relevant_articles)
        
        tokenized_query = bm25_tokenizer(question)
        doc_scores = bm25.get_scores(tokenized_query)
        
        # Get top N
        # N large -> reduce precision, increase recall
        # N small -> increase precision, reduce recall
        predictions = np.argpartition(doc_scores, len(doc_scores) - cfg.top_k_bm25)[-cfg.top_k_bm25:]

        # suggest to investigate delta score for tricking
        #TODO: calculate histogram of the delta_scores to cutoff at right place

        # Trick to balance precision and recall
        if doc_scores[predictions[1]] - doc_scores[predictions[0]] >= 2.7:
            predictions = [predictions[1]]

        true_positive = 0
        false_positive = 0
        for idx, idx_pred in enumerate(predictions):
            pred = doc_refers[idx_pred]            
            
            # suggest to investigate/pick thresh_score
            #TODO: calculate histogram of the thresh_score to cutoff at right place

            thresh_score = 20
            # if doc_scores[idx_pred] < thresh_score:
            #     # print(pred)
            #     print(doc_scores[idx_pred])
            
            # Remove prediction with too low score: 20
            if doc_scores[idx_pred] >= thresh_score:
                for article in relevant_articles:
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
                           f"bm25plus_k{cfg.bm25_k1}_b{cfg.bm25_b}_F2_{100*total_f2/k:0.0f}_P_{100*total_precision/k:0.0f}_R_{100*total_recall/k:0.0f}"), "wb"
                           ) as bm_file:
        pickle.dump(bm25, bm_file)