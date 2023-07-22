import pickle
import os
import numpy as np
import json
import torch
from tqdm import tqdm
import argparse
import warnings 
from sentence_transformers import SentenceTransformer, util
warnings.filterwarnings("ignore")

from utils import segment_long_text, sbert_rank_paragraphs

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", default="alqac23_data", type=str, help="directory to raw data")
    parser.add_argument("--sbert_model_path", default="", type=str, help="path to sentence bert model")
    parser.add_argument("--corpus_name", default="alqac23", type=str, choices=["alqac23", "alqac22", "zalo", "all"], help="corpus name for mining")
    parser.add_argument("--top_k", default=20, type=int, help="top k documents for negative sample mining")
    parser.add_argument("--load_embedding", action="store_true", help="load pre-computed embedding")
    parser.add_argument("--save_dir", default="generated_data", type=str)
    args = parser.parse_args()

    # get sbert model name
    sbert_model_name = os.path.basename(args.sbert_model_path)

    # load training data from json
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
    
    print(len(train_items))

    with open(f"generated_data/{args.corpus_name}_flat_corpus.pkl", "rb") as flat_corpus_file:
        flat_corpus_data = pickle.load(flat_corpus_file)

    with open(f"generated_data/{args.corpus_name}_legal_dict.json") as legal_dict_file:
        doc_data = json.load(legal_dict_file)
    
    # load sbert model
    model = SentenceTransformer(args.sbert_model_path)

    # pre-compute embedding data and save  
    if not args.load_embedding:
        embeddings = []
        for k, v in tqdm(doc_data.items()):
            embedded = model.encode(v['text'])
            embeddings.append(embedded)
        np_embeddings = np.array(embeddings)

        with open(f'{args.save_dir}/{args.corpus_name}_embedding_{sbert_model_name}.pkl', 'wb') as pkl:
            pickle.dump(np_embeddings, pkl)

    with open(f'{args.save_dir}/{args.corpus_name}_embedding_{sbert_model_name}.pkl', 'rb') as pkl:
        corpus_embeddings = pickle.load(pkl)

    pred_list = []
    top_k = args.top_k
    save_pairs = []
    max_length = 2000
    sliding_thresh = 1000
    for idx, item in tqdm(enumerate(train_items)):
        question = item["text"]
        relevant_articles = item["relevant_articles"]
        actual_positive = len(relevant_articles)
        
        for article in relevant_articles:
            save_dict = {}
            save_dict["question"] = question
            concat_id = article["law_id"] + "_" + article["article_id"]
            text = doc_data[concat_id]["text"]

            if len(text) - text.count("\n") <= max_length:                
                save_dict["document"] = text
            else:
                paragraphs = segment_long_text(text, max_length, sliding_thresh)
                candidate_text = sbert_rank_paragraphs(question, paragraphs, model, util)  
                save_dict["document"] = candidate_text
            save_dict["relevant"] = 1
            save_pairs.append(save_dict)
        
        encoded_question  = model.encode(question)

        all_cosine = util.cos_sim(encoded_question, corpus_embeddings).numpy().squeeze(0)
        predictions = np.argpartition(all_cosine, -top_k)[-top_k:]        
        
        for idx, idx_pred in enumerate(predictions):
            pred = flat_corpus_data[idx_pred]
                
            check = 0
            for article in relevant_articles:
                check += 1 if pred[0] == article["law_id"] and pred[1] == article["article_id"] else 0

            if check == 0:
                save_dict = {}
                save_dict["question"] = question
                concat_id = pred[0] + "_" + pred[1]
                text = doc_data[concat_id]["text"]

                if len(text) - text.count("\n") <= max_length:                
                    save_dict["document"] = text
                else:
                    paragraphs = segment_long_text(text, max_length, sliding_thresh)
                    candidate_text = sbert_rank_paragraphs(question, paragraphs, model, util)  
                    save_dict["document"] = candidate_text
                save_dict["relevant"] = 0
                save_pairs.append(save_dict)

    save_path = args.save_dir
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, f"{args.corpus_name}_qrel_pairs_{sbert_model_name}_top{top_k}_v2.pkl"), "wb") as pair_file:
        pickle.dump(save_pairs, pair_file)
