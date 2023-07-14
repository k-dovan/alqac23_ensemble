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

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--sbert_model_path", default="", type=str, help="path to round 1 sentence bert model")
    parser.add_argument("--data_path", default="alqac23_data", type=str, help="path to input data")
    parser.add_argument("--save_path", default="generated_data", type=str)
    parser.add_argument("--top_k", default=20, type=int, help="top k hard negative mining")
    parser.add_argument("--flattened_corpus", default="generated_data/flattened_corpus.pkl", type=str, help="path to flattened corpus")
    parser.add_argument("--legal_dict_path", default="generated_data/legal_dict.json", type=str, help="path to legal dict")
    parser.add_argument("--load_embedding", action="store_true", help="load pre-computed embedding")
    args = parser.parse_args()

    # get sbert model name
    sbert_model_name = os.path.basename(args.sbert_model_path)

    # load training data from json
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
    
    print(len(train_items))

    with open(args.flattened_corpus, "rb") as flat_corpus_file:
        flat_corpus_data = pickle.load(flat_corpus_file)

    with open(args.legal_dict_path) as legal_dict_file:
        doc_data = json.load(legal_dict_file)
    
    # load sbert model
    model = SentenceTransformer(args.sbert_model_path)

    # pre-compute embedding data and save  
    if not args.load_embedding:
        embed_list = []
        for k, v in tqdm(doc_data.items()):
            embed = model.encode(v['text'])
            doc_data[k]['embedding'] = embed

        with open(f'{args.save_path}/corpus_embedding_{sbert_model_name}.pkl', 'wb') as pkl:
            pickle.dump(doc_data, pkl)

    with open(f'{args.save_path}/corpus_embedding_{sbert_model_name}.pkl', 'rb') as pkl:
        data = pickle.load(pkl)

    pred_list = []
    top_k = args.top_k
    save_pairs = []

    for idx, item in tqdm(enumerate(train_items)):
        question = item["text"]
        relevant_articles = item["relevant_articles"]
        actual_positive = len(relevant_articles)
        
        for article in relevant_articles:
            save_dict = {}
            save_dict["question"] = question
            concat_id = article["law_id"] + "_" + article["article_id"]
            save_dict["document"] = doc_data[concat_id]["text"]
            save_dict["relevant"] = 1
            save_pairs.append(save_dict)
        
        encoded_question  = model.encode(question)
        list_embs = []

        for k, v in data.items():
            emb_2 = torch.tensor(v['embedding']).unsqueeze(0)
            list_embs.append(emb_2)

        matrix_emb = torch.cat(list_embs, dim=0)
        all_cosine = util.cos_sim(encoded_question, matrix_emb).numpy().squeeze(0)
        predictions = np.argpartition(all_cosine, len(all_cosine) - top_k)[-top_k:]
        
        
        for idx, idx_pred in enumerate(predictions):
            pred = flat_corpus_data[idx_pred]
                
            check = 0
            for article in relevant_articles:
                check += 1 if pred[0] == article["law_id"] and pred[1] == article["article_id"] else 0

            if check == 0:
                save_dict = {}
                save_dict["question"] = question
                concat_id = pred[0] + "_" + pred[1]
                save_dict["document"] = doc_data[concat_id]["text"]
                save_dict["relevant"] = 0
                save_pairs.append(save_dict)

    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, f"qrel_pairs_{sbert_model_name}_top{top_k}.pkl"), "wb") as pair_file:
        pickle.dump(save_pairs, pair_file)
