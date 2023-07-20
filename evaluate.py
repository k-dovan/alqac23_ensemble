import numpy as np
import json
import torch
from tqdm import tqdm
from rank_bm25 import *
import argparse
import os
import pickle
import glob
from utils import bm25_tokenizer, calculate_f2

from sentence_transformers import SentenceTransformer, util

# BM25 model used in `ensemble` evaluation mode
BM25_MODEL = "bm25/<<<corpus_name>>>_bm25plus_k1.5_b0.75"

# ----------------------------------------------------------------------------------------
# Set up data version to pick proper pre-embedded data for evaluation and testing
# 
# Note: We must change appropriate items for SBERT_MODELS_ROUND1, SBERT_MODELS_ROUND2
# whenever we change data version.
# 
# Models for data v1 without and v2 with `all_corpora_ep5` in their names.
#
# ----------------------------------------------------------------------------------------
# v1 - data version used alqac23 & alqac22 for training LMs, Co-/Condenser, SBerts
# v2 - data version used alqac23, alqac22 & zalo for training LMs, Co-/Condenser, SBerts
SBERT_TRAINED_DATA_VERSION = "v2"

# candidate models for round 1 (single and/or ensemble)
SBERT_MODELS_ROUND1 = [
    "sbert_round1_epoch4_top50_mlm_finetuned_vibert_base_cased_all_corpora_ep5",
    "sbert_round1_epoch4_top50_mlm_finetuned_phobert_large_all_corpora_ep5",
    "sbert_round1_epoch4_top50_condenser_phobert_large_all_corpora_ep5",
    "sbert_round1_epoch4_top50_cocondenser_phobert_large_all_corpora_ep5",
]

# candidate models for round 2 (single and/or ensemble)
SBERT_MODELS_ROUND2 = [
    "sbert_round2_epoch4_top35_mlm_finetuned_vibert_base_cased_all_corpora_ep5",
    "sbert_round2_epoch4_top35_mlm_finetuned_phobert_large_all_corpora_ep5",
    "sbert_round2_epoch4_top35_condenser_phobert_large_all_corpora_ep5",
    "sbert_round2_epoch4_top35_cocondenser_phobert_large_all_corpora_ep5",
]

def all_models_encode_corpus(models, corpus_name, eval_round):
    legal_dict_path = f"generated_data/{corpus_name}_legal_dict.json"
    # print(legal_dict_path)
    doc_data = json.load(open(legal_dict_path))
    # print(len(doc_data))
    list_emb_models = []
    for model in models:
        emb2_list = []
        for k, _ in tqdm(doc_data.items()):
            emb2 = model.encode(doc_data[k]["text"])
            emb2_list.append(emb2)
        emb2_arr = np.array(emb2_list)
        list_emb_models.append(emb2_arr)
    
    # save embedded data to file
    with open(f"generated_data/{corpus_name}_all_models_round{eval_round}_embeddings_{SBERT_TRAINED_DATA_VERSION}.pkl", "wb") as embedded_corpus_file:
        pickle.dump(list_emb_models, embedded_corpus_file)

    return list_emb_models

def load_all_models_emdbeddings(corpus_name, eval_round):
    print("Start loading all models embeddings")
    embedded_corpus_path = f"generated_data/{corpus_name}_all_models_round{eval_round}_embeddings_{SBERT_TRAINED_DATA_VERSION}.pkl"
    with open(embedded_corpus_path, "rb") as embedded_corpus_file:
        emb_legal_data = pickle.load(embedded_corpus_file)
    return emb_legal_data

def single_model_encode_corpus(model, model_name, corpus_name):
    legal_dict_path = f"generated_data/{corpus_name}_legal_dict.json"
    # print(legal_dict_path)
    doc_data = json.load(open(legal_dict_path))
    # print(len(doc_data))
    embeddings = []
    for k, _ in tqdm(doc_data.items()):
        embedded = model.encode(doc_data[k]["text"])
        embeddings.append(embedded)
    np_embeddings = np.array(embeddings)        
    
    # save embedded data to file
    with open(f"generated_data/{corpus_name}_embedding_{model_name}.pkl", "wb") as embedded_corpus_file:
        pickle.dump(np_embeddings, embedded_corpus_file)

    return np_embeddings

def load_single_model_emdbeddings(model_name, corpus_name):
    print(f"Start loading {model_name} embeddings")
    embedded_corpus_path = f"generated_data/{corpus_name}_embedding_{model_name}.pkl"
    with open(embedded_corpus_path, "rb") as embedded_corpus_file:
        emb_legal_data = pickle.load(embedded_corpus_file)
    return emb_legal_data

def all_models_encode_questions(models, question_data):
    print("Start encoding questions")
    question_embs = []
    for model in models:
        emb_quest_dict = {}
        for _, item in tqdm(enumerate(question_data)):
            qid = "question_id" if "question_id" in item.keys() else "id"
            question_id = item[qid]
            question = item["text"]
            emb_quest_dict[question_id] = model.encode(question)
        question_embs.append(emb_quest_dict)
    return question_embs

def single_model_encode_questions(model, question_data):
    print("Start encoding questions")
    question_embeddings = {}
    for _, item in tqdm(enumerate(question_data)):
        qid = "question_id" if "question_id" in item.keys() else "id"
        question_id = item[qid]
        question = item["text"]
        question_embeddings[question_id] = model.encode(question)
    return question_embeddings

def load_bm25(saved_model_path, bm25_path):
    model_path = os.path.join(saved_model_path, bm25_path)
    with open(model_path, "rb") as bm_file:
        bm25 = pickle.load(bm_file)
    return bm25

def load_models(saved_model_path, model_names):
    models = []
    for model_name in tqdm(model_names):
        model_path = os.path.join(saved_model_path, model_name)
        models.append(SentenceTransformer(model_path))
    return models

def ensemble_model_predict(bm25_model, 
                      sbert_models, 
                      flat_corpus_data, 
                      all_models_corpus_embeddings, 
                      predicting_questions,
                      scoring_weights,
                      sqrt_lexical,
                      range_score,
                      max_relevants
                      ):
    # encode question for query
    all_models_questions_embeddings = all_models_encode_questions(sbert_models, predicting_questions)

    pred_list = []

    print("Start calculating results")
    for _, item in tqdm(enumerate(predicting_questions)):
        qid = "question_id" if "question_id" in item.keys() else "id"
        question_id = item[qid]
        question = item["text"]
        
        tokenized_query = bm25_tokenizer(question)
        doc_scores = bm25_model.get_scores(tokenized_query)

        cos_sim = []
        for idx_2, _ in enumerate(models):
            emb1 = all_models_questions_embeddings[idx_2][question_id]
            emb2 = all_models_corpus_embeddings[idx_2]
            scores = util.cos_sim(emb1, emb2)
            cos_sim.append(scoring_weights[idx_2] * scores)
        cos_sim = torch.cat(cos_sim, dim=0)
        
        cos_sim = torch.sum(cos_sim, dim=0).squeeze(0).numpy()
        if sqrt_lexical:
            combined_scores = np.sqrt(doc_scores) * cos_sim
        else:            
            combined_scores = doc_scores * cos_sim
        max_score = np.max(combined_scores)
        
        map_ids = np.where(combined_scores >= (max_score - range_score))[0]
        top_scores = combined_scores[map_ids]
        
        print ("top_scores: ", top_scores)

        if top_scores.shape[0] > max_relevants:
            candidate_indices = np.argpartition(top_scores, -max_relevants)[-max_relevants:]
            map_ids = map_ids[candidate_indices]
            
        pred_dict = {}
        pred_dict["question_id"] = question_id
        pred_dict["relevant_articles"] = []
        
        dup_ans = []
        for _, idx_pred in enumerate(map_ids):
            pred = flat_corpus_data[idx_pred]
            law_id = pred[0]
            article_id = pred[1]
            
            if law_id + "_" + article_id not in dup_ans:
                dup_ans.append(law_id + "_" + article_id)
                pred_dict["relevant_articles"].append({"law_id": law_id, "article_id": article_id})
        pred_list.append(pred_dict)

    return pred_list

def ensemble_model_lexfirst_predict(
                      bm25_model, 
                      sbert_models, 
                      flat_corpus_data, 
                      all_models_corpus_embeddings, 
                      predicting_questions, 
                      scoring_weights,
                      sqrt_lexical,
                      lexical_top_k,
                      range_score,
                      max_relevants
                      ):
    """
    Ensemble model where lexical matching (lexfirst) is done first to filter lexical-matching documents before semantic matching will be done afterwards to rank all remaining documents and then get the final result.
    """
    
    # encode questions for query
    all_models_questions_embeddings = all_models_encode_questions(sbert_models, predicting_questions)

    pred_list = []

    print("Start calculating results")
    for _, item in tqdm(enumerate(predicting_questions)):
        qid = "question_id" if "question_id" in item.keys() else "id"
        question_id = item[qid]
        question = item["text"]
        
        tokenized_query = bm25_tokenizer(question)
        doc_scores = bm25_model.get_scores(tokenized_query)

        # get top_k lexical-matching documents
        map_ids = np.argpartition(doc_scores, -lexical_top_k)[-lexical_top_k:]
        top_lex_scores = doc_scores[map_ids]

        combined_scores = []
        for idx_2, _ in enumerate(models):
            emb1 = all_models_questions_embeddings[idx_2][question_id]
            emb2 = all_models_corpus_embeddings[idx_2][map_ids]
            scores = util.cos_sim(emb1, emb2)
            combined_scores.append(scoring_weights[idx_2] * scores)
        combined_scores = torch.cat(combined_scores, dim=0)
        combined_scores = torch.sum(combined_scores, dim=0).squeeze(0).numpy()        
        if sqrt_lexical:
            combined_scores = np.sqrt(top_lex_scores) * combined_scores
        else:            
            combined_scores = top_lex_scores * combined_scores
        max_score = np.max(combined_scores)
        
        top_score_ids = np.where(combined_scores >= (max_score - range_score))[0]
        top_scores = combined_scores[top_score_ids]
        map_ids = map_ids[top_score_ids]
        
        print ("top_scores: ", top_scores)

        if top_scores.shape[0] > max_relevants:
            candidate_indices = np.argpartition(top_scores, -max_relevants)[-max_relevants:]
            map_ids = map_ids[candidate_indices]
            
        pred_dict = {}
        pred_dict["question_id"] = question_id
        pred_dict["relevant_articles"] = []
        
        dup_ans = []
        for _, idx_pred in enumerate(map_ids):
            pred = flat_corpus_data[idx_pred]
            law_id = pred[0]
            article_id = pred[1]
            
            if law_id + "_" + article_id not in dup_ans:
                dup_ans.append(law_id + "_" + article_id)
                pred_dict["relevant_articles"].append({"law_id": law_id, "article_id": article_id})
        pred_list.append(pred_dict)

    return pred_list

def single_model_predict(sbert_model, 
                    flat_corpus_data, 
                    corpus_embeddings, 
                    predicting_questions, 
                    range_score,
                    max_relevants
                    ):
    # encode question for query
    single_model_questions_embeddings = single_model_encode_questions(sbert_model, predicting_questions)

    pred_list = []

    print("Start calculating results")
    for _, item in tqdm(enumerate(predicting_questions)):
        qid = "question_id" if "question_id" in item.keys() else "id"
        question_id = item[qid]

        emb1 = single_model_questions_embeddings[question_id]
        sbert_scores = util.cos_sim(emb1, corpus_embeddings).numpy().squeeze(0)        
        max_score = np.max(sbert_scores)
        
        map_ids = np.where(sbert_scores >= (max_score - range_score))[0]
        top_scores = sbert_scores[map_ids]

        print ("top_scores: ", top_scores)

        if top_scores.shape[0] > max_relevants:
            candidate_indices = np.argpartition(top_scores, -max_relevants)[-max_relevants:]
            map_ids = map_ids[candidate_indices]
            
        pred_dict = {}
        pred_dict["question_id"] = question_id
        pred_dict["relevant_articles"] = []
        
        dup_ans = []
        for _, idx_pred in enumerate(map_ids):
            pred = flat_corpus_data[idx_pred]
            law_id = pred[0]
            article_id = pred[1]
            
            if law_id + "_" + article_id not in dup_ans:
                dup_ans.append(law_id + "_" + article_id)
                pred_dict["relevant_articles"].append({"law_id": law_id, "article_id": article_id})
        pred_list.append(pred_dict)

    return pred_list

def evaluate_results(train_questions, predictions):
    print("Start evaluating results")
    total_f2 = 0
    total_precision = 0
    total_recall = 0
    k = len(train_questions)
    for _, item in tqdm(enumerate(train_questions)):
        qid = "question_id" if "question_id" in item.keys() else "id"
        question_id = item[qid]
        relevant_articles = item["relevant_articles"]
        actual_positive = len(relevant_articles)

        pred_articles = list(filter(lambda x: x["question_id"] == question_id, predictions))[0]["relevant_articles"]
        
        true_positive = 0
        false_positive = 0
        for _, pred_art in enumerate(pred_articles):
            law_id = pred_art["law_id"]
            article_id = pred_art["article_id"]
            
            for article in relevant_articles:
                if law_id == article["law_id"] and article_id == article["article_id"]:
                    true_positive += 1
                else:
                    false_positive += 1
        
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", default="alqac23_data", type=str, help="path to raw corpus")
    parser.add_argument("--model_dir", default="saved_model", type=str, help="path to saved models")
    parser.add_argument("--eval_mode", default="ensemble", type=str, choices=["ensemble", "single"], help="evaluation mode")
    parser.add_argument("--eval_round", default=2, type=int, help="round at which models are evaluated in `ensemble` mode")
    parser.add_argument("--scoring_weights", default="0.2,0.1,0.3,0.4", type=str, help="4-element-array scoring weights for 4 combined SBert models (viBert, phobert-large, condenser, co-condenser)")
    parser.add_argument("--lexical_nodiff", action="store_true", help="lexical and semantic scoring together in `ensemble` mode. If this is not set, lexical-matching first then semantic-matching.")
    parser.add_argument("--lexical_top_k", default=100, type=int, help="number of lexical-matching documents extracted with `lexical_nodiff` param unset")
    parser.add_argument("--sqrt_lexical", action="store_true", help="apply `sqrt` for lexical score")
    parser.add_argument("--eval_model", default="", type=str, help="sbert model name to evaluate in `single` mode")
    parser.add_argument("--corpus_name", default="alqac23", type=str, choices=["alqac23", "alqac22", "zalo"], help="corpus to evaluate")
    parser.add_argument("--range_score", default=2.6, type=float, help="range of cosin score for multiple-answer")
    parser.add_argument("--max_relevants", default=1, type=int, help="max relevant articles should be produced")
    parser.add_argument("--encode_corpus", action="store_true", help="encode corpus if not pre-computed")
    parser.add_argument("--eval_on", default="train", type=str, choices=["train", "test"], help="evaluate on train or test data")
    args = parser.parse_args()

    # parse scoring weights
    scoring_weights = args.scoring_weights.split(',')
    for idx, weight in enumerate(scoring_weights):
        scoring_weights[idx] = float(weight)

    # load questions
    predicting_items = []
    if args.eval_on == "train":
        alqac23_corpus_path_train = os.path.join(args.raw_data_dir, "train.json")
        alqac22_corpus_path_train = os.path.join(args.raw_data_dir, "additional_data/ALQAC_2022_training_data/question.json")
        zalo_corpus_path_train = os.path.join(args.raw_data_dir, "additional_data/zalo/zalo_question.json")
        
        train_paths = {
            "alqac23": alqac23_corpus_path_train,
            "alqac22": alqac22_corpus_path_train,
            "zalo": zalo_corpus_path_train
        }  
        
        data_path = train_paths[args.corpus_name]
        predicting_items = json.load(open(data_path))
    
    elif args.eval_on == "test":
        alqac23_corpus_path_test = os.path.join(args.raw_data_dir, "public_test.json")
        predicting_items = json.load(open(alqac23_corpus_path_test))
    
    print("Number of questions: ", len(predicting_items))

    # load flat corpus to search
    print("Load flat corpus data")
    with open(f"generated_data/{args.corpus_name}_flat_corpus.pkl", "rb") as flat_corpus_file:
        flat_corpus_data = pickle.load(flat_corpus_file)
    
    if args.eval_mode == "ensemble":
        if args.eval_round == 1:
            model_paths = SBERT_MODELS_ROUND1
        elif args.eval_round == 2:
            model_paths = SBERT_MODELS_ROUND2
        
        print("Start loading models")
        # load bm25 model
        bm25_model = BM25_MODEL.replace("<<<corpus_name>>>", args.corpus_name)
        bm25 = load_bm25(args.model_dir, bm25_model)
        models = load_models(args.model_dir, model_paths)
        print("Number of sbert models: ", len(models))

        # encode or load pre-encoded embedding of the corpus
        if args.encode_corpus:
            all_models_corpus_embeddings = all_models_encode_corpus(models, args.corpus_name, args.eval_round)
        else:
            all_models_corpus_embeddings = load_all_models_emdbeddings(args.corpus_name, args.eval_round)
        
        if args.lexical_nodiff:
            predictions = ensemble_model_predict(bm25, models, flat_corpus_data, all_models_corpus_embeddings, predicting_items, scoring_weights, args.sqrt_lexical, args.range_score, args.max_relevants)
        else:
            predictions = ensemble_model_lexfirst_predict(bm25, models, flat_corpus_data, all_models_corpus_embeddings, predicting_items, scoring_weights, args.sqrt_lexical, args.lexical_top_k, args.range_score, args.max_relevants)
            
        if args.eval_on == "test":            
            if args.lexical_nodiff:
                if args.sqrt_lexical:
                    submission_name = f"{args.corpus_name}_ensemble_model_round{args.eval_round}_sqrtlex_submission.json"
                else:                    
                    submission_name = f"{args.corpus_name}_ensemble_model_round{args.eval_round}_submission.json"
            else:
                if args.sqrt_lexical:                
                    submission_name = f"{args.corpus_name}_ensemble_model_lexfirst{args.lexical_top_k}_round{args.eval_round}_sqrtlex_submission.json"
                else:
                    submission_name = f"{args.corpus_name}_ensemble_model_lexfirst{args.lexical_top_k}_round{args.eval_round}_submission.json"

            with open(f'results/{submission_name}', 'w', encoding='utf-8') as outfile:
                json_object = json.dumps(predictions, indent=4, ensure_ascii=False)
                outfile.write(json_object)

    elif args.eval_mode == "single":
        model_name = args.eval_model
        assert model_name != "" , "eval_model param in single evaluation mode must be not empty"

        print(f"Start loading {model_name} model")
        model = load_models(args.model_dir, [model_name])[0]

        # encode or load pre-encoded embedding of the corpus
        if args.encode_corpus:
            corpus_embeddings = single_model_encode_corpus(model, args.eval_model, args.corpus_name)
        else:
            corpus_embeddings = load_single_model_emdbeddings(args.eval_model, args.corpus_name)
        
        predictions = single_model_predict(model, flat_corpus_data, corpus_embeddings, predicting_items, args.range_score, args.max_relevants)

        if args.eval_on == "test":
            with open(f'results/{args.corpus_name}_{model_name}_submission.json', 'w', encoding='utf-8') as outfile:
                json_object = json.dumps(predictions, indent=4, ensure_ascii=False)
                outfile.write(json_object)
    
    if args.eval_on == "train":
        evaluate_results(predicting_items, predictions)