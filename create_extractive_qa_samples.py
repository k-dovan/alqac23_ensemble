import json
import re
from tqdm import tqdm
import random
import numpy as np
import os

from utils import remove_newlines

if __name__ == '__main__':

    # load data from alqac23 train
    with open("generated_data/alqac23_legal_dict.json") as legal_dict_file:
        alqac23_law_dict = json.load(legal_dict_file)
    with open("generated_data/alqac22_legal_dict.json") as legal_dict_file:
        alqac22_law_dict = json.load(legal_dict_file)

    # questions from alqac23 train
    alqac23_train_items = json.load(open("alqac23_data/train.json"))
    alqac22_train_items = json.load(open("alqac23_data/additional_data/ALQAC_2022_training_data/question.json"))

    samples = []
    count = 0
    for item in tqdm(alqac23_train_items):
        q_type = item["question_type"]
        if q_type != "Tự luận":
            continue        
        count += 1
        qid = "question_id" if "question_id" in item.keys() else "id"
        question_id = item[qid]
        # question = remove_newlines(item["text"])
        question = item["text"]
        answer = item["answer"]

        relevant_articles = item["relevant_articles"]
        context = ""
        for art in relevant_articles:
            law_id = art["law_id"]
            article_id = art["article_id"]
            dict_key = law_id + "_" + article_id
            
            # cxt = remove_newlines(alqac23_law_dict[dict_key]["text"])
            cxt = alqac23_law_dict[dict_key]["text"]
            context = ' '.join([context, cxt]).strip()
        
        # find answer start index
        start_idx = context.find(answer)
        if start_idx < 0:
            print (f">>> alqac23 -> answer not found:\n >>> answer: {answer} \n >>> context: {context}\n\n")
            continue
        
        qa_sample = {"id": question_id, "question": question, "context": context, "answers": {'answer_start': [start_idx], 'text': [answer]}}
        samples.append(qa_sample)
    
    for item in tqdm(alqac22_train_items):
        count += 1
        qid = "question_id" if "question_id" in item.keys() else "id"
        question_id = item[qid]
        # question = remove_newlines(item["text"])
        question = item["text"]
        answer = item["answer"]

        relevant_articles = item["relevant_articles"]
        context = ""
        for art in relevant_articles:
            law_id = art["law_id"]
            article_id = art["article_id"]
            dict_key = law_id + "_" + article_id
            
            # cxt = remove_newlines(alqac22_law_dict[dict_key]["text"])
            cxt = alqac22_law_dict[dict_key]["text"]
            context = ' '.join([context, cxt]).strip()
        
        # find answer start index
        start_idx = context.find(answer)
        if start_idx < 0:
            print (f">>> alqac22 -> answer not found:\n >>> answer: {answer} \n >>> context: {context}\n\n")
            continue
        
        qa_sample = {"id": question_id, "question": question, "context": context, "answers": {'answer_start': [start_idx], 'text': [answer]}}
        samples.append(qa_sample) 

    # sampling for train and validation sets
    np_samples = np.array(samples)
    num_val_samples = round(0.10 * len(samples))
    all_indices = set(range(len(samples)))
    val_indices = set(random.choices(list(all_indices), k=num_val_samples))
    train_indices = all_indices.difference(val_indices)
    val_samples = list(np_samples[list(val_indices)])
    train_samples = list(np_samples[list(train_indices)])
    
    save_dir = "generated_data/qa_extractive_question_data/"
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/extractive_qa_training_samples.json", "w", encoding="utf-8") as outfile:
        json_object = json.dumps({"data": train_samples}, indent=4, ensure_ascii=False)
        outfile.write(json_object)
    with open(f"{save_dir}/extractive_qa_validation_samples.json", "w", encoding="utf-8") as outfile:
        json_object = json.dumps({"data": val_samples}, indent=4, ensure_ascii=False)
        outfile.write(json_object)
    
    print (f"Number of span-based answers: {count}")
    print (f"Number of extracted samples: {len(samples)}")
    print (f"Number of training samples: {len(train_samples)}")
    print (f"Number of validation samples: {len(val_samples)}")
    print ("Extractive QA samples created!")
