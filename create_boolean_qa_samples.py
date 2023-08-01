import json
import re
from tqdm import tqdm
import random
import numpy as np
import os
from utils import clean_multichoice_question, add_choice_metadata, remove_newlines

if __name__ == '__main__':
    
    # =====================================================================================================
    # train/val data from Google-BoolQ-Dataset for training model round 1 
    # =====================================================================================================
    # load translated samples
    translated_train_data = json.load(open("generated_data/translated/translated_train.json", "r", encoding="utf-8"))
    translated_val_data = json.load(open("generated_data/translated/translated_dev.json", "r", encoding="utf-8"))
    
    samples = []
    train_samples, val_samples = [], []
    count = 0
    for sample in translated_train_data:
        count += 1
        question = sample["translated_question"]
        context = sample["translated_passage"]
        assert isinstance(sample["answer"], bool)
        answer = "True" if sample["answer"] else "False"
        
        qa_sample = {"question": question, "context": context, "label": answer}
        train_samples.append(qa_sample)
        
    for sample in translated_val_data:
        count += 1
        question = sample["translated_question"]
        context = sample["translated_passage"]
        assert isinstance(sample["answer"], bool)
        answer = "True" if sample["answer"] else "False"
        
        qa_sample = {"question": question, "context": context, "label": answer}
        val_samples.append(qa_sample)
    
    save_dir = "generated_data/qa_boolean_question_data"
    os.makedirs(save_dir, exist_ok=True)  

    with open(f"{save_dir}/google_boolean_qa_training_samples.json", "w", encoding="utf-8") as outfile:
        json_object = json.dumps(train_samples, indent=4, ensure_ascii=False)
        outfile.write(json_object)
    with open(f"{save_dir}/google_boolean_qa_validation_samples.json", "w", encoding="utf-8") as outfile:
        json_object = json.dumps(val_samples, indent=4, ensure_ascii=False)
        outfile.write(json_object)
    
    print (f"> Google-Boolean-Dataset:")
    print (f">> Number of boolean samples: {count}")
    print (f">> Number of valid samples: {len(train_samples) + len(val_samples)}")
    print (f">> Number of training samples: {len(train_samples)}")
    print (f">> Number of validation samples: {len(val_samples)}")


    # =====================================================================================================
    # train/val data from ALQAC23 corpus for training model round 2
    # =====================================================================================================

    # load data from alqac23 train
    with open("generated_data/alqac23_legal_dict.json") as legal_dict_file:
        alqac23_law_dict = json.load(legal_dict_file)

    # questions from alqac23 train
    alqac23_train_items = json.load(open("alqac23_data/train.json"))

    samples = []
    count_boolq, count_multchoice = 0, 0
    max_length = 2000
    for item in tqdm(alqac23_train_items):
        q_type = item["question_type"]
        question_id = item["question_id"]
        question = remove_newlines(item["text"])
        question = item["text"]
        answer = item["answer"]

        relevant_articles = item["relevant_articles"]
        context = ""
        for art in relevant_articles:
            law_id = art["law_id"]
            article_id = art["article_id"]
            dict_key = law_id + "_" + article_id
            
            cxt = remove_newlines(alqac23_law_dict[dict_key]["text"])
            # cxt = alqac23_law_dict[dict_key]["text"]
            context = ' '.join([context, cxt]).strip()
        
        if len(context) > max_length:
            print (f"Too long artile refered by {question_id}.")
            continue

        if q_type == "Đúng/Sai":                    
            count_boolq += 1

            if answer == "Đúng":
                answer_en = "True"
            elif answer == "Sai":
                answer_en = "False"
            else:
                print(f"> Incorrect label at question {question_id}, answer: {answer}, expected: [Đúng, Sai]")
                continue

            qa_sample = {"question": question, "context": context, "label": answer_en}
            samples.append(qa_sample)

        elif q_type == "Trắc nghiệm":
            count_multchoice += 1
            assert answer in ['A','B','C','D'], f"Multi-choices' questions answer must be A, B, C, or D. Actual: {answer}"
            
            q_text = clean_multichoice_question(remove_newlines(question))
            q_meta = add_choice_metadata(item)
            if answer in q_meta["metadata"]["standalones"]:
                c_text = q_meta["choices"][answer]
                query = ' '.join([q_text, c_text])
                qa_sample = {"question": query, "context": context, "label": "True"}
                samples.append(qa_sample)

                # add all other standalone choices as negative samples
                for c_label in q_meta["metadata"]["standalones"]:
                    if c_label == answer:
                        continue
                    c_text = q_meta["choices"][c_label]
                    query = ' '.join([q_text, c_text])
                    qa_sample = {"question": query, "context": context, "label": "False"}
                    samples.append(qa_sample)
            elif answer == list(q_meta["metadata"]["trues"].keys())[0]:
                refered_choices = q_meta["metadata"]["trues"][answer]
                # add all refered choices as positive samples
                for c_label in refered_choices:
                    c_text = q_meta["choices"][c_label]
                    query = ' '.join([q_text, c_text])
                    qa_sample = {"question": query, "context": context, "label": "True"}
                    samples.append(qa_sample)                
            elif answer == list(q_meta["metadata"]["falses"].keys())[0]:
                refered_choices = q_meta["metadata"]["falses"][answer]
                # add all refered choices as negative samples
                for c_label in refered_choices:
                    c_text = q_meta["choices"][c_label]
                    query = ' '.join([q_text, c_text])
                    qa_sample = {"question": query, "context": context, "label": "False"}
                    samples.append(qa_sample)            

    # sampling for train and validation sets
    np_samples = np.array(samples)
    num_val_samples = round(0.10 * len(samples))
    all_indices = set(range(len(samples)))
    val_indices = set(random.choices(list(all_indices), k=num_val_samples))
    train_indices = all_indices.difference(val_indices)
    train_samples = list(np_samples[list(train_indices)])  
    val_samples = list(np_samples[list(val_indices)])
    
    with open(f"{save_dir}/alqac23_boolean_qa_training_samples.json", "w", encoding="utf-8") as outfile:
        json_object = json.dumps(train_samples, indent=4, ensure_ascii=False)
        outfile.write(json_object)
    with open(f"{save_dir}/alqac23_boolean_qa_validation_samples.json", "w", encoding="utf-8") as outfile:
        json_object = json.dumps(val_samples, indent=4, ensure_ascii=False)
        outfile.write(json_object)

    print (f"> ALQAC23's training corpus:")
    print (f">> Number of boolean samples: {count_boolq}")
    print (f">> Number of mulit-choice samples: {count_multchoice}")
    print (f">> Number of valid samples: {len(samples)}")
    print (f">> Number of training samples: {len(train_samples)}")
    print (f">> Number of validation samples: {len(val_samples)}")
