import argparse
import json
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")
import re

def prepare_extractive_questions(questions: list):
    input_dicts = []
    extractive_questions = []

    # load data from alqac23 train
    with open("generated_data/alqac23_legal_dict.json") as legal_dict_file:
        alqac23_law_dict = json.load(legal_dict_file)

    for q in questions:
        question_id = q["question_id"]
        if "TL-" not in question_id:
            continue

        question = q["text"]
        relevant_articles = q["relevant_articles"]
        context = ""
        for art in relevant_articles:
            law_id = art["law_id"]
            article_id = art["article_id"]
            dict_key = law_id + "_" + article_id
            
            # ctx = remove_newlines(alqac23_law_dict[dict_key]["text"])
            ctx = alqac23_law_dict[dict_key]["text"]
            context = ' '.join([context, ctx])
        
        extractive_questions.append(q)
        input_dicts.append({"question": question, "context": context})
    
    return input_dicts, extractive_questions

def clean_answer(answer: str):
    answer = answer.strip(",.:;?\n")
    answer = re.sub(r"\n+", " ", answer)
    return answer

def predict_extractive_questions(pipeline, questions: list):
    preds = []

    input_dicts, extractive_questions = prepare_extractive_questions(questions)

    # predict answers by pipeline
    pred_answers = pipeline(input_dicts)

    assert len(extractive_questions) == len(pred_answers), "Number of extractive questions must be equal to number of predicted answers from pipeline."

    preds = []
    for idx, q in enumerate(extractive_questions):     
        # override answer if exists
        q["answer"] = clean_answer(pred_answers[idx]["answer"])
        preds.append(q)
    
    return preds

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="saved_model/finetuned_ep20_vi_mrc_large/checkpoint-800", type=str, help="path to extractive QA model")
    parser.add_argument("--eval_on", default="train", type=str, choices=["train", "public_test", "private_test"], help="evaluate on train, public test or private test data")
    args = parser.parse_args()

    print ("Loading the model")        
    # model_checkpoint = "nguyenvulebinh/vi-mrc-large"
    model_checkpoint = args.model_path
    extractive_pipe = pipeline('question-answering', model=model_checkpoint,
                       tokenizer=model_checkpoint)
    print ("Done")   

    data_paths = [
        "alqac23_data/train.json",                                                        # train      
        "results/alqac23_ensemble_2035_lexfirst100_round2_public_test.json.json",         # public test              
        "alqac23_data/private_test_GOLD_TASK_1.json"                                      # private test
    ]
    
    if args.eval_on == "train":
        data_path = data_paths[0]
    elif args.eval_on == "public_test":
        data_path = data_paths[1]
    elif args.eval_on == "private_test":
        data_path = data_paths[2]

    # load questions
    questions = json.load(open(data_path))

    preds = predict_extractive_questions(extractive_pipe, questions)

    print ("preds: ", preds) 

    with open(f'results/extractive_qa_result-ckpt800.json', 'w', encoding='utf-8') as outfile:
        json_object = json.dumps(preds, indent=4, ensure_ascii=False)
        outfile.write(json_object)
