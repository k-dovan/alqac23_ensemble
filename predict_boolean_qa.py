
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import argparse
import torch
import json
from utils import segment_long_text, remove_newlines

def prepare_bool_questions(questions: list):
    question_ids = []
    input_tuples = []
    bool_questions = []

    max_length = 2000
    sliding_thresh = 1000
    # load data from alqac23 train
    with open("generated_data/alqac23_legal_dict.json") as legal_dict_file:
        alqac23_law_dict = json.load(legal_dict_file)

    for q in tqdm(questions):
        q_type = q["question_type"]
        if q_type != "Đúng/Sai":
            continue

        bool_questions.append(q)
        question_id = q["question_id"]
        question = remove_newlines(q["text"])
        # question = q["text"]

        relevant_articles = q["relevant_articles"]
        for art in relevant_articles:
            law_id = art["law_id"]
            article_id = art["article_id"]
            dict_key = law_id + "_" + article_id
            
            ctx = remove_newlines(alqac23_law_dict[dict_key]["text"])
            # ctx = alqac23_law_dict[dict_key]["text"]
            if len(ctx)  :
                pass
            if len(ctx) <= max_length:
                question_ids.append(question_id)
                input_tuples.append((question, ctx))
            else:
                paragraphs = segment_long_text(ctx, max_length, sliding_thresh)
                for p in paragraphs:
                    question_ids.append(question_id)
                    input_tuples.append((question, p))
    
    return question_ids, input_tuples, bool_questions

def evaluate_results(questions, predictions):
    print("Start evaluating results")
    print ("questions: ", questions)
    print ("predictions: ", predictions)
    correct = 0
    for item in tqdm(questions):
        question_id = item["question_id"]
        answer = item["answer"]
        answer_en = "True" if answer == "Đúng" else "False"
        
        if answer_en == predictions[question_id]:
            correct += 1

    print(f"Accuracy: {correct/len(questions):0.3f}")

def predict_bool_questions(model, tokenizer, questions: list, eval_on: str):
    
    question_ids, input_tuples, bool_questions = prepare_bool_questions(questions)

    batch_inputs = tokenizer(input_tuples, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        logits = model(**batch_inputs).logits
    print ("logits: ", logits)

    class_ids = logits.argmax(dim=1).tolist()
    print (class_ids)

    preds = {}
    for idx, qid in enumerate(question_ids):
        if qid not in preds:
            preds[qid] = model.config.id2label[class_ids[idx]]
        else:
            if model.config.id2label[class_ids[idx]] == "True":
                preds[qid] = model.config.id2label[class_ids[idx]]    
    
    assert len(bool_questions) == len(preds), "Number of bool questions must be equal to number of predictions."

    if eval_on == "train":
        evaluate_results(bool_questions, preds)
    
    preds = [{"question_id": k, "answer": v} for k,v in preds.items()]
    
    return preds

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="saved_model/boolq_alqac23_ep3_google_ep3_finetuned_vi_mrc_large", type=str, help="path to boolean QA model")
    parser.add_argument("--eval_on", default="train", type=str, choices=["train", "public_test", "private_test"], help="evaluate on train, public test or private test data")
    args = parser.parse_args()

    print ("Loading the model")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, device_map="auto", load_in_8bit=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, device_map="auto", load_in_8bit=True)
    print ("Done")

    data_paths = [
        "alqac23_data/train.json",                                                              # train      
        "results/alqac23_ensemble_2035_lexfirst100_ro2_private_test_submission_2.json",         # public test              
        "alqac23_data/private_test_GOLD_TASK_1.json"                                            # private test
    ]
    
    if args.eval_on == "train":
        data_path = data_paths[0]
    elif args.eval_on == "public_test":
        data_path = data_paths[1]
    elif args.eval_on == "private_test":
        data_path = data_paths[2]

    # load questions
    questions = json.load(open(data_path))

    preds = predict_bool_questions(model, tokenizer, questions, args.eval_on)    