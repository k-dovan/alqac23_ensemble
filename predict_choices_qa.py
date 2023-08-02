
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import argparse
import torch
import json
from utils import segment_long_text, remove_newlines, clean_multichoice_question, add_choice_metadata

def prepare_choices_questions(questions: list):
    question_choice_ids = []
    input_tuples = []
    choices_questions = []

    max_length = 2000
    sliding_thresh = 1000
    # load data from alqac23 train
    with open("generated_data/alqac23_legal_dict.json") as legal_dict_file:
        alqac23_law_dict = json.load(legal_dict_file)

    for q in tqdm(questions):
        question_id = q["question_id"]
        if "TN-" not in question_id:
            continue
        
        question = clean_multichoice_question(remove_newlines(q["text"]))
        # question = q["text"]

        # update metadata for the question
        q_meta = add_choice_metadata(q)
        choices_questions.append(q_meta)

        # process standalone choice
        for c_label in q_meta["metadata"]["standalones"]:
            c_text = q_meta["choices"][c_label]
            query = ' '.join([question, c_text])

            relevant_articles = q["relevant_articles"]
            for art in relevant_articles:
                law_id = art["law_id"]
                article_id = art["article_id"]
                dict_key = law_id + "_" + article_id
                
                ctx = remove_newlines(alqac23_law_dict[dict_key]["text"])
                # ctx = alqac23_law_dict[dict_key]["text"]

                if len(ctx) <= max_length:
                    question_choice_ids.append((question_id, c_label))
                    input_tuples.append((query, ctx))
                else:
                    paragraphs = segment_long_text(ctx, max_length, sliding_thresh)
                    for p in paragraphs:
                        question_choice_ids.append((question_id, c_label))
                        input_tuples.append((query, p))
    
    return question_choice_ids, input_tuples, choices_questions

def evaluate_results(questions, predictions):
    print("Start evaluating results")
    # print ("questions: ", questions)
    # print ("predictions: ", predictions)
    correct = 0
    for item in tqdm(questions):
        question_id = item["question_id"]
        answer = item["answer"]
        
        if answer == predictions[question_id]:
            correct += 1

    print(f"Accuracy: {correct/len(questions):0.3f}")

def scoring(logit):
    w1, w2, w3 = 0.1, 0.8, 0.1
    return  -w1*logit[0] + w2*(logit[1] - logit[0]) + w3*logit[1]

def pick_max_logit_score_index(logits):
    max_idx = 0
    max_score = scoring(logits[0])
    for idx, l in enumerate(logits[1:]):
        if scoring(l) > max_score:
            max_score = scoring(l)
            max_idx = idx + 1
    return max_idx

def calculate_answers(logits_dict, choices_questions, false_thresh: float = -0.8, true_thresh: float = 0.8):

    preds = {}
    for q in choices_questions:
        question_id = q["question_id"]

        standalones = q["metadata"]["standalones"]
        if len(standalones) == 0:
            continue

        standalone_choice_logits = []
        standalone_choice_scores = []
        for label in standalones:
            key = question_id + label
            standalone_choice_logits.append(logits_dict[key])
            standalone_choice_scores.append(scoring(logits_dict[key]))

        total_choices = len(q["choices"].keys())
        
        # divide into multi-choice cases
        if len(standalones) == total_choices:
            # 1. all choices are standalone -> pick the best scored choice
            preds[question_id] = standalones[pick_max_logit_score_index(standalone_choice_logits)]
        elif len(q["metadata"]["trues"].keys()) > 0:
            if len(q["metadata"]["falses"].keys()) > 0:
                # 2. exist both refering true/false choices
                if min(standalone_choice_scores) > true_thresh:
                    if max(standalone_choice_scores) < false_thresh:
                        true_gap = min(standalone_choice_scores) - true_thresh
                        false_gap = false_thresh - max(standalone_choice_scores)
                        if true_gap > false_gap:
                            preds[question_id] = list(q["metadata"]["trues"].keys())[0]
                        else:
                            preds[question_id] = list(q["metadata"]["falses"].keys())[0]
                    else:
                        preds[question_id] = list(q["metadata"]["trues"].keys())[0]
                else:
                    if max(standalone_choice_scores) < false_thresh:
                        preds[question_id] = list(q["metadata"]["falses"].keys())[0]                            
                    else:
                        preds[question_id] = standalones[pick_max_logit_score_index(standalone_choice_logits)]
            else:
                # 3. exist only refering all-true choices
                if min(standalone_choice_scores) > true_thresh:
                    preds[question_id] = list(q["metadata"]["trues"].keys())[0]
                else:
                    preds[question_id] = standalones[pick_max_logit_score_index(standalone_choice_logits)]
        elif len(q["metadata"]["falses"].keys()) > 0:
            # 4. exist only refering all-false choices
            if max(standalone_choice_scores) < false_thresh:
                preds[question_id] = list(q["metadata"]["falses"].keys())[0]
            else:
                preds[question_id] = standalones[pick_max_logit_score_index(standalone_choice_logits)]
    return preds

def build_question_choice_logits(question_choice_ids, logits):
    logits_dict = {}
    for idx, qc in enumerate(question_choice_ids):
        key = qc[0] + qc[1]
        if key not in logits_dict:
            logits_dict[key] = logits[idx]
        else:
            # pick the best logit
            old_score = scoring(logits_dict[key])
            new_score = scoring(logits[idx])
            if new_score > old_score:
                logits_dict[key] = logits[idx]
    return logits_dict

def predict_choices_questions(model, tokenizer, questions: list, eval_on: str):
    
    question_choice_ids, input_tuples, choices_questions = prepare_choices_questions(questions)

    batch_inputs = tokenizer(input_tuples, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        logits = model(**batch_inputs).logits.tolist()
    # print ("logits: ", logits)

    # convert question_choice_ids to dict of unique quesiont_choice with its logits
    logits_dict = build_question_choice_logits(question_choice_ids, logits)

    # calculate answer based on logits and question's calculated metadata
    preds = calculate_answers(logits_dict, choices_questions)
    
    assert len(choices_questions) == len(preds), "Number of bool questions must be equal to number of predictions."

    if eval_on == "train":
        evaluate_results(choices_questions, preds)
    
    return preds

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="saved_model/boolq_alqac23_ep50_google_ep3_finetuned_ep20_vi_mrc_large/checkpoint-200", type=str, help="path to boolean QA model")
    parser.add_argument("--eval_on", default="train", type=str, choices=["train", "public_test", "private_test"], help="evaluate on train, public test or private test data")
    args = parser.parse_args()

    print ("Loading the model")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, device_map="auto", load_in_8bit=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, device_map="auto", load_in_8bit=True)
    print ("Done")

    data_paths = [
        "alqac23_data/train.json",                                                   # train      
        "results/alqac23_ensemble_2035_lexfirst100_round2_public_test.json",         # public test              
        "alqac23_data/private_test_GOLD_TASK_1.json"                                 # private test
    ]
    
    if args.eval_on == "train":
        data_path = data_paths[0]
    elif args.eval_on == "public_test":
        data_path = data_paths[1]
    elif args.eval_on == "private_test":
        data_path = data_paths[2]

    # load questions
    questions = json.load(open(data_path))

    preds = predict_choices_questions(model, tokenizer, questions, args.eval_on)    