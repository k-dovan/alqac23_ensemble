
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import argparse
import json
from predict_boolean_qa import predict_bool_questions
from predict_choices_qa import predict_choices_questions


def evaluate_results(questions, predictions):
    print("Start evaluating results")
    # print ("questions: ", questions)
    # print ("predictions: ", predictions)
    correct = 0
    count = 0
    for item in tqdm(questions):
        question_id = item["question_id"]
        question_type = item["question_type"]

        if question_type == "Tự luận":
            continue
        count += 1

        answer = item["answer"]
        if question_type == "Đúng/Sai":
            answer = "True" if answer == "Đúng" else "False"
        
        if answer == predictions[question_id]:
            correct += 1

    print(f"Accuracy: {correct/count:0.3f}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="saved_model/boolq_alqac23_ep50_google_ep3_finetuned_vi_mrc_large/checkpoint-200", type=str, help="path to boolean QA model")
    parser.add_argument("--eval_on", default="train", type=str, choices=["train", "public_test", "private_test"], help="evaluate on train, public test or private test data")
    parser.add_argument("--submission_sig", default="cpt200", type=str, help="submission signature")
    args = parser.parse_args()

    print ("Loading the model")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, device_map="auto", load_in_8bit=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, device_map="auto", load_in_8bit=True)
    print ("Done")

    data_paths = [
        "alqac23_data/train.json",                                                              # train      
        "results/alqac23_ensemble_2035_lexfirst100_ro2_public_test_submission.json",            # public test              
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

    bool_preds = predict_bool_questions(model, tokenizer, questions, args.eval_on)   
    choice_preds = predict_choices_questions(model, tokenizer, questions, args.eval_on) 

    preds =  {**bool_preds, **choice_preds}

    if args.eval_on == "train":
        evaluate_results(questions, preds)
    elif args.eval_on == "public_test":
        submission_name = f"task2_pubic_test_{args.submission_sig}_submission.json"
    elif args.eval_on == "private_test":        
        submission_name = f"task2_private_test_{args.submission_sig}_submission.json"
    
    if args.eval_on != "train":
        with open(f'results/{submission_name}', 'w', encoding='utf-8') as outfile:
            json_object = json.dumps(preds, indent=4, ensure_ascii=False)
            outfile.write(json_object)