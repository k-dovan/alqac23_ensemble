
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from tqdm import tqdm
import argparse
import json
from predict_boolean_qa import predict_bool_questions
from predict_choices_qa import predict_choices_questions
from predict_extractive_qa import predict_extractive_questions

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
    parser.add_argument("--extr_model_path", default="saved_model/finetuned_ep20_vi_mrc_large/checkpoint-800", type=str, help="path to extractive QA model")
    parser.add_argument("--boolq_model_path", default="saved_model/boolq_alqac23_ep50_google_ep3_finetuned_ep20_vi_mrc_large/checkpoint-200", type=str, help="path to boolean/choice QA model")
    parser.add_argument("--eval_on", default="train", type=str, choices=["train", "public_test", "private_test"], help="evaluate on train, public test or private test data")
    parser.add_argument("--extqa_model_sig", default="extqa-ep20-ckpt-800", type=str, help="submission signature")
    parser.add_argument("--boolqa_model_sig", default="boolqa-eps-50-3-20-ckpt-200", type=str, help="submission signature")
    args = parser.parse_args()

    print ("Loading the models")
    # boolean/multiple-choice QA model
    tokenizer = AutoTokenizer.from_pretrained(args.boolq_model_path, device_map="auto", load_in_8bit=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.boolq_model_path, device_map="auto", load_in_8bit=True)
    # extractive QA model
    extractive_pipe = pipeline('question-answering', model=args.extr_model_path,
                       tokenizer=args.extr_model_path)
    print ("Done")

    data_paths = [
        "alqac23_data/train.json",                                                      # train      
        "results/alqac23_ensemble_2035_lexfirst100_round2_public_test.json",            # public test              
        "alqac23_data/private_test_GOLD_TASK_1.json"                                    # private test
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
    extractive_preds = predict_extractive_questions(extractive_pipe, questions)
    preds =  {**bool_preds, **choice_preds}  

    if args.eval_on == "train":
        evaluate_results(questions, preds)
    elif args.eval_on == "public_test":
        boolchoice_submission = f"task2_pubic_test_{args.boolqa_model_sig}_submission.json"
        extractive_submission = f"task2_pubic_test_{args.extqa_model_sig}_submission.json"
    elif args.eval_on == "private_test":        
        boolchoice_submission = f"task2_private_test_{args.boolqa_model_sig}_submission.json"
        extractive_submission = f"task2_private_test_{args.extqa_model_sig}_submission.json"
    
    if args.eval_on != "train":
        preds_submission = [{"question_id": k, "answer": v} for k,v in preds.items()] 
        with open(f'results/{boolchoice_submission}', 'w', encoding='utf-8') as outfile:
            json_object = json.dumps(preds_submission, indent=4, ensure_ascii=False)
            outfile.write(json_object)
        with open(f'results/{extractive_submission}', 'w', encoding='utf-8') as outfile:
            json_object = json.dumps(extractive_preds, indent=4, ensure_ascii=False)
            outfile.write(json_object)