import json
import argparse
from predict import evaluate_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_file_name", 
                        default="private_test_GOLD_TASK_1.json", type=str, help="gold label file name")
    parser.add_argument("--pred_file_names", 
                        default="alqac23_ensemble_2035_lexfirst100_ro1_private_test_submission_1.json,alqac23_ensemble_2035_lexfirst100_ro2_private_test_submission_2.json,alqac23_ensemble_2035_lexfirst200_ro1_private_test_submission_3.json", 
                        type=str, help="prediction files' names")
    args = parser.parse_args()

    gold_path = f"alqac23_data/{args.gold_file_name}"

    pred_names = args.pred_file_names.split(',')
    pred_paths = [f"results/{pred_f_name}" for pred_f_name in pred_names]

    gold_data = json.load(open(gold_path, "r", encoding="utf-8"))
    pred_datas = [json.load(open(pred_path, "r", encoding="utf-8")) \
                               for pred_path in pred_paths]
    
    num_questions = len(gold_data)
    for f_idx in range(len(pred_datas)):
        pred_data = pred_datas[f_idx]
        assert len(pred_data) == num_questions, "length of prediction and of gold label must be the same"

        print (f">>> {pred_names[f_idx]}:")
        evaluate_results(gold_data, pred_data)

            





    