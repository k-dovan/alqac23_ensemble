# Notes for ALQAC 2023's task 2

## Train extractive question answering models
```
screen -dm bash -c "python train_extractive_qa.py   --model_name_or_path nguyenvulebinh/vi-mrc-large --train_file generated_data/qa_extractive_question_data/extractive_qa_training_samples.json --validation_file generated_data/qa_extractive_question_data/extractive_qa_validation_samples.json  --do_train   --do_eval   --per_device_train_batch_size 12   --learning_rate 3e-5   --num_train_epochs 20  --max_seq_length 512   --doc_stride 128  --logging_steps 200 --save_steps 200 --eval_steps 200 --save_total_limit 5  --output_dir saved_model/finetuned_ep20_vi_mrc_large/ 2> logs/finetuned_ep20_vi_mrc_large.log"
```

## Train yes/no question answering models

### On Google BoolQ Dataset

```
python train_boolean_qa.py --model_name_or_path  saved_model/finetuned_vi_mrc_large/ --train_file generated_data/qa_boolean_question_data/google_boolean_qa_training_samples.json --validation_file generated_data/qa_boolean_question_data/google_boolean_qa_validation_samples.json   --shuffle_train_dataset --metric_name accuracy --question_column_name question  --context_column_name context --label_column_name label --do_train --do_eval --max_seq_length 512 --per_device_train_batch_size 16 --learning_rate 2e-5 --num_train_epochs 3 --logging_steps [1000] --save_steps [1000] --eval_steps [1000] --save_total_limit [5]  --output_dir saved_model/boolq_google_ep3_finetuned_vi_mrc_large/
```

### On ALQAC 2023's boolean questions

```
python train_boolean_qa.py --model_name_or_path  saved_model/boolq_google_ep3_finetuned_vi_mrc_large/ --train_file generated_data/qa_boolean_question_data/alqac23_boolean_qa_training_samples.json --validation_file generated_data/qa_boolean_question_data/alqac23_boolean_qa_validation_samples.json  --shuffle_train_dataset --metric_name accuracy --question_column_name question  --context_column_name context --label_column_name label --do_train --do_eval --max_seq_length 512 --per_device_train_batch_size 16 --learning_rate 2e-5 --num_train_epochs 50 --logging_steps 100 --save_steps 100 --eval_steps 100 --save_total_limit 5 --output_dir saved_model/boolq_alqac23_ep50_google_ep3_finetuned_vi_mrc_large/
```


# Experiments

- Model `boolq_alqac23_ep50_google_ep3_finetuned_vi_mrc_large`:                boolean acc: 0.88/ multi-choice acc: 0.875
- Model `boolq_alqac23_ep50_google_ep3_finetuned_vi_mrc_large/checkpoint-100`: boolean acc: 0.84/ multi-choice acc: 0.95
- Model `boolq_alqac23_ep50_google_ep3_finetuned_vi_mrc_large/checkpoint-200`: boolean acc: 0.88/ multi-choice acc: 0.90
- Model `boolq_alqac23_ep50_google_ep3_finetuned_vi_mrc_large/checkpoint-300`: boolean acc: 0.88/ multi-choice acc: 0.90
- Model `boolq_alqac23_ep50_google_ep3_finetuned_vi_mrc_large/checkpoint-400`: boolean acc: 0.88/ multi-choice acc: 0.875

## TODOS:
## most of wrong predictions from all-standalone-choice questions -> model embedding improve?
## all-true-/all-false-choice questions are quite good -> model scoring fairly good -> train more data

- Train more data
- Ensemble models for scoring
- Check cleaned multi-choice question methods + choice -> remain the meaning of questions
- Create script for task 2 -> public test .json
