# Notes for ALQAC 2023's task 2

## Train extractive question answering models
```
python train_extractive_qa.py   --model_name_or_path "nguyenvulebinh/vi-mrc-large" --train_file generated_data/qa_extractive_question_data/extractive_qa_training_samples.json --validation_file generated_data/qa_extractive_question_data/extractive_qa_validation_samples.json  --do_train   --do_eval   --per_device_train_batch_size 12   --learning_rate 3e-5   --num_train_epochs 6   --max_seq_length 512   --doc_stride 128   --output_dir saved_model/finetuned_vi_mrc_large/
```

## Train yes/no question answering models

### On Google BoolQ Dataset

```
python train_boolean_qa.py --model_name_or_path  saved_model/finetuned_vi_mrc_large/ --train_file generated_data/qa_boolean_question_data/google_boolean_qa_training_samples.json --validation_file generated_data/qa_boolean_question_data/google_boolean_qa_validation_samples.json   --shuffle_train_dataset --metric_name accuracy --question_column_name question  --context_column_name context --label_column_name label --do_train --do_eval --max_seq_length 512 --per_device_train_batch_size 16 --learning_rate 2e-5 --num_train_epochs 3 --output_dir saved_model/boolq_google_ep3_finetuned_vi_mrc_large/
```

### On ALQAC 2023's boolean questions

```
python train_boolean_qa.py --model_name_or_path  saved_model/boolq_google_ep3_finetuned_vi_mrc_large/ --train_file generated_data/qa_boolean_question_data/alqac23_boolean_qa_training_samples.json --validation_file generated_data/qa_boolean_question_data/alqac23_boolean_qa_validation_samples.json  --shuffle_train_dataset --metric_name accuracy --question_column_name question  --context_column_name context --label_column_name label --do_train --do_eval --max_seq_length 512 --per_device_train_batch_size 8 --learning_rate 2e-5 --num_train_epochs 3 --output_dir saved_model/boolq_alqac23_ep3_google_ep3_finetuned_vi_mrc_large/
```