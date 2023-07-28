# Notes for ALQAC 2023's task 2

## Train extractive question answering models
```
python train_extractive_qa.py   --model_name_or_path "nguyenvulebinh/vi-mrc-large" --train_file generated_data/extractive_qa_training_samples.json --validation_file generated_data/extractive_qa_validation_samples.json  --do_train   --do_eval   --per_device_train_batch_size 12   --learning_rate 3e-5   --num_train_epochs 2   --max_seq_length 512   --doc_stride 128   --output_dir saved_model/finetuned_vi_mrc_large/
```

## Train yes/no question answering models

```
python train_boolean_qa.py --model_name_or_path  bert-base-uncased --dataset_name amazon_reviews_multi --dataset_config_name en --shuffle_train_dataset --metric_name accuracy --text_column_name "review_title,review_body,product_category" --text_column_delimiter "\n" --label_column_name stars --do_train --do_eval --max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 1 --output_dir saved_model/amazon_reviews_multi_en/
```