# Commands for MLM finetuning, Co-/Condenser pre-training, Sentence-Transformers training

## Finetune models with MLM tasks

### vibert-base-cased
python run_mlm.py --model_name_or_path 'FPTAI/vibert-base-cased' --train_file 'generated_data/corpus.txt' --do_train --do_eval --output_dir 'saved_model/mlm_finetuned_vibert_base_cased' --line_by_line --overwrite_output_dir --save_steps 2000 --num_train_epochs 20 --per_device_eval_batch_size 32 --per_device_train_batch_size 32 --max_seq_length 512

### phobert-large
python run_mlm.py --model_name_or_path 'vinai/phobert-large' --train_file 'generated_data/corpus.txt' --do_train --do_eval --output_dir 'saved_model/mlm_finetuned_phobert_large' --line_by_line --overwrite_output_dir --save_steps 2000 --num_train_epochs 20 --per_device_eval_batch_size 32 --per_device_train_batch_size 32

## Create data for pre-training condenser
python Condenser/helper/create_train.py --tokenizer_name saved_model/mlm_finetuned_phobert_large --corpus_file generated_data/corpus.txt --save_path generated_data/condenser_data_encoded --max_len 256 

## pre-train condenser
python Condenser/run_pre_training.py --output_dir saved_model/condenser_phobert_large --model_name_or_path saved_model/mlm_finetuned_phobert_large --do_train --save_steps 2000 --per_device_train_batch_size 32 --gradient_accumulation_steps 4 --fp16 --warmup_ratio 0.1 --learning_rate 5e-5 --num_train_epochs 8 --overwrite_output_dir --dataloader_num_workers 32 --remove_unused_columns False --n_head_layers 2 --skip_from 6 --max_seq_length 256 --train_dir generated_data/condenser_data_encoded --weight_decay 0.01 --late_mlm

## Create data for pre-training co-condenser
python Condenser/helper/create_train_co.py --tokenizer vinai/phobert-large --file generated_data/cocondenser_data.json --save_path generated_data/cocondenser_data_encoded

## pre-train co-condenser
python Condenser/run_co_pre_training.py --output_dir saved_model/cocondenser_phobert_large/ --model_name_or_path saved_model/condenser_phobert_large/ --do_train --save_steps 2000 --model_type bert --per_device_train_batch_size 32 --gradient_accumulation_steps 1 --fp16 --warmup_ratio 0.1 --learning_rate 5e-5 --num_train_epochs 10 --dataloader_drop_last --overwrite_output_dir --dataloader_num_workers 32 --n_head_layers 2 --skip_from 6 --max_seq_length 256 --train_dir generated_data/cocondenser_data_encoded/ --weight_decay 0.01 --late_mlm --cache_chunk_size 32 --save_total_limit 1

## Train sentence transformers (round 1)

### mlm_finetuned_vibert_base_cased
python train_sentence_bert.py --pretrained_model saved_model/mlm_finetuned_vibert_base_cased --max_seq_length 512 --pair_data_path generated_data/qrel_pairs_bm25_top20 --round 1 --num_eval 1000 --epochs 4 --saved_model saved_model/sbert_round1_epoch4_top20_mlm_finetuned_vibert_base_cased --batch_size 32

### mlm_finetuned_phobert_large
python train_sentence_bert.py --pretrained_model saved_model/mlm_finetuned_phobert_large --max_seq_length 256 --pair_data_path generated_data/qrel_pairs_bm25_top20 --round 1 --num_eval 1000 --epochs 4 --saved_model saved_model/sbert_round1_epoch4_top20_mlm_finetuned_phobert_large --batch_size 32

### condenser_phobert_large, initialize with cls_pooling
python train_sentence_bert.py --pretrained_model saved_model/condenser_phobert_large --max_seq_length 256 --pooling_mode cls  --pair_data_path generated_data/qrel_pairs_bm25_top20 --round 1 --num_eval 1000 --epochs 4 --saved_model saved_model/sbert_round1_epoch4_top20_condenser_phobert_large --batch_size 32

### cocondenser_phobert_large, initialize with cls_pooling
python train_sentence_bert.py --pretrained_model saved_model/cocondenser_phobert_large --max_seq_length 256 --pooling_mode cls  --pair_data_path generated_data/qrel_pairs_bm25_top20 --round 1 --num_eval 1000 --epochs 4 --saved_model saved_model/sbert_round1_epoch4_top20_cocondenser_phobert_large --batch_size 32

## Negative samples mining from trained sbert models

### sbert_vibert_base_cased
python hard_negative_mining.py --sbert_model_path saved_model/sbert_round1_epoch4_top20_mlm_finetuned_vibert_base_cased --data_path alqac23_data --save_path generated_data --top_k 35 [--load_embedding]

### sbert_phobert_large
python hard_negative_mining.py --sbert_model_path saved_model/sbert_round1_epoch4_top20_mlm_finetuned_phobert_large --data_path alqac23_data --save_path generated_data --top_k 35 [--load_embedding]


### sbert_condenser_phobert_large
python hard_negative_mining.py --sbert_model_path saved_model/sbert_round1_epoch4_top20_condenser_phobert_large --data_path alqac23_data --save_path generated_data --top_k 35 [--load_embedding]


### sbert_cocondender_phobert_large
python hard_negative_mining.py --sbert_model_path saved_model/sbert_round1_epoch4_top20_cocondenser_phobert_large --data_path alqac23_data --save_path generated_data --top_k 35 [--load_embedding]

## Train sentence transformers (round 2)

### sbert_round1_epoch4_top20_mlm_finetuned_vibert_base_cased
python train_sentence_bert.py --pretrained_model saved_model/sbert_round1_epoch4_top20_mlm_finetuned_vibert_base_cased --max_seq_length 512 --pair_data_path generated_data/qrel_pairs_sbert_round1_epoch4_top20_mlm_finetuned_vibert_base_cased_top35.pkl --round 2 --num_eval 1000 --epochs 4 --saved_model saved_model/sbert_round2_epoch4_top35_mlm_finetuned_vibert_base_cased --batch_size 32

### sbert_round1_epoch4_top20_mlm_finetuned_phobert_large
python train_sentence_bert.py --pretrained_model saved_model/sbert_round1_epoch4_top20_mlm_finetuned_phobert_large --max_seq_length 256 --pair_data_path generated_data/qrel_pairs_sbert_round1_epoch4_top20_mlm_finetuned_phobert_large_top35.pkl --round 2 --num_eval 1000 --epochs 4 --saved_model saved_model/sbert_round2_epoch4_top35_mlm_finetuned_phobert_large --batch_size 32

### sbert_round1_epoch4_top20_condenser_phobert_large
python train_sentence_bert.py --pretrained_model saved_model/sbert_round1_epoch4_top20_condenser_phobert_large --max_seq_length 256 --pair_data_path generated_data/qrel_pairs_sbert_round1_epoch4_top20_condenser_phobert_large_top35.pkl --round 2 --num_eval 1000 --epochs 4 --saved_model saved_model/sbert_round2_epoch4_top35_condenser_phobert_large --batch_size 32

### sbert_round1_epoch4_top20_cocondenser_phobert_large
python train_sentence_bert.py --pretrained_model saved_model/sbert_round1_epoch4_top20_cocondenser_phobert_large --max_seq_length 256 --pair_data_path generated_data/qrel_pairs_sbert_round1_epoch4_top20_cocondenser_phobert_large_top35.pkl --round 2 --num_eval 1000 --epochs 4 --saved_model saved_model/sbert_round2_epoch4_top35_cocondenser_phobert_large --batch_size 32