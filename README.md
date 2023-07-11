# alqac2023 source code
Source code for ALQAC 2023.

# Solution:

## Pipeline

We use the pipepline in the picture below:
<p align="center">
    <img src="figs/pipeline.png">
</p> 
Our pipeline is combination of BM25 and Sentence Transfromer. 
Let us describe our approach briefly:

- Step 1: We trained a BM25 model for searching similar pair. We used BM25 to create negative sentence pairs for training Sentence Transformer in Step 3.
- Step 2: We trained Masked Language Model using legal corpus from training data. Our masked languague models are:
```
FPTAI/ViBert
VinAI/PhoBert-Large
```
- Step 3: Train Sentence Transformer + Contrative loss with 4 settings:
```
1. MLM ViBert -> Sentence Transformer
2. MLM PhoBert Large -> Sentence Transformer 
3. MLM PhoBert Large -> Condenser -> Sentence Transformer
4. MLM PhoBert Large -> Co-Condenser -> Sentence Transformer
```
- Step 4: Using 4 models from step 3 to generate corresponding hard negative sentences for training round 2 in step 5.
- Step 5: Training 4 above models round 2.
- Step 5: Ensemble 4 models obtained from step 5.
 
## Data
Raw data is in ``alqac23_data``

## Create Folder

Create a new folder for generated data for training ``mkdir generated_data``

## Train BM 25
To train BM25: ``python bm25_train.py``
Use load_docs to save time for later run: ``python bm25_train.py --load_docs``

To evaluate: ``python bm25_create_pairs.py``
This step will also create top_k negative pairs from BM25. We choose top_k= 20, 50
Pairs will be saved to: generated_data/

These pairs will be used to train round 1 Sentence Transformer model

## Create corpus: 
Run ``python create_corpus.py``
This step will create:
- corpus.txt  (for finetune language model)
- cocondenser_data.json (for finetune CoCondenser model)

## Finetune language model using Huggingface
Pretrained model:
- viBERT: FPTAI/vibert-base-cased (`used`)
- phobert-large: vinai/phobert-large (`used`)
- vELECTRA: FPTAI/velectra-base-discriminator-cased (`not used`)
- phobert-base: vinai/phobert-base (`not used`)

Run the following cmd to train Masked Language Model:
```
python run_mlm.py \
    --model_name_or_path "vinai/phobert-large" \
    --train_file "generated_data/corpus.txt" \
    --do_train \
    --do_eval \
    --output_dir "saved_model" \
    --line_by_line \
    --overwrite_output_dir \
    --save_steps 2000 \
    --num_train_epochs 20 \
    --per_device_eval_batch_size 32 \
    --per_device_train_batch_size 32
```

##  Train condenser and cocondenser from language model checkpoint
Original source code here: https://github.com/luyug/Condenser (we modified several lines of code to make it compatible with current version of transformers)

### Create data for Condenser: 
 
```
python helper/create_train.py \
    --tokenizer_name $MODEL_NAME \
    --corpus_file $DATA_FILE \
    --save_path $SAVE_CONDENSER \
    --max_len $MAX_LENGTH 

$MODEL_NAME=vinai/phobert-large
$DATA_FILE=../generated_data/corpus.txt
$SAVE_CONDENSER=../generated_data/
$MAX_LENGTH=256
```

#### **Note**: $MODEL_NAME checkpoint from finetuned language model.

```
python run_pre_training.py \
  --output_dir $OUTDIR \
  --model_name_or_path $MODEL_NAME \
  --do_train \
  --save_steps 2000 \
  --per_device_train_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $ACCUMULATION_STEPS \
  --fp16 \
  --warmup_ratio 0.1 \
  --learning_rate 5e-5 \
  --num_train_epochs 8 \
  --overwrite_output_dir \
  --dataloader_num_workers 32 \
  --n_head_layers 2 \
  --skip_from 6 \
  --max_seq_length $MAX_LENGTH \
  --train_dir $SAVE_CONDENSER \
  --weight_decay 0.01 \
  --late_mlm
```
We use this setting to run Condenser:
```
python run_pre_training.py   \
    --output_dir saved_model_1/  \
    --model_name_or_path '/path/to/language/model'   \
    --do_train   
    --save_steps 2000   \
    --per_device_train_batch_size 32   \
    --gradient_accumulation_steps 4   \
    --fp16   \
    --warmup_ratio 0.1   \
    --learning_rate 5e-5   \
    --num_train_epochs 8   \
    --overwrite_output_dir   \
    --dataloader_num_workers 32   \
    --n_head_layers 2   \
    --skip_from 6   \
    --max_seq_length 256   \
    --train_dir ../generated_data/   \
    --weight_decay 0.01   \
    --late_mlm
```


## Train cocodenser:
First, we create data for cocodenser

```
python helper/create_train_co.py \
    --tokenizer vinai/phobert-large \
    --file ../generated_data/cocondenser_data.json \
    --save_path ../generated_data \
```

Run the following cmd to train co-condenser model:
```
python  run_co_pre_training.py   \
    --output_dir saved_model/cocondenser/   \
    --model_name_or_path $CONDENSER_CKPT   \
    --do_train   \
    --save_steps 2000   \
    --model_type bert   \
    --per_device_train_batch_size 32   \
    --gradient_accumulation_steps 1   \
    --fp16   \
    --warmup_ratio 0.1   \
    --learning_rate 5e-5   \
    --num_train_epochs 10   \
    --dataloader_drop_last   \
    --overwrite_output_dir   \
    --dataloader_num_workers 32   \
    --n_head_layers 2   \
    --skip_from 6   \
    --max_seq_length 256   \
    --train_dir ../generated_data/cocondenser/   \
    --weight_decay 0.01   \
    --late_mlm  \
    --cache_chunk_size 32 \
    --save_total_limit 1
```

## Train Sentence Transformer
### Round 1: using negative pairs of sentence generated from BM25
For each Masked Language Model, we trained a sentence transformer corresponding to it
Run the following command to train round 1 of sentence bert model

Note: Use cls_pooling for condenser and cocodenser
```
python train_sentence_bert.py 
    --pretrained_model /path/to/your/pretrained/mlm/model\
    --max_seq_length 256 \
    --pair_data_path /path/to/your/negative/pairs/data\
    --round 1 \
    --num_val $NUM_VAL\
    --epochs 10\
    --saved_model /path/to/your/save/model/directory\
    --batch_size 32\
```

here we pick $NUM_VAL is 50 * 20 and 50 * 50 for top 20 and 50 pairs data respectively

### Round 2: using hard negative pairs create from Round 1 model
- Step 1: Run the following cmd to generate hard negative pairs from round 1 model:
```
python hard_negative_mining.py \
    --model_path /path/to/your/sentence/bert/model\
    --data_path /path/to/the/lagal/corpus/json\
    --save_path /path/to/directory/to/save/neg/pairs\
    --top_k top_k_negative_pair
```
Here we pick top k is 20 and 50.
- Use the data generated from step 1 to train round 2 of sentence bert model for each model from round 1:
To train round 2, please use the following command:
```
python train_sentence_bert.py 
    --pretrained_model /path/to/your/pretrained/mlm/model\
    --max_seq_length 256 \
    --pair_data_path /path/to/your/negative/pairs/data\
    --round 2 \
    --num_val $NUM_VAL\
    --epochs 5\
    --saved_model /path/to/your/save/model/directory\
    --batch_size 32\
```
Tips: Use small learning rate for model convergence

## Prediction
### For reproducing result.
To get the prediction, we use 4 2-round trained models with mlm pretrained is viBert-based, PhoBert-Large, PhoBert-Large-Condenser, and Pho-Bert-Large-CoCondenser. Final models and their corresponding weights are below:
- 1 x FPTAI/ViBert-base-round2: 0.2
- 1 x PhoBert-Large-Round2: 0.1
- 1 x Condenser-PhoBert-Large-round2: 0.3
- 1 x Co-Condenser-PhoBert-Large-round2: 0.4

``flattened_corpus.pkl`` and ``legal_dict.json`` are generated in `bm25_train` and `create_corpus`, respectively. 

We provide embedding vectors which is pre-encoded by ensemble model in ``generated_data/embedded_corpus_data.pkl``.
If you want to verified and get the final submission, please run the following command:
```
python3 predict.py --data /path/to/test/data --legal_data generated_data/flattened_corpus.pkl --encode_legal_data
```

If you already have ``generated_data/embedded_corpus_data.pkl``, run the following command:
```
python3 predict.py --data /path/to/test/data --legal_data generated_data/flattened_corpus.pkl
```

### Just for inference

Run the following command 
```
chmod +x predict.sh
./predict.sh
```

## post-processing techniques:

- fix typo of nd-cp
- multiply cos-sim score with score from bm25, we pick score-range = [max-score - 2.6, max-score] and pick top 5 sentences for a question with multiple answers . 


## Methods used but not work

- Training Round 3 for Sentence Transformer.
- Pseudo Label: Improve our single model performace but hurt ensembel preformance.

## Contributors:
Thanks our teamates for great works: [Dzung Le](https://github.com/dzunglt24), [Hong Nguyen](https://github.com/Hong7Cong)
