from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

id2label = {0: "false", 1: "true"}
label2id = {"false": 0, "true": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    "saved_model/mlm_finetuned_phobert_large_all_corpora_ep10/checkpoint-4000", num_labels=2, id2label=id2label, label2id=label2id
)