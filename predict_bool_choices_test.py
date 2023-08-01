
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import argparse
import torch
import json
from utils import segment_long_text, remove_newlines, clean_multichoice_question

if __name__ == '__main__':

    print ("Loading the model")
    tokenizer = AutoTokenizer.from_pretrained("saved_model/boolq_alqac23_ep3_google_ep3_finetuned_vi_mrc_large", device_map="auto", load_in_8bit=True)
    model = AutoModelForSequenceClassification.from_pretrained("saved_model/boolq_alqac23_ep3_google_ep3_finetuned_vi_mrc_large", device_map="auto", load_in_8bit=True)
    print ("Done")

    question = "Thời hạn cơ quan đăng ký cư trú thẩm định, cập nhật thông tin về hộ gia đình liên quan đến việc tách hộ vào Cơ sở dữ liệu về cư trú là bao lâu kể từ ngày nhận được hồ sơ đầy đủ và hợp lệ?"
    choices = [
        "Dịch vụ thể thao.",
        "Dịch vụ ăn uống.",
        "Dịch vụ chăm sóc sức khỏe."
    ]
    context = "Các loại dịch vụ du lịch khác\n1. Dịch vụ ăn uống.\n\n2. Dịch vụ mua sắm.\n\n3. Dịch vụ thể thao.\n\n4. Dịch vụ vui chơi, giải trí.\n\n5. Dịch vụ chăm sóc sức khỏe.\n\n6. Dịch vụ liên quan khác phục vụ khách du lịch."

    question = clean_multichoice_question(question)
    print ("cleaned question: ", question)

    # input_tuples = [(' '.join([question, opt]), context) for opt in choices]

    # batch_inputs = tokenizer(input_tuples, return_tensors="pt", truncation=True, padding=True)

    # with torch.no_grad():
    #     logits = model(**batch_inputs).logits
    # print ("logits: ", logits)

    # class_ids = logits.argmax(dim=1).tolist()
    # print (class_ids)   