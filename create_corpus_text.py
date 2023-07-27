import json
import os
import re
from tqdm import tqdm
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", default="alqac23_data", type=str, help="directory to raw data")
    parser.add_argument("--corpus_list", default="alqac23,alqac22", type=str, help="corpus name list for corpus text and co-condenser data. The item must be in [`alqac23`,`alqac22`,`zalo`]")
    parser.add_argument("--save_dir", default="generated_data", type=str, help="directory to save corpus text")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir,exist_ok=True)

    # remove accidentally duplicate corpus with the order kept
    corpus_names = []
    for corpus in args.corpus_list.split(','):
        if corpus not in corpus_names:
            corpus_names.append(corpus)

    for corpus_name in corpus_names:
        assert corpus_name in ['alqac23', 'alqac22', 'zalo'], "corpus name item must be in [`alqac23`,`alqac22`,`zalo`"
    
    save_corpus_name = f"{'_'.join(corpus_names)}_corpus.txt"
    cp = open(os.path.join(args.save_dir, save_corpus_name), "w")

    alqac23_corpus_path = os.path.join(args.raw_data_dir, "law.json")
    alqac22_corpus_path = os.path.join(args.raw_data_dir, "additional_data/ALQAC_2022_training_data/law.json")
    zalo_corpus_path = os.path.join(args.raw_data_dir, "additional_data/zalo/zalo_corpus.json")
        
    corpus_paths = {
        "alqac23": alqac23_corpus_path,
        "alqac22": alqac22_corpus_path,
        "zalo": zalo_corpus_path
    }
    
    data = []
    for corpus_name in corpus_names:
        data.extend(json.load(open(corpus_paths[corpus_name])))

    save_cocondenser_name = f"{'_'.join(corpus_names)}_cocondenser_data.json"
    co_f = open(os.path.join(args.save_dir, save_cocondenser_name), "w")
    count = 0
    unique_keys = []
    for law_article in tqdm(data):
        law_id = law_article["id"]
        law_articles = law_article["articles"]
        
        for sub_article in law_articles:
            article_id = sub_article["id"]
            article_text = sub_article["text"]
            
            if (law_id + "_" + article_id) in unique_keys:
                continue
            unique_keys.append(law_id + "_" + article_id)

            article_full = re.sub(r'\n+', " ", article_text)
            cp.write(article_full + "\n")
            
            # Save data for cocondenser 
            spans = []
            passages = re.split(r"\n+[0-9]{1,3}\.", article_text)
            for p in passages:
                if p != "":
                    p_1 = re.sub(r'([\:;.])\n+', r'\1 ', p)
                    p_2 = re.sub(r'\n+', r'. ', p_1)
                    spans.append(p_2)
            co_f.write("#".join(spans) + "\n")

    co_f.close()
    print(f"Co-condenser data for {corpus_names} created!")

    # enrich corpus from train files
    alqac23_corpus_path_train = os.path.join(args.raw_data_dir, "train.json")
    alqac22_corpus_path_train = os.path.join(args.raw_data_dir, "additional_data/ALQAC_2022_training_data/question.json")
    zalo_corpus_path_train = os.path.join(args.raw_data_dir, "additional_data/zalo/zalo_question.json")
    
    train_corpus_paths = {
        "alqac23": alqac23_corpus_path_train,
        "alqac22": alqac22_corpus_path_train,
        "zalo": zalo_corpus_path_train
    }
    
    train_items = []    
    for corpus_name in corpus_names:
        train_items.extend(json.load(open(train_corpus_paths[corpus_name])))

    for item in tqdm(train_items):
        question = re.sub(r'\s*\n+\s*', " ", item["text"]) 
        cp.write(question + "\n")

    # enrich corpus from public test file
    if "alqac23" in corpus_names:
        alqac23_corpus_path_test = os.path.join(args.raw_data_dir, "public_test.json")
        public_items = json.load(open(alqac23_corpus_path_test))

        for item in tqdm(public_items):
            question = re.sub(r'\s*\n+\s*', " ", item["text"]) 
            cp.write(question + "\n")

    cp.close()
    print(f"Corpus text from {corpus_names} created!")
