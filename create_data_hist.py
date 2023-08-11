# ========================================================================================================
# This file is for generating data distribution/histogram of questions/articles on specific data corpus.
# ========================================================================================================

import os
import argparse
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", default="alqac23_data", type=str, help="path to raw corpus")
    parser.add_argument("--corpus_name", default="alqac23", type=str, choices=["alqac23", "alqac22", "zalo"], help="corpus to evaluate")
    parser.add_argument("--save_path", default="generated_data/histograms", type=str, help="path to save histograms")
    args = parser.parse_args()

    # articles' paths
    alqac23_corpus_path = os.path.join(args.raw_data_dir, "law.json")
    alqac22_corpus_path = os.path.join(args.raw_data_dir, "additional_data/ALQAC_2022_training_data/law.json")
    zalo_corpus_path = os.path.join(args.raw_data_dir, "additional_data/zalo/zalo_corpus.json")

    # questions' paths
    alqac23_corpus_path_train = os.path.join(args.raw_data_dir, "train.json")
    alqac22_corpus_path_train = os.path.join(args.raw_data_dir, "additional_data/ALQAC_2022_training_data/question.json")
    zalo_corpus_path_train = os.path.join(args.raw_data_dir, "additional_data/zalo/zalo_question.json")
        
    corpus_paths = {
        "alqac23": alqac23_corpus_path,
        "alqac22": alqac22_corpus_path,
        "zalo": zalo_corpus_path
    }
    
    train_corpus_paths = {
        "alqac23": alqac23_corpus_path_train,
        "alqac22": alqac22_corpus_path_train,
        "zalo": zalo_corpus_path_train
    }
    
    global_data_lengths = []
    for corpus_name, data_path in corpus_paths.items():
        data = json.load(open(data_path))
            
        data_lengths = []
        for law_article in tqdm(data):
            law_articles = law_article["articles"]
            
            outlier_thresh = 7000
            for sub_article in law_articles:
                data_len = len(sub_article["text"])
                # from find outliers -> threshold of outliers
                if data_len < outlier_thresh:
                    data_lengths.append(data_len)
                    global_data_lengths.append(data_len)

        plt.title(label = f"[{corpus_name}] Article Length Distribution", loc='center')
        plt.xlabel("Length")  
        plt.ylabel("Frequency")  
        plt.hist(data_lengths, bins=20)
        plt.savefig(f'{args.save_path}/hist_{corpus_name}_articles.png')
        plt.close()
    
    # create combined histogram
    plt.title(label = f"Article Length Distribution", loc='center')
    plt.xlabel("Length")  
    plt.ylabel("Frequency")
    plt.hist(global_data_lengths, bins=20)
    plt.savefig(f'{args.save_path}/hist_all_articles.png')
    plt.close()
    
    global_data_lengths = []
    for corpus_name, train_path in train_corpus_paths.items():
        data = json.load(open(train_path))
            
        data_lengths = []
        for item in tqdm(data):
            data_len = len(item["text"])
            data_lengths.append(data_len)   
            global_data_lengths.append(data_len)                 
        
        plt.title(label = f"[{corpus_name}] Question Length Distribution", loc='center')
        plt.xlabel("Length")  
        plt.ylabel("Frequency")  
        plt.hist(data_lengths, bins=20)
        plt.savefig(f'{args.save_path}/hist_{corpus_name}_questions.png')
        plt.close()
    
    # create combined histogram
    plt.title(label = f"Question Length Distribution", loc='center')
    plt.xlabel("Length")  
    plt.ylabel("Frequency")
    plt.hist(global_data_lengths, bins=20)
    plt.savefig(f'{args.save_path}/hist_all_questions.png')
    plt.close()
                
def find_outliers(corpus_name, data_path):
    data = json.load(open(data_path))
            
    data_lengths = []
    for law_article in tqdm(data):
        law_articles = law_article["articles"]
        
        for sub_article in law_articles:
            data_lengths.append(len(sub_article["text"]))
    
    np_arr = np.array(data_lengths)
    top_value_idxs = np.argpartition(np_arr, -50)[-50:]
    print (np_arr[top_value_idxs])
    plt.hist(np_arr[top_value_idxs], bins=20)
    plt.savefig(f'{args.save_path}/hist_{corpus_name}_outliers.png')
    plt.close()
        