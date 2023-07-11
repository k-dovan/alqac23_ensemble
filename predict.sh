python3 create_corpus.py --data alqac23_data --save_dir generated_data
python3 preprocessing.py --raw_data alqac23_data --save_path generated_data
CUDA_VISIBLE_DEVICES=0 python3 predict.py --data $data --legal_data generated_data/flattened_corpus.pkl