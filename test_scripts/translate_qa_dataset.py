import json
import threading
from threading import Thread
from googletrans import Translator

def translate_data(data: list, type: str = "train", chunk: int = 0):
    
    translator = Translator()
    dest_lang = "vi"

    thread_id = threading.current_thread().ident
    # translate data
    for idx in range(len(data)):
        question = data[idx]["question"]
        passage = data[idx]["passage"]

        translated_question = translator.translate(question, dest=dest_lang)
        translated_passage = translator.translate(passage, dest=dest_lang)
        
        # update data
        data[idx]["translated_question"] = translated_question.text
        data[idx]["translated_passage"] = translated_passage.text

        print (f">>> thread-{thread_id}: train item {idx+1} processed!")

    # save translation
    with open(f"data/translated_{type}_{chunk}.json", "w", encoding="utf-8") as outfile:
        json_object = json.dumps(data, indent=4, ensure_ascii=False)
        outfile.write(json_object) 

    print (f"Thread-{thread_id}: samples translated: {len(data)}")

train_file_name = "data/train.json"
dev_file_name = "data/dev.json"

train_data = json.load(open(train_file_name, "r", encoding="utf-8"))
dev_data = json.load(open(dev_file_name, "r", encoding="utf-8"))

num1_threads = 5
num2_threads = 3

sample1_per_thread = len(train_data)//num1_threads
sample2_per_thread = len(dev_data)//num2_threads

train_chunks = [[]]*num1_threads
dev_chunks = [[]]*num2_threads

for idx in range(num1_threads):
    if idx < num1_threads - 1:
        train_chunks[idx] = train_data[sample1_per_thread*idx : sample1_per_thread*(idx+1)]
    else:        
        train_chunks[idx] = train_data[sample1_per_thread*idx : len(train_data)]

for idx in range(num2_threads):
    if idx < num2_threads - 1:
        dev_chunks[idx] = dev_data[sample2_per_thread*idx : sample2_per_thread*(idx+1)]
    else:        
        dev_chunks[idx] = dev_data[sample2_per_thread*idx : len(dev_data)]

threads = [None]*(num1_threads+num2_threads)
for idx in range(num1_threads):
    threads[idx] = Thread(target = translate_data, args = (train_chunks[idx], "train", idx))
    threads[idx].start()


for idx in range(num2_threads):
    threads[num1_threads+idx] = Thread(target = translate_data, args = (dev_chunks[idx], "dev", idx))
    threads[num1_threads+idx].start()

# wait for all threads complete
for thread in threads:
    thread.join()