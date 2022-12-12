import argparse
import timeit
import nltk
import numpy as np
from tqdm.notebook import tqdm_notebook 
import tqdm
import pandas as pd
import os
import math
import warnings
warnings.filterwarnings('ignore')
import argparse
import timeit
from ir_system import IRSystem
def load_data(path):
    
    
    #_____________ Read data from CISI.ALL file and store in dictinary ________________
    
    with open(os.path.join(path, 'CISI.ALL')) as f:
        lines = ""
        for l in f.readlines():
            lines += "\n" + l.strip() if l.startswith(".") else " " + l.strip()
        lines = lines.lstrip("\n").split("\n")
 
    doc_set = {}
    doc_id = ""
    doc_text = ""

    for l in lines:
        if l.startswith(".I"):
            doc_id = l.split(" ")[1].strip() 
        elif l.startswith(".X"):
            doc_set[doc_id] = doc_text.lstrip(" ")
            doc_id = ""
            doc_text = ""
        elif l.startswith(".T") or l.startswith(".W"):
            doc_text += l.strip()[3:] + " "

    print(f"Number of documents = {len(doc_set)}")
    print(doc_set["1"]) 
    
    
    doc_set = {int(id):doc for (id,doc) in doc_set.items()}
    
    return doc_set

doc_set= load_data('documents')
ExpReg = nltk. RegexpTokenizer('(?:[A-Za-z]\.)+|\d+(?:\.\d+)?%?|\w+(?:\-\w+)*')
MotsVides = nltk.corpus.stopwords.words('english')
Porter = nltk.PorterStemmer()
Lancaster = nltk.LancasterStemmer()

stop_words = ['is', 'a', 'for', 'the', 'of']

def parse_args():
    parser = argparse.ArgumentParser(description='Information Retrieval System Configuration')
    return parser.parse_args()

def main():
    args = parse_args()
    ir = IRSystem(doc_set, stop_words=stop_words)

    while True:
        query = 'information AND classification OR NOT title AND computers'

        start = timeit.default_timer()
        results = ir.process_query(query)
        stop = timeit.default_timer()
        if results is not None:
            print ('Processing time: {:.5} secs'.format(stop - start))
            print('\nDoc IDS: ')
            print(results)
        print()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        print('EXIT')