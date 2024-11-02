#!/usr/bin/env python3
import json
import gzip
from tira import ir_datasets

def load_all_docs():
    if not os.path.exists('../../data/expansion-documents-text.json.gz'):
        splits = json.load(gzip.open('../../data/expansion-documents.json.gz', 'rt'))
        docs = []
        for split, training_data in splits.items():
            for query, docs in training_data.items():
                for doc in docs:
                    docs += [doc]
        docs = set(docs)
        ret = {}
        for dataset_id in tqdm(['longeval-train-20230513-training', 'longeval-heldout-20230513-training longeval-short-july-20230513-training', 'longeval-long-september-20230513-training', 'longeval-2023-01-20240423-training' 'longeval-2023-06-20240418-training', 'longeval-2023-08-20240418-training']):
            dataset = ir_datasets.load(f'ir-benchmarks/{dataset_id}')
            for doc in dataset.docs_iter():
                if doc.doc_id in docs:
                    ret[doc.doc_id] = doc.default_text()
        if len(docs) != len(ret):
            raise ValueError(len(docs), '!=', len(ret))
        with gzip.open('../../data/expansion-documents-text.json.gz', 'wt') as f:
            json.dump(f, ret)
    return json.load(gzip.open('../../data/expansion-documents-text.json.gz', 'rt'))

if __name__ == '__main__':
    docs = load_all_docs()
    

    
    print(len(docs))
