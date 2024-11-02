#!/usr/bin/env python3
import json
import gzip
import os
from tira.third_party_integrations import ir_datasets, ensure_pyterrier_is_loaded
from tqdm import tqdm
import pyterrier as pt

def load_all_docs():
    if not os.path.exists('../../data/expansion-documents-text.json.gz'):
        splits = json.load(gzip.open('../../data/expansion-documents.json.gz', 'rt'))
        docs = []
        for split, training_data in splits.items():
            for query, docs_for_query in training_data.items():
                for doc in docs_for_query:
                    docs += [doc]
        docs = set(docs)
        ret = {}
        for dataset_id in tqdm(['longeval-train-20230513-training', 'longeval-heldout-20230513-training', 'longeval-short-july-20230513-training', 'longeval-long-september-20230513-training', 'longeval-2023-01-20240423-training', 'longeval-2023-06-20240418-training', 'longeval-2023-08-20240418-training']):
            dataset = ir_datasets.load(f'ir-benchmarks/{dataset_id}')
            for doc in dataset.docs_iter():
                if doc.doc_id in docs:
                    ret[doc.doc_id] = doc.default_text()
        if len(docs) != len(ret):
            raise ValueError(len(docs), '!=', len(ret))
        with gzip.open('../../data/expansion-documents-text.json.gz', 'wt') as f:
            json.dump(ret, f)
    return json.load(gzip.open('../../data/expansion-documents-text.json.gz', 'rt'))

if __name__ == '__main__':
    docs = load_all_docs()
    ensure_pyterrier_is_loaded()
    splits = json.load(gzip.open('../../data/expansion-documents.json.gz', 'rt'))
    for split, training_data in splits.items():
        docs_for_split = []
        for query, docs_for_query in training_data.items():
            for doc in docs_for_query:
                docs_for_split += [doc]
        docs_for_split = set(docs_for_split)
        iter_indexer = pt.IterDictIndexer(os.path.abspath(f"indices/{split}/"), meta={'docno': 50, 'text': 4096}, overwrite=True)
        iter_indexer.index(tqdm([{'docno': i, 'text': docs[i]} for i in docs_for_split], 'Index'))

