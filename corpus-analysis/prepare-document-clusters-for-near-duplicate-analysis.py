#!/usr/bin/env python3
from tira.third_party_integrations import ir_datasets
import gzip
import json
from tqdm import tqdm
import csv

def all_doc_groups():
    with gzip.open('../data/document-groups-judged.csv.gz', 'rt') as f:
        for l in csv.reader(f):
            url = l[0]
            l.remove(url)
            assert len(l) == 6
            l = [i for i in l if i]

            if len(l) <= 1:
                continue
            yield url, l


DS = [
    'longeval-2023-01-20240423-training',
    'longeval-2023-06-20240418-training',
    'longeval-2023-08-20240418-training',
    'longeval-heldout-20230513-training',
    'longeval-long-september-20230513-training',
    'longeval-short-july-20230513-training',
    'longeval-train-20230513-training'
]

docs = {}
for d in tqdm(DS, 'Load docs_stores'):
    d = ir_datasets.load(f'ir-benchmarks/{d}').docs_store()
    for _, i in all_doc_groups():
        for j in i:
            if j in d:
                docs[j] = d.default_text()


with open('../data/document-groups.jsonl', 'w') as outp:
    for url, l in all_doc_groups():
        outp.write(json.dumps({'url': url, 'docs': {i: docs[i] for i in l}}) + '\n')

