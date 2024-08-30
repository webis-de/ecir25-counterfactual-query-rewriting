#!/usr/bin/env python3
from tira.third_party_integrations import ir_datasets
import gzip
import json
from tqdm import tqdm
import csv

def all_doc_groups():
    with gzip.open('../data/document-groups-judged-extended.csv.gz', 'rt') as f:
        for l in csv.reader(f):
            url = l[0]
            if url == 'url':
                continue # skip header :)
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

expected_docs = set()
for _, i in all_doc_groups():
    for j in i:
        expected_docs.add(j)


docs = {}
for d in DS:
    #docs_iter = ir_datasets.load(f'ir-benchmarks/{d}').docs_iter()

    # needs to run the above first
    with gzip.open(f'/mnt/ceph/storage/data-tmp/current/kibi9872/.tira/extracted_datasets/ir-benchmarks/{d}/input-data/documents.jsonl.gz') as f:
        for i in tqdm(f, d):
            i = json.loads(i)
            if i['docno'] not in expected_docs:
                continue
            docs[i['docno']] = i['text']


with open('../data/document-groups.jsonl', 'w') as outp:
    for url, l in all_doc_groups():
        outp.write(json.dumps({'url': url, 'docs': {i: docs[i] for i in l}}) + '\n')

