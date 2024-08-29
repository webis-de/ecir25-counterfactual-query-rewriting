#!/usr/bin/env python3
from tira.third_party_integrations import ir_datasets
import gzip
import json
from tqdm import tqdm
import csv

DS = [
    'longeval-2023-01-20240423-training',
    'longeval-2023-06-20240418-training',
    'longeval-2023-08-20240418-training',
    'longeval-heldout-20230513-training',
    'longeval-long-september-20230513-training',
    'longeval-short-july-20230513-training',
    'longeval-train-20230513-training'
]

DS = [ir_datasets.load(f'ir-benchmarks/{i}').docs_store() for i in tqdm(DS, 'Load docs_stores')]

def default_text(docid):
    for d in DS:
        if docid in d:
            return d[docid].default_text()

    raise ValueError('Could not find docid:', docid)

with gzip.open('../data/document-groups-judged.csv.gz', 'rt') as f, open('../data/document-groups.jsonl', 'w') as outp:
    for l in csv.reader(f):
        url = l[0]
        l.remove(url)
        assert len(l) == 6
        l = [i for i in l if i]

        if len(l) <= 1:
            continue
        outp.write(json.dumps({'url': url, 'docs': {i: default_text(i) for i in l}}) + '\n')

