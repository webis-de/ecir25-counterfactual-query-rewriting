#!/usr/bin/env python3
from tira.third_party_integrations import ir_datasets

DS = [
    'longeval-2023-01-20240423-training',
    'longeval-2023-06-20240418-training',
    'longeval-2023-08-20240418-training',
    'longeval-heldout-20230513-training',
    'longeval-long-september-20230513-training',
    'longeval-short-july-20230513-training',
    'longeval-train-20230513-training'
]

for d in DS:
    ir_datasets.load(f'ir-benchmarks/{d}').docs_store()
