#!/usr/bin/env python3
import pandas as pd
from tira.rest_api_client import Client
from tira.third_party_integrations import ensure_pyterrier_is_loaded, persist_and_normalize_run, ir_datasets
from tqdm import tqdm

# This method ensures that that PyTerrier is loaded so that it also works in the TIRA sandbox
ensure_pyterrier_is_loaded()
import pyterrier as pt

LAG_TO_DATASET_ID = {
    'lag1': 'longeval-2023-01-20240423-training',
    'lag6': 'longeval-2023-06-20240418-training',
    'lag8': 'longeval-2023-08-20240418-training'
}

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('lag', type=str, help='The dataset to run on', choices=['lag1', 'lag6', 'lag8'])
    return parser.parse_args()

def __normalize_queries(q):
    return q.lower().strip()

def main(lag):
    tira = Client()
    dataset_id = LAG_TO_DATASET_ID[lag]
    pt_dataset = pt.get_dataset(f'irds:ir-benchmarks/{dataset_id}')
    oracle_index = load_oracle_index()
    bm25_raw = tira.pt.from_submission('ir-benchmarks/tira-ir-starter/BM25 (tira-ir-starter-pyterrier)', dataset_id) %900
    index = tira.pt.index('ir-benchmarks/tira-ir-starter/Index (tira-ir-starter-pyterrier)', dataset_id)
    docs = bm25_raw(pt_dataset.get_topics('text'))

    print('Determine overlapping queries...')

    overlapping_queries = {i.query_id: i.default_text() for i in pt_dataset.irds_ref().queries_iter()}
    overlapping_queries = {k: v for k, v in overlapping_queries.items() if __normalize_queries(v) in oracle_index}

    print(f'Found {len(overlapping_queries)} overlapping queries.')


    doc_ids = []

    for _, i in bm25_raw(pt_dataset.get_topics('text')).iterrows():
        if i['qid'] in overlapping_queries:
            doc_ids.append(i['docno'])

    doc_ids = set(doc_ids)
    print('Use docs queries for overlap: {len(doc_ids)}')

    docs_for_reformulation = []

    for i in tqdm(pt_dataset.get_corpus_iter()):
        if i['docno'] not in doc_ids:
            continue
        docs_for_reformulation += [i]

    additional_docs = {}

    for i in oracle_index:
        for j in oracle_index[i]:
            additional_docs[j['doc_id']] = j['doc']

    additional_docs = [{'docno': 'ADD_' + k, 'text': v} for k, v in additional_docs.items()]

    print('Select overlapping topics...')
    topics = pt_dataset.get_topics('text')
    topics = topics[topics['qid'].isin(overlapping_queries.keys())]

    print(f'Done. Found {len(topics)} overlapping topics.')
    
    print('docs_for_reformulation:', len(docs_for_reformulation))
    print('additional_docs:', len(additional_docs))

    iter_indexer = pt.IterDictIndexer(f"/tmp/reformulation-index-{lag}", meta={'docno': 50, 'text': 4096}, overwrite=True)
    index_old = iter_indexer.index(tqdm(docs_for_reformulation + additional_docs, 'Index'))


    oracle_retrieval = []

    for _, topic in topics.iterrows():
        r = 0
        for hit in sorted(oracle_index[__normalize_queries(overlapping_queries[topic['qid']])], key=lambda x: x['relevance'], reverse=True):
            #print(hit)
            r += 1
            oracle_retrieval += [{'qid': topic['qid'], 'query': topic['query'], 'docno': 'ADD_' + hit['doc_id'], 'rank': r, 'score': 100-r, 'run_id': 'oracle'}]

    oracle_retrieval = pd.DataFrame(oracle_retrieval)
    oracle_retrieval = pt.transformer.get_transformer(oracle_retrieval)


    bo1_keyquery_bm25 = oracle_retrieval >> pt.rewrite.Bo1QueryExpansion(index_old, fb_docs=10, fb_terms=20) >> pt.BatchRetrieve(index, wmodel="BM25")

    print('create-run')
    run = bo1_keyquery_bm25(topics)
    persist_and_normalize_run(run, 'bo1-keyquery-bm25', f'results/{lag}-run.txt')

def load_oracle_index(file_name='../oracle-indexing/oracle-index.jsonl.gz', allowed_dataset_ids=('longeval-train-20230513-training', 'longeval-heldout-20230513-training', 'longeval-short-july-20230513-training', 'longeval-long-september-20230513-training', 'longeval-2023-01-20240423-training')):
    entries = pd.read_json(file_name, orient='records', lines=True)
    entries = [i.to_dict() for _, i in entries.iterrows() if i['relevance'] > 0]
    ds_id_to_matches = {i: 0 for i in allowed_dataset_ids}
    ret = {}

    for entry in entries:
        if entry['dataset'] not in allowed_dataset_ids:
            continue
        query = __normalize_queries(entry['query'])
        ds_id_to_matches[entry['dataset']] += 1
        if query not in ret:
            ret[query] = []
        ret[query].append(entry)

    print(f'Found overlaps: {ds_id_to_matches}.')

    return ret

if __name__ == '__main__':
    args = parse_args()
    main(args.lag)
