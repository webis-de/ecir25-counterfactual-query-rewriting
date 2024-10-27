#!/usr/bin/env python3
import pathlib
import pandas as pd
import argparse
from tira.third_party_integrations import ensure_pyterrier_is_loaded
from tira.rest_api_client import Client
from tqdm import tqdm
from random import randint
from pathlib import Path
from ir_measures import NumRet
import os


def __normalize_queries(q):
    return q.lower().strip()

def load_oracle_index(file_name, allowed_dataset_ids):
    print(f'Load oracle index from {file_name}')
    entries = pd.read_json(file_name, orient='records', lines=True)
    entries = [i.to_dict() for _, i in entries.iterrows() if i['relevance'] > 0]
    ret = {}
    dataset_counts = {i: 0 for i in allowed_dataset_ids}

    for entry in entries:
        if entry['dataset'] not in allowed_dataset_ids:
            continue
        query = __normalize_queries(entry['query'])
        if query not in ret:
            ret[query] = []
        ret[query].append(entry)
        dataset_counts[entry['dataset']] += 1

    print(f'Done. Loaded entries from the oracle {dataset_counts}.')

    return ret

def parse_args():
    parser = argparse.ArgumentParser(description='Construct neighbors')
    parser.add_argument('--input-dataset', type=str, help='Input file', default='cranfield-20230107-training')
    parser.add_argument('--output-dir', type=str, help='Output file', required=True)
    parser.add_argument('--query-document-pairs', type=str, help='Output file', default=str((pathlib.Path(__file__).parent.parent.parent.resolve() / 'data' / 'input-data-for-keyqueries.jsonl.gz').absolute()), required=False)
    parser.add_argument('--fb-terms', type=list, help='fb_terms passed to pyterrier', nargs='+', default=[10], required=False)
    parser.add_argument('--w-models', type=list, help='weighting models passed to pyterrier', nargs='+', default=['BM25', 'DirichletLM'], required=False)
    parser.add_argument('--oracle-dataset-ids', type=str, help='allowed dataset ids', nargs='+', required=True)
    parser.add_argument('--fb-docs', type=list, help='fb_docs passed to pyterrier', nargs='+', default=[5], required=False)
    parser.add_argument('--first-stage-top-k', type=int, help='top-k documents used for query reformulation', default=900, required=False)
    return parser.parse_args()


def get_overlapping_queries(pt_dataset, oracle_index):
    print('Look for overlapping queries...')
    overlapping_queries = {i.query_id: i.default_text() for i in pt_dataset.irds_ref().queries_iter()}
    overlapping_queries = {k: v for k, v in overlapping_queries.items() if __normalize_queries(v) in oracle_index}

    print(f'Done. Found {len(overlapping_queries)} overlapping queries.')
    return overlapping_queries


def get_overlapping_topics(pt_dataset, overlapping_queries):
    print('Select overlapping topics...')
    topics = pt_dataset.get_topics('text')
    topics = topics[topics['qid'].isin(overlapping_queries.keys())]

    print(f'Done. Found {len(topics)} overlapping topics.')
    return topics

def all_expanded_queries(query, min_length):
    from itertools import chain, combinations
    def powerset(iterable):
        s = list(iterable)
        ret = chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
        return ['applypipeline:off ' + (' '.join(i)) for i in ret if len(i) >= min_length and len(i) < 7]

    terms = [i for i in query['query'].split() if '^' in i]
    mandatory_terms = [i for i in terms if float(i.split('^')[1]) > 0.1]
    ret_1 = list(powerset(terms))
    ret = []
    for i in ret_1:
        add = True
        for mandatory_term in mandatory_terms:
            if mandatory_term not in i:
                add = False
        if add:
            ret += [i]
    return ret


def find_best_expansion(oracle_retrieval_results, query, wmodel, index, output_dir):
    df_file = f'{output_dir}/{wmodel}_{query["qid"]}.jsonl.gz'
    if not os.path.isfile(df_file):
        qrels = []
        for _, i in oracle_retrieval_results(pd.DataFrame([query])).iterrows():
            if str(i['qid']) == str(query['qid']):
                qrels += [{'qid': '1', 'docno': i['docno'], 'relevance': i['relevance']}]

        qrels = pd.DataFrame(qrels)

        assert len(qrels) > 0

        df = []
        retriever = pt.BatchRetrieve(index, wmodel=wmodel)
        for candidate in all_expanded_queries(query, 3):
            topics = pd.DataFrame([{'qid': '1', 'query': candidate}])
            results = pt.Experiment([retriever], topics, qrels, eval_metrics=['ndcg_cut_10', NumRet()])
            assert len(results) == 1
            score = results.iloc[0]['ndcg_cut_10']
            num_ret = results.iloc[0]['NumRet']
            if num_ret < 100:
               continue
            df += [{'topic': query, 'candidate': candidate, 'ndcg_cut_10': score, 'num_ret': num_ret}]

        df = pd.DataFrame(df)
        df = df.sort_values('ndcg_cut_10', ascending=False).head(50)
        df.to_json(df_file, lines=True, orient='records')

    return pd.read_json(df_file, lines=True).iloc[0]['candidate']

def build_reformulation_index(oracle_index, bm25_raw, topics, pt_dataset):
    additional_docs = {}

    for i in oracle_index:
        for j in oracle_index[i]:
            additional_docs[j['doc_id']] = j['doc']

    additional_docs = [{'docno': 'ADD_' + k, 'text': v} for k, v in additional_docs.items()]
    print(f'Have {len(additional_docs)} documents from the oracle.')

    doc_ids = []

    for _, i in bm25_raw(topics).iterrows():
        if i['qid'] in overlapping_queries:
            doc_ids.append(i['docno'])
    
    doc_ids = set(doc_ids)

    docs_for_reformulation = []

    for i in tqdm(pt_dataset.get_corpus_iter()):
        if i['docno'] not in doc_ids:
            continue
        docs_for_reformulation += [i]
    
    print(f'Have {len(additional_docs)} documents for reformulation.')

    iter_indexer = pt.IterDictIndexer(f"/tmp/{randint(0,1000000000)}/reformulation-index", meta={'docno': 50, 'text': 4096}, overwrite=True)
    return iter_indexer.index(tqdm(docs_for_reformulation + additional_docs, 'Index'))

def get_oracle_retrieval_results(topics, oracle_index, overlapping_queries):
    ret = []

    for _, topic in topics.iterrows():
        r = 0
        for hit in sorted(oracle_index[__normalize_queries(overlapping_queries[topic['qid']])], key=lambda x: x['relevance'], reverse=True):
            r += 1
            ret += [{'qid': topic['qid'], 'query': topic['query'], 'docno': 'ADD_' + hit['doc_id'], 'rank': r, 'score': 100-r, 'run_id': 'oracle', 'relevance': hit['relevance']}]

    ret = pd.DataFrame(ret)
    return pt.transformer.get_transformer(ret)

def run_foo(index, reformulation_index, weighting_models, fb_terms, fb_docs, out_dir, oracle_retrieval_results, topics):
    
    for wmodel in weighting_models:
        for fb_term in fb_terms:
            for fb_doc in fb_docs:
                print(f'Run RM3 on {wmodel} {fb_term} {fb_doc}')
                rm3_query_terms = oracle_retrieval_results >> pt.rewrite.RM3(reformulation_index, fb_docs=fb_doc, fb_terms=fb_term)
                rm3_query_terms = rm3_query_terms(topics)
                
                reformulated_topics = []
                for _, i in tqdm(list(rm3_query_terms.iterrows()), 'queries'):
                    i = i.to_dict()
                    reformulated_topic = find_best_expansion(oracle_retrieval_results, i, wmodel, reformulation_index, out_dir)
                    i['query'] = reformulated_topic
                    reformulated_topics += [i]

                retriever = pt.BatchRetrieve(index, wmodel=wmodel)

                rm3_keyquery_bm25(pd.DataFrame(reformulated_topics)).to_json(f'{out_dir}/{wmodel}_{fb_term}_{fb_doc}.jsonl.gz', index=False, lines=True, orient='records')

if __name__ == '__main__':
    args = parse_args()
    ensure_pyterrier_is_loaded()
    import pyterrier as pt

    tira = Client('https://api.tira.io')

    pt_dataset = pt.get_dataset(f'irds:ir-benchmarks/{args.input_dataset}')
    oracle_index = load_oracle_index(args.query_document_pairs, args.oracle_dataset_ids)
    overlapping_queries = get_overlapping_queries(pt_dataset, oracle_index)
    topics = get_overlapping_topics(pt_dataset, overlapping_queries)
    oracle_retrieval_results = get_oracle_retrieval_results(topics, oracle_index, overlapping_queries)

    bm25_raw = tira.pt.from_submission('ir-benchmarks/tira-ir-starter/BM25 (tira-ir-starter-pyterrier)', args.input_dataset) % args.first_stage_top_k
    index = tira.pt.index('ir-benchmarks/tira-ir-starter/Index (tira-ir-starter-pyterrier)', args.input_dataset)
    reformulation_index = build_reformulation_index(oracle_index, bm25_raw, topics, pt_dataset)

    run_foo(index, reformulation_index, args.w_models, args.fb_terms, args.fb_docs, args.output_dir, oracle_retrieval_results, topics)
