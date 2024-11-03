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
import json
import os


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
    parser.add_argument('--fb-terms', type=list, help='fb_terms passed to pyterrier', nargs='+', default=[10], required=False)
    parser.add_argument('--w-models', type=list, help='weighting models passed to pyterrier', nargs='+', default=['BM25', 'DirichletLM'], required=False)
    parser.add_argument('--fb-docs', type=list, help='fb_docs passed to pyterrier', nargs='+', default=[5], required=False)
    parser.add_argument('--first-stage-top-k', type=int, help='top-k documents used for query reformulation', default=900, required=False)
    return parser.parse_args()


def get_overlapping_queries(pt_dataset, output_dir):
    print('Look for overlapping queries...')
    splits = json.load(gzip.open('../../data/expansion-documents.json.gz', 'rt'))
    queries = set()
    for query in splits[output_dir]:
        queries.add(query)

    overlapping_queries = {i.query_id: i.default_text() for i in pt_dataset.irds_ref().queries_iter()}
    overlapping_queries = {k: v for k, v in overlapping_queries.items() if k in queries}

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
    df_file = f'{output_dir}/queries/{wmodel}_{query["qid"]}.jsonl.gz'
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
            results = pt.Experiment([retriever], topics, qrels, eval_metrics=['ndcg_cut_10', NumRet(), 'recall_10'])
            assert len(results) == 1
            score = results.iloc[0]['ndcg_cut_10']
            num_ret = results.iloc[0]['NumRet']
            if num_ret < 15:
               continue
            df += [{'topic': query, 'candidate': candidate, 'ndcg_cut_10': score, 'num_ret': num_ret, 'recall@10': 'recall_10', 'len(qrels)': len(qrels)}]

        df = pd.DataFrame(df)
        df = df.sort_values('ndcg_cut_10', ascending=False).head(50)
        df.to_json(df_file, lines=True, orient='records')

    return pd.read_json(df_file, lines=True).iloc[0]['candidate']

def get_reformulation_index(output_dir):
    return pt.IndexRef.of(os.path.abspath(f"indices/{split}/"))

def get_oracle_retrieval_results(topics, oracle_index, overlapping_queries):
    ret = []

    qid_to_query = {}
    for _, topic in topics.iterrows():
        qid_to_query[topic['qid']] = topic['query']

    splits = json.load(gzip.open('../../data/expansion-documents.json.gz', 'rt'))
    queries = set()
    ret = []

    for query, docs_for_query in splits[output_dir].items:
        r = 0
        for hit in docs_for_query:
            r += 1
            ret += [{'qid': query, 'query': qid_to_query['query'], 'docno': hit, 'rank': r, 'score': 100-r, 'run_id': 'oracle', 'relevance': 1}]

    ret = pd.DataFrame(ret)
    return pt.transformer.get_transformer(ret)

def run_foo(index, reformulation_index, weighting_models, fb_terms, fb_docs, out_dir, oracle_retrieval_results, topics):
    
    for wmodel in weighting_models:
        for fb_term in fb_terms:
            for fb_doc in fb_docs:
                print(f'Run keyqueries on {wmodel} {fb_term} {fb_doc}')
                rm3_query_terms = oracle_retrieval_results >> pt.rewrite.RM3(reformulation_index, fb_docs=fb_doc, fb_terms=fb_term)
                rm3_query_terms = rm3_query_terms(topics)
                
                reformulated_topics = []
                for _, i in tqdm(list(rm3_query_terms.iterrows()), 'queries'):
                    i = i.to_dict()
                    reformulated_topic = find_best_expansion(oracle_retrieval_results, i, wmodel, reformulation_index, out_dir)
                    i['query'] = reformulated_topic
                    reformulated_topics += [i]

                reformulated_topics = pd.DataFrame(reformulated_topics)
                retriever = pt.BatchRetrieve(index, wmodel=wmodel)
                retrieval_results = retriever(reformulated_topics)

                for split, docs_to_skip in all_splits(out_dir).items():
                    results_for_split = []
                    for _, i in retrieval_results.iterrows():
                        if i['docno'] in docs_to_skip:
                            continue
                        results_for_split += [i.to_dict()]

                    results_for_split = pd.DataFrame(results_for_split)
                    results_for_split['Q0'] = 0
                    results_for_split['system'] = f'{wmodel}-keyquery'
                    results_for_split = results_for_split.copy().sort_values(["qid", "score", "docno"], ascending=[True, False, False]).reset_index()

                    results_for_split[["qid", "Q0", "docno", "rank", "score", "system"]].to_csv(f'{out_dir}/{wmodel}-split-{split}.run.gz', sep=" ", header=False, index=False)

if __name__ == '__main__':
    args = parse_args()
    ensure_pyterrier_is_loaded()
    import pyterrier as pt

    tira = Client('https://api.tira.io')

    pt_dataset = pt.get_dataset(f'irds:ir-benchmarks/{args.input_dataset}')
    overlapping_queries = get_overlapping_queries(pt_dataset, oracle_index)
    topics = get_overlapping_topics(pt_dataset, overlapping_queries)
    oracle_retrieval_results = get_oracle_retrieval_results(topics, oracle_index, overlapping_queries)

    bm25_raw = tira.pt.from_submission('ir-benchmarks/tira-ir-starter/BM25 (tira-ir-starter-pyterrier)', args.input_dataset) % args.first_stage_top_k
    index = tira.pt.index('ir-benchmarks/tira-ir-starter/Index (tira-ir-starter-pyterrier)', args.input_dataset)
    reformulation_index = get_reformulation_index(oracle_index, bm25_raw, topics, pt_dataset)

    run_foo(index, reformulation_index, args.w_models, args.fb_terms, args.fb_docs, args.output_dir, oracle_retrieval_results, topics)
