#!/usr/bin/env python3
from keyqueries import *

timestamp_to_ir_datasets_id = {
    't1': 'longeval-short-july-20230513-training',
    't2': 'longeval-long-september-20230513-training',
    't3': 'longeval-2023-01-20240423-training',
    't4': 'longeval-2023-06-20240418-training',
    't5': 'longeval-2023-08-20240418-training',
}

def persist_runs(system_name, out_dir, input_dataset, topics):
    ret = tira.pt.from_submission(system_name, input_dataset)
    system_name = system_name.split('/')[-1].split()[0]
    retrieval_results = ret(topics)

    for split, docs_to_skip in all_splits(out_dir).items():
        results_for_split = []
        for _, i in retrieval_results.iterrows():
            if i['docno'] in docs_to_skip:
                continue
            results_for_split += [i.to_dict()]

        results_for_split = pd.DataFrame(results_for_split)
        results_for_split['Q0'] = 0
        results_for_split['system'] = system_name
        results_for_split = results_for_split.copy().sort_values(["qid", "score", "docno"], ascending=[True, False, False]).reset_index()

        results_for_split[["qid", "Q0", "docno", "rank", "score", "system"]].to_csv(f'../../data/results_baseline/{system_name}-{out_dir}-split-{split}.run.gz', sep=" ", header=False, index=False)


if __name__ == '__main__':
    ensure_pyterrier_is_loaded()
    import pyterrier as pt

    tira = Client('https://api.tira.io')
    for out_dir, input_dataset in tqdm(timestamp_to_ir_datasets_id.items()):
        pt_dataset = pt.get_dataset(f'irds:ir-benchmarks/{input_dataset}')
        topics = pt_dataset.get_topics('text')

        persist_runs('ir-benchmarks/fschlatt/castorini-list-in-t5-150', out_dir, input_dataset, topics)

        persist_runs('ir-benchmarks/tira-ir-starter/MonoT5 Base (tira-ir-starter-gygaggle)', out_dir, input_dataset, topics)

        persist_runs('ir-benchmarks/tira-ir-starter/ColBERT Re-Rank (tira-ir-starter-pyterrier)', out_dir, input_dataset, topics)
        
