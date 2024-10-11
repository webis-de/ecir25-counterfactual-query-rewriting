import pyterrier as pt
import yaml
import os
import sqlite3
import pandas as pd
from tqdm import tqdm

if not pt.java.started():
    pt.java.add_package("com.github.terrierteam", "terrier-prf", "-SNAPSHOT")


BASE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
RESULTS_PATH = BASE_PATH + "/results"

def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))

def extend_with_doc_ids(run, sub_collection):
    conn = sqlite3.connect(BASE_PATH + "/database.db")
    docids = run["docno"].unique()
    
    # create patch for docid to url mapping for docs in run
    query_base = "SELECT docid, url FROM Document WHERE docid IN ({})"

    results = []
    for chunk in chunker(docids, 100_000):
        placeholders = ", ".join(["?"] * len(chunk))
        query = query_base.format(placeholders)
        result = pd.read_sql_query(query, conn, params=chunk)
        results.append(result)
        
    docmapper = pd.concat(results, ignore_index=True)
    
    # create patch for docid to subcollection mapping for docs in run
    query_base = "SELECT docid, url, sub_collection FROM Document WHERE url IN ({})"

    results = []
    for chunk in tqdm(
        chunker(docmapper["url"].unique(), 10_000), total=len(docmapper) / 10_000
        ):
        placeholders = ", ".join(["?"] * len(chunk))
        query = query_base.format(placeholders)
        params = list(chunk)
        result = pd.read_sql_query(query, conn, params=params)
        results.append(result)

    docid_map = pd.concat(results, ignore_index=True)
    docid_map = docid_map.pivot(index="url", columns="sub_collection", values="docid")
    
    
    run_docids_extended = run.merge(
        docid_map.add_prefix("docno_"),
        left_on="docno",
        right_on=f"docno_{sub_collection}",
        how="left",
    )
    
    return run_docids_extended


def extend_with_query_ids(run, sub_collection):
    conn = sqlite3.connect(BASE_PATH + "/database.db")

    queryids = run["qid"].unique()
    query = "SELECT queryid, text_fr FROM Topic WHERE queryid IN (%s);" % ",".join(
        "?" * len(queryids)
    )
    querymapper = pd.read_sql_query(query, conn, params=queryids)

    query = (
        "SELECT queryid, text_fr, text_en, sub_collection FROM Topic WHERE text_fr IN (%s);"
        % ",".join("?" * len(querymapper["text_fr"].unique()))
    )
    query_text_fr_map = pd.read_sql_query(
        query, conn, params=querymapper["text_fr"].unique()
    )
    query_text_fr_map = query_text_fr_map.pivot(
        index=["text_fr", "text_en"], columns="sub_collection", values="queryid"
    )
    
        
    run_extended = run.merge(
        query_text_fr_map.add_prefix("qid_"),
        left_on="qid",
        right_on=f"qid_{sub_collection}",
        how="left",
    )
    
    return run_extended


def extend_with_qrels(run, history, sub_collection, train_docids=None):
    conn = sqlite3.connect(BASE_PATH + "/database.db")
       
    query = "SELECT * FROM qrel"
    qrels_all = pd.read_sql_query(query, conn)
    
    
    qrels_all["key"] = qrels_all["queryid"] + qrels_all["docid"]
    qrels_map = qrels_all[["key", "relevance"]].set_index("key").to_dict()["relevance"]
    
    
    def get_qrel(row, h, train_docids):
        query_id = row[f"qid_{h}"]
        docid_old = row[f"docno_{h}"]
        doc_id_now = row[f"docno_{sub_collection}"]
        
        if doc_id_now not in train_docids:
            return None
        
        elif isinstance(query_id, str) and isinstance(docid_old, str):
            return qrels_map.get(
                row[f"qid_{h}"] + row[f"docno_{h}"],
                None,
            )
        else:
            return None

    for h in history:
        run[f"qrel_{h}"] = run.progress_apply(
            get_qrel, h=h, train_docids=train_docids, axis=1
        )
    return run
    
    
def qrel_boost(run_in, history, _lambda=0.5, mu=2):
        run = run_in.copy()
        run["score"] = run.groupby("qid")["score"].transform(lambda x: x / x.max())

        for h in history:
            run.loc[run[f"qrel_{h}"] == 1, "score"] *= _lambda ** 2
            run.loc[run[f"qrel_{h}"] == 2, "score"] *= (_lambda ** 2) * mu
            run.loc[(run[f"qrel_{h}"] == 0) | (run[f"qrel_{h}"].isna()), "score"] *= (1 - _lambda) ** 2

        run = run.sort_values(["qid", "score"], ascending=False).groupby("qid").head(1000)
        run["rank"] = run.groupby("qid")["score"].rank(ascending=False).astype(int)
        return run[["qid", "docno", "score", "rank"]]



def system_qrel_boost(
    train_docids, sub_collection, topics, index, history, fold_no, _lambda=0.5, mu=2
):
    print(">>> Use history:", history)
    run_name = f"/BM25+qrel_boost_{sub_collection}_F{fold_no}_H{''.join(history)}_l{_lambda}_m{mu}"


    # BM25 top 1500 as baseline, retrieve more results to filter
    BM25 = pt.BatchRetrieve(
        index, wmodel="BM25", verbose=True, num_results=1500
    )
    run = BM25(topics)
    
    if not train_docids:
        train_docids = set(run["docno"].unique().tolist())
    
    print(">>> Extend with query ids")
    run = extend_with_query_ids(run, sub_collection)
    
    print(">>> Extend with doc ids")
    run = extend_with_doc_ids(run, sub_collection)
    
    print(">>> Extend with qrels")
    run = extend_with_qrels(run, history, sub_collection, train_docids)
    
    print(">>> qrel_boost")
    run = qrel_boost(run, history, _lambda, mu)

    pt.io.write_results(run, RESULTS_PATH + run_name)

    return run
