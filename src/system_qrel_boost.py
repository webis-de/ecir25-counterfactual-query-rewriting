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


def extend_run_full(run_path, sub_collection):
    # Connect to the SQLite database
    conn = sqlite3.connect(BASE_PATH + "/database.db")

    print(">>> Loaded run")
    run = pd.read_csv(
        run_path,
        sep=" ",
        names=["queryid", "0", "docid", "relevance", "score", "run"],
        index_col=False,
    )

    # Load doc map
    print(">>> Load doc map")
    docids = run["docid"].unique()

    def chunker(seq, size):
        return (seq[pos : pos + size] for pos in range(0, len(seq), size))

    query_base = "SELECT docid, url FROM Document WHERE docid IN ({})"

    results = []
    for chunk in chunker(docids, 100_000):
        placeholders = ", ".join(["?"] * len(chunk))
        query = query_base.format(placeholders)
        result = pd.read_sql_query(query, conn, params=chunk)
        results.append(result)

    docmapper = pd.concat(results, ignore_index=True)

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

    # Querymap
    print(">>> Load query map")
    queryids = run["queryid"].unique()
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

    # Merge
    print(">>> Extend run")
    run_docids_extended = run.merge(
        docid_map.add_prefix("docid_"),
        left_on="docid",
        right_on=f"docid_{sub_collection}",
        how="left",
    )
    run_extended = run_docids_extended.merge(
        query_text_fr_map.add_prefix("queryid_"),
        left_on="queryid",
        right_on=f"queryid_{sub_collection}",
        how="left",
    )

    run_extended.to_csv(
        run_path[:-5] + "_extended." + sub_collection, sep=" ", index=False
    )


def extend_with_qrels(run_name, history, train_docids):
    conn = sqlite3.connect(BASE_PATH + "/database.db")
    query = "SELECT * FROM qrel"
    qrels = pd.read_sql_query(query, conn)

    run = pd.read_csv(RESULTS_PATH + run_name, sep=" ")
    qrels_map = qrels[["docid", "relevance"]].set_index("docid").to_dict()["relevance"]

    def get_qrel(row, subcollection):
        query_id = row[f"queryid_{subcollection}"]
        doc_id = row[f"docid_{subcollection}"]
        if doc_id in train_docids:
            if isinstance(query_id, str) and isinstance(doc_id, str):
                return qrels_map.get(
                    row[f"queryid_{subcollection}"] + row[f"docid_{subcollection}"],
                    None,
                )
            else:
                return None
        else:
            return None

    for subcollection in history:
        run[f"qrel_{subcollection}"] = run.apply(
            get_qrel, subcollection=subcollection, axis=1
        )

    return run


def qrel_boost(run, history, _lambda=0.5, mu=2):
    # min max normalization per topic
    run["score"] = run.groupby("queryid")["score"].transform(lambda x: x / x.max())

    for subcollection in history:
        # Relevant
        run.loc[run[f"qrel_{subcollection}"] == 1, "score"] = (
            run.loc[run[f"qrel_{subcollection}"] > 0, "score"] * _lambda**2
        )
        run.loc[run[f"qrel_{subcollection}"] > 0, "score"] = (
            run.loc[run[f"qrel_{subcollection}"] > 0, "score"] * (_lambda**2) * mu
        )

        # All Not Relevant
        run.loc[
            (run[f"qrel_{subcollection}"] == 0) | (run[f"qrel_{subcollection}"].isna()),
            "score",
        ] = (
            run.loc[
                (run[f"qrel_{subcollection}"] == 0)
                | (run[f"qrel_{subcollection}"].isna()),
                "score",
            ]
            * (1 - _lambda) ** 2
        )

    run = (
        run.sort_values(["queryid", "score"], ascending=False)
        .groupby("queryid")
        .head(1000)
    )
    run["rank"] = run.groupby("queryid")["score"].rank(ascending=False).astype(int)

    run = run[["queryid", "0", "docid", "score", "rank", "run"]].rename(
        columns={"queryid": "qid", "docid": "docno"}
    )

    return run


def system_qrel_boost(
    train_docids, sub_collection, topics, index, history, fold_no, _lambda=0.5, mu=2
):
    print(">>> Use history:", history)
    run_name = f"/BM25+qrel_boost_{sub_collection}_F{fold_no}_H{''.join(history)}_l{_lambda}_m{mu}"

    # BM25 top 1500 as baseline, retrieve more results to filter
    BM25 = pt.BatchRetrieve(
        index, wmodel="BM25", verbose=True, num_results=1500
    )  
    pt.io.write_results(BM25(topics), RESULTS_PATH + run_name + "-long")
    
    extend_run_full(RESULTS_PATH + run_name + "-long", sub_collection)

    # we need the topic sub-collection to merge the qrels
    history_complete = history + [sub_collection]
    run = extend_with_qrels(
        run_name + "_extended." + sub_collection, history_complete, train_docids
    )

    run = qrel_boost(run, history, _lambda, mu)

    pt.io.write_results(run, RESULTS_PATH + run_name)
    
    #cleanup
    os.remove(RESULTS_PATH + run_name + "-long")
    os.remove(RESULTS_PATH + run_name + "_extended." + sub_collection)

    return pt.io.read_results(RESULTS_PATH + run_name)
