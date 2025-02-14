from create_index import load_data, load_folds, create_index
import pyterrier as pt
from system_qrel_boost import system_qrel_boost
from system_relevance_feedback import system_relevance_feedback
import os

BASE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
RESULTS_PATH = BASE_PATH + "/results"

if not pt.java.started():
    pt.java.add_package("com.github.terrierteam", "terrier-prf", "-SNAPSHOT")


def main():
    folds = load_folds()
    # we skip t0 as it is the base collection and has no history
    sub_collections = [
        "t1",
        "t2",
        "t3",
        "t4",
        "t5",
    ]

    for sub_collection in sub_collections:
        # Load data for sub-collection
        topics, qrels = load_data(sub_collection)

        # for pyterrier
        for fold_no in range(0, len(folds[sub_collection])):
            fold_no = str(fold_no)
            train_docids = folds[sub_collection][fold_no]["train"]
            test_docids = folds[sub_collection][fold_no]["test"]


            # Create index
            test_index = create_index(sub_collection, fold_no, test_docids)
            test_index = BASE_PATH + f"/index/{sub_collection}_{fold_no}"
            history = ["t0"] + sub_collections[: sub_collections.index(sub_collection)]

            
            # System: BM25
            BM25 = pt.terrier.Retriever(test_index, wmodel="BM25", verbose=True)
            bm25_run = BM25(topics)
            pt.io.write_results(bm25_run, os.path.join(RESULTS_PATH, f"BM25_{sub_collection}_F{fold_no}"))


            # System: BM25+Bo1
            bo1 = pt.rewrite.Bo1QueryExpansion(test_index)
            bm25_bo1_pipe = pt.Transformer.from_df(bm25_run) >> bo1 >> BM25
            bm25_bo1_run = bm25_bo1_pipe(topics)
            pt.io.write_results(bm25_bo1_run, os.path.join(RESULTS_PATH, f"BM25+Bo1_{sub_collection}_F{fold_no}"))


            # System: BM25+RM3
            rm3 = pt.rewrite.RM3(test_index)
            bm25_rm3_pipe = pt.Transformer.from_df(bm25_run) >> rm3 >> BM25
            bm25_rm3_run = bm25_rm3_pipe(topics)
            pt.io.write_results(bm25_rm3_run, os.path.join(RESULTS_PATH, f"BM25+RM3_{sub_collection}_F{fold_no}"))


            # System: qrel_boost
            run_qrel_boost = system_qrel_boost(
                train_docids,
                sub_collection,
                topics,
                test_index,
                history=history,
                fold_no=fold_no,
                _lambda=0.7,
                mu=2,
            )


            # System: RF
            run_relevance_feedback = system_relevance_feedback(
                train_docids,
                sub_collection,
                topics,
                test_index,
                history=history,
                fold_no=fold_no,
            )
            
            # Evaluate
            print(">>> Evaluate")
            qrels = qrels[~qrels["docno"].isin(train_docids)]

            res = pt.Experiment(
                # [bm25_run, run_qrel_boost, run_relevance_feedback],
                [bm25_run, bm25_bo1_run, bm25_rm3_run],
                topics,
                qrels,
                eval_metrics=["ndcg", "bpref", "map", "ndcg_cut.10", "P.10", "recall.100"],
            )

            print(res)


if __name__ == "__main__":
    main()