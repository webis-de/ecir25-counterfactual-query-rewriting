TIRA_CACHE_DIR=.tira/ ./create-indices.py


LD_BIND_NOW=1 TIRA_CACHE_DIR=.tira/ ./keyqueries.py --input-dataset longeval-short-july-20230513-training --output-dir t1

LD_BIND_NOW=1 TIRA_CACHE_DIR=.tira/ ./keyqueries.py --input-dataset longeval-long-september-20230513-training --output-dir t2

LD_BIND_NOW=1 TIRA_CACHE_DIR=.tira/ ./keyqueries.py --input-dataset longeval-2023-01-20240423-training --output-dir t3

LD_BIND_NOW=1 TIRA_CACHE_DIR=.tira/ ./keyqueries.py --input-dataset longeval-2023-06-20240418-training --output-dir t4

LD_BIND_NOW=1 TIRA_CACHE_DIR=.tira/ ./keyqueries.py --input-dataset longeval-2023-08-20240418-training --output-dir t5




./keyqueries/keyqueries.py \
    --oracle-dataset-ids longeval-train-20230513-training longeval-heldout-20230513-training longeval-short-july-20230513-training longeval-long-september-20230513-training \
    --query-document-pairs oracle-indexing/oracle-index.jsonl.gz \
    --input-dataset longeval-2023-01-20240423-training \
    --output-dir foo

tira-run \
    --image time-keyquery:latest \
    --input-dataset ir-benchmarks/cranfield-20230107-training \
    --command '/keyqueries.py --oracle-dataset-ids longeval-train-20230513-training longeval-heldout-20230513-training longeval-short-july-20230513-training longeval-long-september-20230513-training  --query-document-pairs /oracle-index.jsonl.gz --input-dataset $inputDataset --output-dir $outputDir'

tira-run \
    --image time-keyquery:latest \
    --input-dataset ir-benchmarks/cranfield-20230107-training \
    --command '/keyqueries.py --oracle-dataset-ids longeval-train-20230513-training longeval-heldout-20230513-training longeval-short-july-20230513-training longeval-long-september-20230513-training longeval-2023-01-20240423-training --query-document-pairs /oracle-index.jsonl.gz --input-dataset $inputDataset --output-dir $outputDir'
