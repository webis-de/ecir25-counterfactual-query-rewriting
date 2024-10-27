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