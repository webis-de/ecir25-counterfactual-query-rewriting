### Document Similarity Analysis

We use the CopyCat S3 implementation in default configuration (as this is already focused on answering the question if two documents contain the same content from IR/novelty principle perspective).

- Step 1: Create document-groups.jsonl: Run `./prepare-document-clusters-for-near-duplicate-analysis.py`
- Step 2: Run CopyCat: Run `./run-document-similarity-scoring.sh`

