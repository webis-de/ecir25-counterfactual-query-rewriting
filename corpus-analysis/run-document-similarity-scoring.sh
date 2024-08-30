#!/usr/bin/bash

echo "Compile the code"

docker run -v ${PWD}:/app -w /app --rm -ti webis/chatnoir-copycat:1.0-jupyter \
	javac -cp /copycat/copycat-cli-1.0-SNAPSHOT-jar-with-dependencies.jar DocumentSimilarityScoring.java

echo "Run the Document Similarity Scoring:"

docker run -v ${PWD}:/app -w /app --rm -ti webis/chatnoir-copycat:1.0-jupyter \
	java -cp /copycat/copycat-cli-1.0-SNAPSHOT-jar-with-dependencies.jar:. DocumentSimilarityScoring --input document-groups.jsonl --output document-similarities.jsonl
