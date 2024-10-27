#docker build -t time-keyquery -f keyqueries/Dockerfile .
# A prepared image with python3.10, java 11, ir_datasets, tira, and PyTerrier installed 
FROM webis/ir-lab-wise-2023:0.0.4

# Update the tira command to use the latest version
RUN pip3 uninstall -y tira \
	&& pip3 install git+https://github.com/tira-io/tira.git@development#\&subdirectory=python-client

ADD src/keyqueries/keyqueries.py data/input-data-for-keyqueries.jsonl.gz /

ENTRYPOINT [ "/keyqueries.py", "--input-dataset",  "$inputDataset", "--output-dir", "$outputDir" ]

