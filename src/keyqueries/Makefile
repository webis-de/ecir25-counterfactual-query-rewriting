docker-bash:
	docker run --rm -ti --entrypoint bash -v ${PWD}/.tira:/root/.tira -v ${PWD}:/app -w /app -p 8080:8080 keyqueries 

build-docker-image:
	docker build -t keyqueries  -f Dockerfile ../../
