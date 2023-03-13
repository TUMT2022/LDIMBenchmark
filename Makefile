rebuild:
	poetry build
	rm -r tests/dist
	cp -r dist tests/dist
	cd tests && docker build . -t testmethod

bdocker:
	poetry build
	rm -rf docker/ldimbenchmark/dist
	mkdir docker/ldimbenchmark/dist
	cp -rf dist/ldimbenchmark-0.0.0-py3-none-any.whl docker/ldimbenchmark/dist/ldimbenchmark-0.0.0-py3-none-any.whl
	cd docker && ./build.sh

dockerimage:
	VERSION=0.1.31 docker build .