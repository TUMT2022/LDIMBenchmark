rebuild:
	poetry build
	rm -r tests/dist
	cp -r dist tests/dist
	cd tests && docker build . -t testmethod
