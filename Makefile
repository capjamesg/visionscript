tests:
	python3 tests/test.py
build:
	python3 -m pip install --upgrade build
	pip3 -m build
install:
	pip3 install .
deploy:
	pip3 install twine
	python3 -m twine upload --repository pypi dist/*
clean:
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
lint:
	black visionscript/*
	isort visionscript/*