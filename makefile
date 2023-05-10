create_env:
	python -m venv venv

activate: 
	source langchain/bin/activate

install: 
	pip install -r requirements.txt