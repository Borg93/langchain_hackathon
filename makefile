create_env:
	python -m venv langchain

activate: 
	source langchain/bin/activate

install: 
	pip install -r requirements.txt