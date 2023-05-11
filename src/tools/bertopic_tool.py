from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from langchain.tools import BaseTool


class BertTopic(BaseTool):
    name = ""
    description = ""

    def __init__(self):
        pass

    def _run(self, text: str):
        pass

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
