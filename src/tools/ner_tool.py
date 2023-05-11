from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from langchain.tools import BaseTool


class NerTool(BaseTool):
    name = "Name entity recognition"
    description = "use this tool when given the directions to extract entities of interest from invoices automatically using Named Entity Recognition (NER) models"

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        self.model = AutoModelForTokenClassification.from_pretrained(
            "dslim/bert-base-NER"
        )

    def _run(self, text: str):
        ner_pipe = pipeline("ner", model=self.model, tokenizer=self.tokenizer)

        ner_results = ner_pipe(text)

        return ner_results

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


tools = [NerTool()]
# add to as tool
# TODO: return tools
# https://www.youtube.com/watch?v=q-HNphrWsDE
# should i use abstract claasses ? https://gradio.app/gradio-and-llm-agents/

if __name__ == "__main__":
    pass
