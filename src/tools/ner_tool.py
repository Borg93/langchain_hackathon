from langchain.tools import BaseTool
from dotenv import dotenv_values
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
import requests


class NerTool(BaseTool):
    name = "Name entity recognition"
    description = "Use this tool when given the directions to extract entities of interest from invoices automatically using Named Entity Recognition (NER) models"

    def _run(self, text: str):
        token_config = dotenv_values("../../.env")

        API_URL = "https://api-inference.huggingface.co/models/dslim/bert-base-NER"
        headers = {"Authorization": f"Bearer {token_config['HF_TOKEN']}"}

        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()

        output = query(
            {
                "inputs": text,
            }
        )

        return output

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


def ner_tool():
    tools = [NerTool()]
    return tools


if __name__ == "__main__":
    token_config = dotenv_values("../../.env")

    chat = ChatOpenAI(
        openai_api_key=token_config["OPENAI_API_KEY"],
        model_name="gpt-3.5-turbo",
        verbose=True,
        temperature=0,
        max_tokens=2000,
        # frequency_penalty=0,
        # presence_penalty=0,
        # top_p=1.0,
    )
    ner_tool = [NerTool()]

    ner_agent = initialize_agent(
        agent="zero-shot-react-description",
        tools=ner_tool,
        llm=chat,
        verbose=True,
        max_iterations=3,
    )

    ner_agent.run(
        "Can you run a Ner model on this input: 'My name is Clara and I live in Berkeley, California.'"
    )
