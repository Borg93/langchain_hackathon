from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from dotenv import dotenv_values

token_config = dotenv_values("../.env")

repo_id = "databricks/dolly-v2-3b"

llm = HuggingFaceHub(
    repo_id=repo_id,
    model_kwargs={"temperature": 0, "max_length": 64},
    huggingfacehub_api_token=token_config["HF_TOKEN"],
)

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

question = "Who won the FIFA World Cup in the year 1994? "


# Reuse the prompt and question from above.
llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
print(llm_chain.run(question))
