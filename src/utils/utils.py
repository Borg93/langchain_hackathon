from langchain.prompts import load_prompt
from langchain.callbacks import get_openai_callback


def load_prompt_json(path):
    # prompt.save("file_name.json")
    prompt = load_prompt(path)
    return prompt


def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f"Spent a total of {cb.total_tokens} tokens")

    return result
