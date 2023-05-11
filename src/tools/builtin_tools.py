from langchain.utilities import WikipediaAPIWrapper
from langchain.utilities import PythonREPL
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent
from langchain.tools import Tool
from langchain.chat_models import ChatOpenAI
from dotenv import dotenv_values


def wiki_repl_search_agent(llm, prompt):
    wikipedia = WikipediaAPIWrapper()
    python_repl = PythonREPL()
    search = DuckDuckGoSearchRun()

    tools = [
        Tool(
            name="python repl",
            func=python_repl.run,
            description="useful for when you need to use python to answer a question. You should input python code",
        )
    ]

    wikipedia_tool = Tool(
        name="wikipedia",
        func=wikipedia.run,
        description="Useful for when you need to look up a topic, country or person on wikipedia",
    )

    duckduckgo_tool = Tool(
        name="DuckDuckGo Search",
        func=search.run,
        description="Useful for when you need to do a search on the internet to find information that another tool can't find. be specific with your input.",
    )

    tools.append(duckduckgo_tool)
    tools.append(wikipedia_tool)

    zero_shot_agent = initialize_agent(
        agent="zero-shot-react-description",
        tools=tools,
        llm=llm,
        verbose=True,
        max_iterations=3,
    )

    zero_shot_agent.run(prompt)


if __name__ == "__main__":
    token_config = dotenv_values("../../.env")

    chat = ChatOpenAI(
        openai_api_key=token_config["OPENAI_API_KEY"],
        model_name="gpt-3.5-turbo",
        verbose=True,
        temperature=0,
        max_tokens=2000,
        frequency_penalty=0,
        presence_penalty=0,
        top_p=1.0,
    )
    wiki_repl_search_agent(llm=chat, prompt="vad heter finska riksarkivet på finska?")
