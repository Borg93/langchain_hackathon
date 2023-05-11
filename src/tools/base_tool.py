# Import things that are needed generically
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from pydantic import BaseModel, Field
from langchain.tools import DuckDuckGoSearchRun


search = DuckDuckGoSearchRun()


# Load the tool configs that are needed.
def math_basetool():
    llm_math_chain = LLMMathChain(llm=llm, verbose=True)
    tools = [
        Tool.from_function(
            func=search.run,
            name="Search",
            description="useful for when you need to answer questions about current events"
            # coroutine= ... <- you can specify an async method if desired as well
        ),
    ]

    return llm_math_chain, tools


llm_math_chain, tools = math_basetool()


class CalculatorInput(BaseModel):
    question: str = Field()


tools.append(
    Tool.from_function(
        func=llm_math_chain.run,
        name="Calculator",
        description="useful for when you need to answer questions about math",
        args_schema=CalculatorInput
        # coroutine= ... <- you can specify an async method if desired as well
    )
)
