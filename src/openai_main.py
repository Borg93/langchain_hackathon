from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents import AgentType, initialize_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import gradio as gr
import datetime
from tools.builtin_tools import wiki_repl_search_tool
from tools.ner_tool import ner_tool
from dotenv import dotenv_values


if __name__ == "__main__":
    token_config = dotenv_values("../.env")

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",  # return_messages=True,
        input_key="input",
        output_key="output",
        ai_prefix="AI",
        human_prefix="User",
        k=3,
        return_messages=True,
    )

    chat = ChatOpenAI(
        openai_api_key=token_config["OPENAI_API_KEY"],
        model_name="gpt-3.5-turbo",
        streaming=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=False,
        temperature=0,
        max_tokens=2000,
        # frequency_penalty=0,
        # presence_penalty=0,
        # top_p=1.0,
    )

    tools = wiki_repl_search_tool()

    prompt = "Vad heter finiska riksarkivet på finska?"

    sys_msg = """Assistant is a large language model trained by OpenAI.

    Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

    Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

    Assistant also doesn't know information about content on webpages and should always check if asked.

    Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
    """

    agent = initialize_agent(
        agent="chat-conversational-react-description",  # "zero-shot-react-description",
        llm=chat,
        verbose=False,
        tools=tools,
        max_iterations=3,
        early_stopping_method="generate",
        memory=memory,
    )

    agent.llm_chain.prompt.messages[0].prompt.template = sys_msg

    print(agent.agent.llm_chain.prompt.messages[0])

    agent.run(input=prompt)
