from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from dotenv import dotenv_values


token_config = dotenv_values(".env")

chat = ChatOpenAI(openai_api_key=token_config['OPENAI_API_KEY'], streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True, temperature=0)

# Make SystemMessagePromptTemplate
prompt=PromptTemplate(
    template="Propose creative ways to incorporate {food_1} and {food_2} in the cuisine of the users choice.",
    input_variables=["food_1", "food_2"]
)

system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)

# Make HumanMessagePromptTemplate
human_template="{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])


chat_prompt_with_values = chat_prompt.format_prompt(food_1="Bacon", \
                                                   food_2="Shrimp", \
                                                   text="I really like food from Germany.")

resp = chat(chat_prompt_with_values.to_messages())
