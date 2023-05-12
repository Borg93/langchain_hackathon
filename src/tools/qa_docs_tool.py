from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from datasets import load_dataset
from dotenv import dotenv_values
from tqdm import tqdm
from uuid import uuid4
from langchain.vectorstores import Pinecone
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI


def create_index_vectorstore():
    data = load_dataset("squad", split="train")

    data = data.to_pandas()

    data.drop_duplicates(subset="context", keep="first", inplace=True)

    model_name = "text-embedding-ada-002"
    embed = OpenAIEmbeddings(
        model=model_name, openai_api_key=token_config["OPENAI_API_KEY"]
    )

    index_name = "langchain-retrieval-agent"

    pinecone.init(
        api_key=token_config["PINECONE_KEY"], environment=token_config["PINECONE_ENV"]
    )

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            metric="dotproduct",
            dimension=1536,  #  embedding size of ada-002
        )

    index = pinecone.Index(index_name)

    batch_size = 100

    texts = []
    metadatas = []

    print(data)
    for i in tqdm(range(0, len(data), batch_size)):
        i_end = min(len(data), i + batch_size)
        batch = data.iloc[i:i_end]

        metadatas = [
            {"title": record["title"], "text": record["context"]}
            for j, record in batch.iterrows()
        ]

        documents = batch["context"]
        embeds = embed.embed_documents(documents)

        ids = batch["id"]

        index.upsert(vectors=zip(ids, embeds, metadatas))


if __name__ == "__main__":
    token_config = dotenv_values("../../.env")

    model_name = "text-embedding-ada-002"
    embed = OpenAIEmbeddings(
        model=model_name, openai_api_key=token_config["OPENAI_API_KEY"]
    )

    text_field = "text"
    index_name = "langchain-retrieval-agent"

    pinecone.init(
        api_key=token_config["PINECONE_KEY"], environment=token_config["PINECONE_ENV"]
    )

    index = pinecone.Index(index_name)

    vectorstore = Pinecone(index, embed.embed_query, text_field)

    # results = vectorstore.similarity_search(query, k=3)

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",  # return_messages=True,
        k=5,
        return_messages=True,
    )

    chat = ChatOpenAI(
        openai_api_key=token_config["OPENAI_API_KEY"],
        model_name="gpt-3.5-turbo",
        streaming=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=False,
        temperature=0.0,
        # max_tokens=2000,
        # frequency_penalty=0,
        # presence_penalty=0,
        # top_p=1.0,
    )

    qa = RetrievalQA.from_chain_type(
        llm=chat, chain_type="stuff", retriever=vectorstore.as_retriever()
    )

    tools = [
        Tool(
            name="Knowledge base",
            func=qa.run,
            description=(
                "use this tool when answering general knowledge queries to get "
                "more information about the topic"
            ),
        )
    ]

    agent = initialize_agent(
        agent="chat-conversational-react-description",  # "zero-shot-react-description",
        llm=chat,
        verbose=True,
        tools=tools,
        max_iterations=3,
        early_stopping_method="generate",
        memory=memory,
    )

    query = "When was the college of engineering at the university of Notre Dame established?"
    agent(query)
