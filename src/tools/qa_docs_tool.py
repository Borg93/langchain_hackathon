from langchain.chains import RetrievalQA


# retrieval qa chain
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
)

from langchain.agents import Tool

tools = [
    Tool(
        name="Knowledge Base",
        func=qa.run,
        description=(
            "use this tool when answering general knowledge queries to get "
            "more information about the topic"
        ),
    )
]

# TODO: https://github.com/pinecone-io/examples/blob/master/generation/langchain/handbook/08-langchain-retrieval-agent.ipynb
# https://www.youtube.com/watch?v=kvdVduIJsc8&list=PLIUOU7oqGTLieV9uTIFMm6_4PXg-hlN6F&index=6&t=10s
