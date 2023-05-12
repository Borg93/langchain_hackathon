from sqlalchemy import MetaData, create_engine, insert
from sqlalchemy import Column, Integer, String, Table, Date, Float
from datetime import datetime
from langchain.sql_database import SQLDatabase
from langchain.chains import SQLDatabaseChain
from dotenv import dotenv_values
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, load_tools, Tool


def insert_obs(obs, visits_table, engine):
    stmt = insert(visits_table).values(
        obs_id=obs[0], archive=obs[1], visits=obs[2], date=obs[3]
    )

    with engine.begin() as conn:
        conn.execute(stmt)


def sql_visit_tool(llm):
    metadata_obj = MetaData()

    visits_table = Table(
        "archive_visit",
        metadata_obj,
        Column("obs_id", Integer, primary_key=True),
        Column("archive", String(4), nullable=False),
        Column("visits", Float, nullable=False),
        Column("date", Date, nullable=False),
    )

    observations = [
        [1, "Sweden", 200, datetime(2023, 1, 1)],
        [2, "Sweden", 208, datetime(2023, 1, 2)],
        [3, "Sweden", 232, datetime(2023, 1, 3)],
        [4, "Sweden", 225, datetime(2023, 1, 4)],
        [5, "Sweden", 226, datetime(2023, 1, 5)],
        [6, "Finland", 810, datetime(2023, 1, 1)],
        [7, "Finland", 803, datetime(2023, 1, 2)],
        [8, "Finland", 798, datetime(2023, 1, 3)],
        [9, "Finland", 795, datetime(2023, 1, 4)],
        [10, "Finland", 791, datetime(2023, 1, 5)],
    ]

    engine = create_engine("sqlite:///:memory:")
    metadata_obj.create_all(engine)

    for obs in observations:
        insert_obs(obs, visits_table, engine)

    db = SQLDatabase(engine)
    sql_chain = SQLDatabaseChain(llm=llm, database=db)

    sql_tool = Tool(
        name="Archive Visit DB",
        func=sql_chain.run,
        description="Useful for when you need to answer questions about number of visits and the dates of visit in the archives for both Sweden and Finland",
    )

    tools = load_tools(["llm-math"], llm=llm)

    tools.append(sql_tool)

    return tools


if __name__ == "__main__":
    token_config = dotenv_values("../../.env")

    chat = ChatOpenAI(
        openai_api_key=token_config["OPENAI_API_KEY"],
        model_name="gpt-3.5-turbo",
        # verbose=True,
        temperature=0,
        max_tokens=2000,
        # frequency_penalty=0,
        # presence_penalty=0,
        # top_p=1.0,
    )
    sql_tool = sql_visit_tool(llm=chat)

    sql_agent = initialize_agent(
        agent="zero-shot-react-description",
        tools=sql_tool,
        llm=chat,
        verbose=False,
        max_iterations=3,
        early_stopping_method="generate",
    )

    sql_agent.run(
        input="What is the number of visits to Sweden's archives? Only answer the number in total."
    )
