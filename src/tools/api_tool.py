from langchain.chains import APIChain
from langchain.agents import Tool
from dotenv import dotenv_values
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
import requests
from bs4 import BeautifulSoup


def api_tool(chat):
    api_docs = """

    # Overview:
    The EHRI portal has an experimental web API, intended for searching and retrieving a subset of EHRI data in structured JSON format. While it is intended that the scope of the API will broaden in future, it is intended to prioritise convenience over semantic precision, providing a somewhat simplified view of EHRI's data relative to that offered by the HTML site.

    At present, information is only available for the following types of item:

    Countries (type: Country)
    Institutions (type: Repository)
    Archival descriptions (type: DocumentaryUnit)
    Virtual archival descriptions (type: VirtualUnit)
    Authorities (also known as Historical Agents, type: HistoricalAgent)
    Keywords (also know as Controlled Vocabulary Concepts, type: CvocConcept)
    The base API URL is /api/v1.


    # API Documentation:
    BASE URL: https://portal.ehri-project.eu/api/v1
    BASIC USAGE
    The queries are formatted in the following way:

    https://api.finna.fi/v1/<action>?<parameters>
    By default the results are returned in JSON  format. JSONP format is used if the request includes a callback parameter:

    https://api.finna.fi/v1/<action>?callback=process
    CORS is also supported, and all origin URLs are allowed.

    The API is not meant for handling large result sets (see the page parameter for more information).
    ## IMPORTANT ## only return one page: limit =1
    
    example: curl "https://portal.ehri-project.eu/api/v1/search?q=Sweden&type=Country&facet=country&page=1&limit=1"



    # Retreive an item:
    For retrieving individual items (of any type) the /{ID} action is provided, with the {ID} being the global EHRI identifier of the item you want.
    example: curl "https://portal.ehri-project.eu/api/v1/us-005578"


    """

    chain_api = APIChain.from_llm_and_api_docs(chat, api_docs, verbose=True)

    tools = [
        Tool(
            name="API tool",
            func=chain_api.run,
            description=(
                "use this tool when you neeed more information EHRI, which stands for European Holocaust Research Infrastructure. The EHRI project brings together an international consortium of archives, libraries, museums, memorial and research institutions."
            ),
        )
    ]

    return tools


class WebPageTool(BaseTool):
    name = "Get Webpage"
    description = "Useful for when you need to get the content from a specific webpage"

    def _run(self, webpage: str):
        response = requests.get(webpage)
        html_content = response.text

        def strip_html_tags(html_content):
            soup = BeautifulSoup(html_content, "html.parser")
            stripped_text = soup.get_text()
            return stripped_text

        stripped_content = strip_html_tags(html_content)
        if len(stripped_content) > 4000:
            stripped_content = stripped_content[:4000]
        return stripped_content

    def _arun(self, webpage: str):
        raise NotImplementedError("This tool does not support async")


def bs4_tool():
    tools = WebPageTool()
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
    tools = api_tool(chat)

    bs4_tools = bs4_tool()

    tools.append(bs4_tools)

    api_agent = initialize_agent(
        agent="zero-shot-react-description",
        tools=tools,
        llm=chat,
        verbose=True,
        max_iterations=3,
        early_stopping_method="generate",
    )

    # api_agent.run(
    #     "Can you find some data about Sweden? limit to one file and return item id"
    # )

    api_agent.run(
        "Can you get data from the webpage: https://portal.ehri-project.eu/help/faq#kix.cp2341mudsau and answer what ehri is?"
    )
