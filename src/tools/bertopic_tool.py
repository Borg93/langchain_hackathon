from langchain.tools import BaseTool
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from dotenv import dotenv_values
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, load_tools, Tool
from bertopic.representation import LangChain
from bertopic import BERTopic
import pandas as pd
from umap import UMAP
from sentence_transformers import SentenceTransformer


class BertTopic(BaseTool):
    name = ""
    description = ""

    def __init__(self):
        pass

    def _run(self, text: str):
        pass

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


if __name__ == "__main__":
    token_config = dotenv_values("../../.env")

    docs = """
    The names "John Doe" for males, "Jane Doe" or "Jane Roe" for females, or "Jonnie Doe" and "Janie Doe" for children, or just "Doe" non-gender-specifically are used as placeholder names for a party whose true identity is unknown or must be withheld in a legal action, case, or discussion. The names are also used to refer to acorpse or hospital patient whose identity is unknown. This practice is widely used in the United States and Canada, but is rarely used in other English-speaking countries including the United Kingdom itself, from where the use of "John Doe" in a legal context originates. The names Joe Bloggs or John Smith are used in the UK instead, as well as in Australia and New Zealand.

John Doe is sometimes used to refer to a typical male in other contexts as well, in a similar manner to John Q. Public, known in Great Britain as Joe Public, John Smith or Joe Bloggs. For example, the first name listed on a form is often John Doe, along with a fictional address or other fictional information to provide an example of how to fill in the form. The name is also used frequently in popular culture, for example in the Frank Capra film Meet John Doe. John Doe was also the name of a 2002 American television series.

Similarly, a child or baby whose identity is unknown may be referred to as Baby Doe. A notorious murder case in Kansas City, Missouri, referred to the baby victim as Precious Doe. Other unidentified female murder victims are Cali Doe and Princess Doe. Additional persons may be called James Doe, Judy Doe, etc. However, to avoid possible confusion, if two anonymous or unknown parties are cited in a specific case or action, the surnames Doe and Roe may be used simultaneously; for example, "John Doe v. Jane Roe". If several anonymous parties are referenced, they may simply be labelled John Doe #1, John Doe #2, etc. (the U.S. Operation Delego cited 21 (numbered) "John Doe"s) or labelled with other variants of Doe / Roe / Poe / etc. Other early alternatives such as John Stiles and Richard Miles are now rarely used, and Mary Major has been used in some American federal cases.

    """

    # df = pd.read_parquet("./data/motioner_2014_2021.parquet")
    # print(f"The dataset consists of a total of {len(df)} motions.")
    # df.head()  # Display only the 5 first rows of the dataframe

    # documents = df["text"].tolist()

    # umap_model = UMAP(
    #     n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=1337
    # )

    # # Load KBLab's Swedish sentence transformer model
    # sentence_model = SentenceTransformer(
    #     "KBLab/sentence-bert-swedish-cased", device="cuda"
    # )

    # # Initialize BERTopic with the settings we want
    # topic_model = BERTopic(
    #     embedding_model=sentence_model,
    #     umap_model=umap_model,
    #     calculate_probabilities=True,
    #     verbose=True,
    #     diversity=0.5,
    # )

    # # Fit the model
    # topics, probs = topic_model.fit_transform(documents)

    my_api_key = "KnDKisUqybMVhdOWgTgUGXaHyU6mv6GTTZLqgfXc"

    import cohere
    from bertopic.representation import Cohere
    from bertopic import BERTopic

    # Create your representation model
    co = cohere.Client(my_api_key)
    representation_model = Cohere(co)

    # Use the representation model in BERTopic on top of the default pipeline
    topic_model = BERTopic(representation_model=representation_model)

    prompt = f"I have the following documents: [{docs}]. What topic do they contain?"
    representation_model = Cohere(co, prompt=prompt)
    print(representation_model)
    quit()

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

    chain = load_qa_chain(chat, chain_type="stuff")

    prompt = "What are these documents about? Please give a single label."
    representation_model = LangChain(chain, prompt=prompt)
    print(representation_model)

    topic_model = BERTopic(representation_model=representation_model)
    print(topic_model)

    https://colab.research.google.com/drive/10kB3wfoHSfZE48vEKmznIw-ff36uR8gs?usp=sharing#scrollTo=8NIfF1S6ayQt
