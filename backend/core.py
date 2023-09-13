from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
import pinecone
import os
from consts import INDEX_NAME

pinecone.init(api_key=os.environ['PINECONE_API_KEY'],
              environment=os.environ['PINECONE_ENVIRONMENT_REGION'])


def run_llm(query: str) -> any:
    embeddings = OpenAIEmbeddings()
    docSearch = Pinecone.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=chat, chain_type="stuff",  retriever=docSearch.as_retriever(),
        return_source_documents=True)
    return qa({'query': query})


if __name__ == '__main__':
    print(run_llm("What is Langchain?"))
