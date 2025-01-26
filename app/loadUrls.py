from langchain_community.document_loaders.url import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from rag import upload_text_to_qdrant, client
from typing import List, Dict

def urlload(urls: str) -> List[Dict[str,str]]:
    links = urls.split(",")
    try:
        loader = UnstructuredURLLoader(
            urls=links, method="elements", 
            strategy="fast"
        )
        docs = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        pages = text_splitter.split_documents(docs)
        contents = [{"text": pages[i].page_content, "url": pages[i].metadata["source"]} for i in range(len(pages))]
        return contents
    except Exception as e:
        return f"An error occurred while parsing the URLs: {e}"


def to_db(contents = List[Dict[str, str]]) -> None:
    c = 0
    for content in contents:
        upload_text_to_qdrant(client, "memory", content, c)
        c+=1
    return
        

