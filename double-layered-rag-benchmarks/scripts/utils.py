from qdrant_client import models
from langchain_community.document_loaders.url import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter


def urlload(urls):
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

class NeuralSearcher:
    def __init__(self, collection_name, client, model):
        self.collection_name = collection_name
        # Initialize encoder model
        self.model = model
        # initialize Qdrant client
        self.qdrant_client = client
    def search(self, text: str, num_return: int):
        # Convert text query into vector
        vector = self.model.encode(text).tolist()

        # Use `vector` for search for closest vectors in the collection
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=None,  # If you don't want any filters for now
            limit=num_return,  # 5 the most closest results is enough
        )
        # `search_result` contains found vector ids with similarity scores along with the stored payload
        # In this function you are interested in payload only
        payloads = [hit.payload for hit in search_result]
        return payloads



def upload_to_qdrant_collection(client, collection_name, encoder, documents):
    client.upload_points(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=idx, vector=encoder.encode(doc["text"]).tolist(), payload=doc
            )
            for idx, doc in enumerate(documents)
        ],
    )
    
def upload_to_qdrant_subcollection(client, collection_name, encoder, documents):
    client.delete_collection(collection_name=collection_name)
    client.create_collection(collection_name = collection_name,
        vectors_config=models.VectorParams(
            size = encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
            distance = models.Distance.COSINE,
        ),
    )
    client.upload_points(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=idx, vector=encoder.encode(doc["text"]).tolist(), payload=doc
            )
            for idx, doc in enumerate(documents)
        ],
    )
