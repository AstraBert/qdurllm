from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from utils import NeuralSearcher, urlload, upload_to_qdrant_collection, upload_to_qdrant_subcollection

client = QdrantClient("http://localhost:6333")
encoder = SentenceTransformer("all-MiniLM-L6-v2")
encoder1 = SentenceTransformer("sentence-t5-base")
coll_name = "Small_HTML_collection"
coll_name1 = "HTML_collection"
subcoll_name = "Subcollection"
small_subcoll_name = "Small_Subcollection"


client.recreate_collection(
    collection_name = coll_name,
    vectors_config=models.VectorParams(
        size = encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
        distance = models.Distance.COSINE,
    ),
)

client.recreate_collection(
    collection_name = coll_name1,
    vectors_config=models.VectorParams(
        size = encoder1.get_sentence_embedding_dimension(),  # Vector size is defined by used model
        distance = models.Distance.COSINE,
    ),
)


client.recreate_collection(
    collection_name = subcoll_name,
    vectors_config=models.VectorParams(
        size = encoder1.get_sentence_embedding_dimension(),  # Vector size is defined by used model
        distance = models.Distance.COSINE,
    ),
)

client.recreate_collection(
    collection_name = small_subcoll_name,
    vectors_config=models.VectorParams(
        size = encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
        distance = models.Distance.COSINE,
    ),
)

def call_upload2qdrant(url, coll_name, encoder):
    global client 
    documents = urlload(url)
    if type(documents) == list:
        try:
            upload_to_qdrant_collection(client, coll_name, encoder, documents)
            return "Successfully uploaded URL content to Qdrant collection"
        except Exception as e:
            return f"An error occured: {e}"
    else:
        return documents

def reranked_rag(client, encoder0, encoder1, collection0, collection1, message):
    txt2txt0 = NeuralSearcher(collection0, client, encoder0)
    search_results0 = txt2txt0.search(message, 10)
    upload_to_qdrant_subcollection(client, collection1, encoder1, search_results0)
    txt2txt1 = NeuralSearcher(collection1, client, encoder1)
    search_results1 = txt2txt1.search(message, 1)
    return search_results1

def direct_search(input_text):
    global client, encoder, encoder1, coll_name, subcoll_name
    results = reranked_rag(client, encoder, encoder1, coll_name, subcoll_name, input_text)
    return results[0]["text"]


if __name__=="__main__":
    docs = urlload("https://www.technologyreview.com/2023/12/05/1084417/ais-carbon-footprint-is-bigger-than-you-think/,https://semiengineering.com/ai-power-consumption-exploding/,https://www.piie.com/blogs/realtime-economics/2024/ais-carbon-footprint-appears-likely-be-alarming,https://news.climate.columbia.edu/2023/06/09/ais-growing-carbon-footprint/")

    print(call_upload2qdrant("https://www.technologyreview.com/2023/12/05/1084417/ais-carbon-footprint-is-bigger-than-you-think/,https://semiengineering.com/ai-power-consumption-exploding/,https://www.piie.com/blogs/realtime-economics/2024/ais-carbon-footprint-appears-likely-be-alarming,https://news.climate.columbia.edu/2023/06/09/ais-growing-carbon-footprint/",coll_name,encoder))

    print(call_upload2qdrant("https://www.technologyreview.com/2023/12/05/1084417/ais-carbon-footprint-is-bigger-than-you-think/,https://semiengineering.com/ai-power-consumption-exploding/,https://www.piie.com/blogs/realtime-economics/2024/ais-carbon-footprint-appears-likely-be-alarming,https://news.climate.columbia.edu/2023/06/09/ais-growing-carbon-footprint/",coll_name1,encoder1))

    from math import ceil
    import time

    newdocs = {}
    import random as r
    for doc in docs:
        text = doc["text"].split(" ")
        n = r.randint(ceil(len(text)/10), ceil(len(text)/4))
        newtext = " ".join(text[:n])
        newdocs.update({newtext: doc["text"]})

    print("Successful fragmentation")
    txt2txt0 = NeuralSearcher(coll_name, client, encoder)
    txt2txt1 = NeuralSearcher(coll_name1, client, encoder1)

    times0 = []
    times1 = []
    times01 = []
    times10 = []
    points0 = 0
    points1 = 0
    points01 = 0
    points10 = 0

    from statistics import mean, stdev

    print("Started benchmark")
    for k in newdocs:
        start0 = time.time()
        results0 = txt2txt0.search(k,1)
        end0 = time.time()
        if results0[0]["text"] == newdocs[k]:
            points0+=1    
        times0.append(end0-start0)
        start1 = time.time()
        results1 = txt2txt1.search(k,1)
        end1 = time.time()
        if results1[0]["text"] == newdocs[k]:
            points1+=1    
        times1.append(end1-start1)
        start01 = time.time()
        results01 = reranked_rag(client, encoder, encoder1, coll_name, subcoll_name, k)
        end01 = time.time()
        if results01[0]["text"] == newdocs[k]:
            points01+=1    
        times01.append(end01-start01)
        start10 = time.time()
        results10 = reranked_rag(client, encoder1, encoder, coll_name1, small_subcoll_name, k)
        end10 = time.time()
        if results10[0]["text"] == newdocs[k]:
            points10+=1    
        times10.append(end10-start10)

    print(f"Avg time for All-MiniLM-L6-v2: {mean(times0)} +/- {stdev(times0)}")
    print(f"Avg time for sentence-t5-base: {mean(times1)} +/- {stdev(times1)}")
    print(f"Avg time for All-MiniLM-L6-v2 + sentence-t5-base: {mean(times01)} +/- {stdev(times01)}")
    print(f"Avg time for sentence-t5-base + All-MiniLM-L6-v2: {mean(times10)} +/- {stdev(times10)}")
    print(f"Correct/Total retrievals for All-MiniLM-L6-v2: {points0/len(newdocs)}")
    print(f"Correct/Total retrievals for sentence-t5-base: {points1/len(newdocs)}")
    print(f"Correct/Total retrievals for All-MiniLM-L6-v2 + sentence-t5-base: {points01/len(newdocs)}")
    print(f"Correct/Total retrievals for sentence-t5-base + All-MiniLM-L6-v2: {points10/len(newdocs)}")


