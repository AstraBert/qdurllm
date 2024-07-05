from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from utils import NeuralSearcher, urlload, upload_to_qdrant_collection, upload_to_qdrant_subcollection
import gradio as gr
import time
import requests

client = QdrantClient(host="host.docker.internal", port="6333")
encoder = SentenceTransformer(".cache/huggingface/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/8b3219a92973c328a8e22fadcfa821b5dc75636a")
encoder1 = SentenceTransformer(".cache/huggingface/models--sentence-transformers--sentence-t5-base/snapshots/50c53e206f8b01c9621484a3c0aafce4e55efebf")
coll_name = "HTML_collection"
subcoll_name = "Subcollection"

client.recreate_collection(
    collection_name = coll_name,
    vectors_config=models.VectorParams(
        size = encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
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

def call_upload2qdrant(url):
    global client, encoder, coll_name 
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

demo0 = gr.Interface(fn=call_upload2qdrant, title="Upload URL content to Qdrant", inputs=gr.Textbox(label="URL(s)", info="Add one URL or more (if more, you should provide them comma-separated, like this: URL1,URL2,...,URLn)"), outputs=gr.Textbox(label="Logs"))

def llama_cpp_respond(query, max_new_tokens, temperature, repeat_penalty, seed):
    url = "http://localhost:8000/completion"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "prompt": query,
        "n_predict": int(max_new_tokens),
        "temperature": temperature,
        "repeat_penalty": repeat_penalty,
        "seed": int(seed),
    }

    response = requests.post(url, headers=headers, json=data)

    a = response.json()
    return a["content"]


def reply(message, history, ntokens, rep_pen, temp, seed):
    global client, encoder, encoder1, coll_name, subcoll_name
    results = reranked_rag(client, encoder, encoder1, coll_name, subcoll_name, message)
    print(results)
    prompt = f'Context: {results[0]["text"]}; Request: {message}; Answer to request based on context: '
    response = llama_cpp_respond(prompt, ntokens, temp, rep_pen, seed)
    response = response + f' [^Reference]\n\n\n[^Reference]: {results[0]["url"]}'
    this_hist = ''
    for char in response:
        this_hist += char
        time.sleep(0.0001)
        yield this_hist

def direct_search(input_text):
    global client, encoder, encoder1, coll_name, subcoll_name
    results = reranked_rag(client, encoder, encoder1, coll_name, subcoll_name, input_text)
    return results[0]["text"], f"Reference website [here]({results[0]['url']})"

demo2 = gr.Interface(fn=direct_search, inputs=gr.Textbox(label="Search Query", value="Input your search query here...", ), outputs=[gr.Textbox(label="Retrieved Content"), gr.Markdown(label="URL")], title="Search your URLs")

user_max_new_tokens = gr.Slider(0, 1024, value=512, label="Max new tokens", info="Select max output tokens (higher number of tokens will result in a longer latency)")
user_max_rep_pen = gr.Slider(0, 1, value=0.5, label="Temperature", info="Select generation temperature")
user_max_temperature = gr.Slider(0, 10, value=1.2, label="Repetition penalty", info="Select repetition penalty")
user_seed = gr.Textbox(label="Seed", info="Write generation seed (if you are not familiar with it, leave it as it is)", value="4294967295")
additional_accordion = gr.Accordion(label="Parameters to be set before you start chatting", open=True)
chatbot = gr.Chatbot(height=400)
demo1 = gr.ChatInterface(fn=reply, title="Chat with your URLs", chatbot=chatbot, additional_inputs=[user_max_new_tokens, user_max_rep_pen, user_max_temperature, user_seed], additional_inputs_accordion=additional_accordion)

my_theme = gr.themes.Soft(primary_hue=gr.themes.colors.rose, secondary_hue=gr.themes.colors.pink)

demo = gr.TabbedInterface([demo0, demo1, demo2], ["Upload URLs", "Chat with URLs", "Direct Search"], theme=my_theme)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)