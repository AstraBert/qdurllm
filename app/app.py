from rag import client, SemanticCache, NeuralSearcher, dense_encoder, sparse_encoder
from texInference import pipe
from loadUrls import urlload, to_db
import gradio as gr
import time


searcher = NeuralSearcher("memory", client, dense_encoder, sparse_encoder)
semantic_cache = SemanticCache(client, dense_encoder, "semantic_cache")


def upload2qdrant(url):
    global client
    documents = urlload(url)
    if type(documents) == list:
        try:
            to_db(documents)
            return "URLs successfully uploaded to Qdrant collection!"
        except Exception as e:
            return f"An error occured: {e}"
    else:
        return documents

demo0 = gr.Interface(fn=upload2qdrant, title="Upload URL content to Qdrant", inputs=gr.Textbox(label="URL(s)", info="Add one URL or more (if more, you should provide them comma-separated, like this: URL1,URL2,...,URLn)"), outputs=gr.Textbox(label="Logs"))


def reply(message, history, ntokens, rep_pen, temp, topp, systemins):
    sr = semantic_cache.search_cache(message)
    if sr:
        response = sr
        this_hist = ''
        for c in response:
            this_hist+=c
            time.sleep(0.001)
            yield this_hist
    else:
        context, url = searcher.search_text(message)
        prompt = [{"role": "system", "content": systemins}, {"role": "user", "content": f"This is the context information to reply to my prompt:\n\n{context}"}, {"role": "user", "content": message}]
        results = pipe(prompt, temp, topp, ntokens, rep_pen)
        results = results.split("<|im_start|>assistant\n")[1]
        response = results.replace("<|im_end|>", "")
        semantic_cache.upload_to_cache(message, response)
        this_hist = ''
        for c in response:
            this_hist+=c
            time.sleep(0.001)
            yield this_hist

def direct_search(input_text):
    context, url = searcher.search_text(input_text)
    return context, f"Reference website [here]({url})"

demo2 = gr.Interface(fn=direct_search, inputs=gr.Textbox(label="Search Query", placeholder="Input your search query here...", ), outputs=[gr.Textbox(label="Retrieved Content"), gr.Markdown(label="URL")], title="Search your URLs")

user_max_new_tokens = gr.Slider(0, 4096, value=512, label="Max new tokens", info="Select max output tokens (higher number of tokens will result in a longer latency)")
user_max_temperature = gr.Slider(0, 1, value=0.1, step=0.1, label="Temperature", info="Select generation temperature")
user_max_rep_pen = gr.Slider(0, 10, value=1.2, step=0.1, label="Repetition penalty", info="Select repetition penalty")
user_top_p = gr.Slider(0.1, 1, value=1, step=0.1, label="top_p", info="Select top_p for the generation")
system_ins = gr.Textbox(label="System Prompt", info="Insert your system prompt here", value="You are an helpful web searching assistant. You reply based on the contextual information you are provided with and on your knowledge.")
additional_accordion = gr.Accordion(label="Parameters to be set before you start chatting", open=True)
demo1 = gr.ChatInterface(fn=reply, title="Chat with your URLs", additional_inputs=[user_max_new_tokens, user_max_temperature, user_max_rep_pen, user_top_p, system_ins], additional_inputs_accordion=additional_accordion)

my_theme = gr.themes.Soft(primary_hue=gr.themes.colors.rose, secondary_hue=gr.themes.colors.pink)

demo = gr.TabbedInterface([demo0, demo1, demo2], ["Upload URLs", "Chat with URLs", "Direct Search"], theme=my_theme)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)