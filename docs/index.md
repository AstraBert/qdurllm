<h1 align="center">qdurllm</h1>
<h2 align="center">Search your favorite websites and chat with them, on your desktop🌐</h2>

# Docs in active development!👷‍♀️

They will be soon available on: https://astrabert.github.io/qdurllm/

In the meantime, refer to the **Quickstart guide** in this README!

## Quickstart

### 1. Prerequisites

- [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) package manager
- [`docker`](https://www.docker.com/) and [`docker compose`](https://docs.docker.com/compose/).

### 2. Installation

> [!IMPORTANT]
> _This is only for the pre-release of `v1.0.0`, i.e. `v1.0.0-rc.0`_

1. Clone the `january-2025` branch of this GitHub repo:

```bash
git clone -b january-2025 --single-branch https://github.com/AstraBert/qdurllm.git
cd qdurllm/
```

2. Create the `conda` environment:

```bash
conda env create -f environment.yml
```

3. Pull `qdrant` from Docker Hub:

```bash
docker pull qdrant/qdrant
```

### 3. Launching

1. Launch `qdrant` vector database services with `docker compose` (from within the `qdurllm` folder):

```bash
docker compose up
```

2. Activate the `qdurllm` conda environment you just created:

```bash
conda activate qdurllm
```

3. Go inside the `app` directory and launch the Gradio application:

```bash
cd app/
python3 app.py
```

You should see the app running on `http://localhost:7860` once all the models are downloaded from HuggingFace Hub.

## Relies on

- [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), with Apache 2.0 license
- [nomic-ai/modernbert-embed-base](https://huggingface.co/nomic-ai/modernbert-embed-base), with Apache 2.0 license
- [prithivida/Splade_PP_en_v1](https://huggingface.co/prithivida/Splade_PP_en_v1), with Apache 2.0 license


## Give feedback!

Comment on the [**discussion thread created for this release**](https://github.com/AstraBert/qdurllm/discussions) with your feedback or create [**issues**](https://github.com/AstraBert/qdurllm/issues) :)

