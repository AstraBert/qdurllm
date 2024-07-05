FROM ubuntu

WORKDIR /app

ADD . /app/

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    musl-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" && (echo; echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"') >> /root/.bashrc && eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)" && cd && /bin/bash -c "$(source .bashrc)" && cd /app

RUN /home/linuxbrew/.linuxbrew/bin/brewbrew install llama.cpp

EXPOSE 8000

ENTRYPOINT [ "/home/linuxbrew/.linuxbrew/bin/llama-server", "-m", "/app/.cache/llama.cpp/gemma-2b-it-q4_k_m.gguf", "--port", "8000", "--host", "0.0.0.0", "-n", "1024", "-np", "10"]