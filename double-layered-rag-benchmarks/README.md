# Small benchmark test results

This small benchmark was conceived to test the solidity of the double-layered RAG approach 
implemented inside *qdurllm* as a retrieval technique. 

## RAG workflow

The RAG workflow goes like this:

- All text batches, obtained by chunking URL contents, are first encoded by `All-MiniLM-L6-v2` and uploaded to a permanent Qdrant collection
- The first retrieval call extract the 10 closest matches to the query in terms of cosine distance, based on `All-MiniLM-L6-v2` search
- The 10 best hits get re-encoded by `sentence-t5-base` and uploaded to a non-permanent Qdrant collection (gets deleted at every round after use)
- `sentence-t5-base` performs a second retrieval call extracting the best match to the original query, and this best match gets returned

## First test

Benchmark is based on the content of 4 web pages: 

- https://www.technologyreview.com/2023/12/05/1084417/ais-carbon-footprint-is-bigger-than-you-think/
- https://semiengineering.com/ai-power-consumption-exploding/
- https://www.piie.com/blogs/realtime-economics/2024/ais-carbon-footprint-appears-likely-be-alarming
- https://news.climate.columbia.edu/2023/06/09/ais-growing-carbon-footprint/ 

The content of these URLs was chunked up and uploaded to Qdrant collections, and at the same time smaller portions of each chunk (encompassing 10-25% of the text) were used for querying, and the retrieved results compared with the original full text.

## First results

The correct/total retrievals ratio for the only `All-MiniLM-L6-v2` is 81.54%, whereas the correct/total retrievals ratio for the previously described double-layered `All-MiniLM-L6-v2` + `sentence-t5-base` goes up 93.85%, equalling the one of `sentence-t5-base` alone. Following a double-layered approach with switched roles for the two encoders yields a correct/total retrievals ratio of 84.62%.

The advantage of this technique is that it does not require that all the chunks of text are encoded in 768-dimensional vectors (as would happen if we adopted `sentence-t5-base` alone), but this step is done dynamically at each vector call. As you can see, it also definitely improves the performance of the sole `All-MiniLM-L6-v2` by little more than 12%.

The disadvantage is in the execution time: on a 8GB RAM-12 cores Windows 10 laptop, double-layered RAG takes an average of 8.39 s, against the 0.23 s of the sole `sentence-t5-base`.

## Second test

The second benchmark is based on []() dataset, available on HuggingFace. It is a Q&A dataset based on a set of 358 answers (used as content to retrieve) and questions (used as retrieval queries).

## Second results

- Avg time for `All-MiniLM-L6-v2`: 0.13889479704117508 +/- 0.018010675079187972
- Avg time for `sentence-t5-base`: 0.3546350625123871 +/- 0.1378480017367839
- Avg time for `All-MiniLM-L6-v2` + `sentence-t5-base`: 10.722357098306164 +/- 1.2552639024596886
- Avg time for `sentence-t5-base` + `All-MiniLM-L6-v2`: 2.722710152020615 +/- 0.30703480688834284
- Correct/Total retrievals for `All-MiniLM-L6-v2`: 0.41853932584269665
- Correct/Total retrievals for `sentence-t5-base`: 0.5084269662921348
- Correct/Total retrievals for `All-MiniLM-L6-v2` + `sentence-t5-base`: 0.5028089887640449
- Correct/Total retrievals for `sentence-t5-base` + `All-MiniLM-L6-v2`: 0.42134831460674155

## Code availability

The benchmark test code is available [here](./scripts/benchmark_test.py) for the first test and [here](./rageval.ipynb) for the second one.

## Contributions

If you happen to have time and a powerful hardware, you can carry on vaster tests using the script referenced before: it would be great!
