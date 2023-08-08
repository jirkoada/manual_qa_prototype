# General information and how to Llama2
Hopefully, this document describes everything important of what I've gathered so far and what is needed to run Llama2 LLM for sentence embeddings and inference.

## How to get access to the weights
Either by copying them from cluster - /lscratch/poludmik or download directly by following instructions on Meta website or meta on huggingface.

### /lscratch/poludmik
(No space on /lscratch, uploaded only th 2-7b model so far)

I downloaded two models from HF: 
* https://huggingface.co/meta-llama/Llama-2-7b-chat (*from_hf_2-7b-chat* folder)
* https://huggingface.co/meta-llama/Llama-2-7b (*from_hf_2-7b*)

### Get access to original Meta models

First, you need to submit a request here (use same email as on huggingface account): https://ai.meta.com/resources/models-and-libraries/llama-downloads/

The allowance e-mail came to me after couple of hours max.

Then, either follow instructions in email and download weights to a desired model via `./download.sh` script, or go to https://huggingface.co/meta-llama and request access for one of their models. They will also check it in a matter of hours and allow you to download the models. I recommend doing it with HF, because the model folders contain _tokenizer.model_ file, which I didn't receive from ./download.sh...

To download any model(repo) from HF, first, create your HF token https://huggingface.co/docs/hub/security-tokens, and then:
```
$ huggingface-cli login --token "your_hf_token"
$ python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="meta-llama/Llama-2-7b-chat", local_dir="path_to_store")
```

## What is a Llama2 model
There are several types of models.

### Models on https://huggingface.co/meta-llama/
* Models differ in size: 7B, 13B, 70B.

* There are Llama-2-7b-**chat** models, which were finetuned for chatbot applications.

* Models marked with Llama-2-7b-**hf** are so called HuggingFace models. They contain different model_file types in their folder(repo), e.g. several .bin files. They can be used directly with **transformers** library without downloading them explicitly:
    ```python
    from transformers import LlamaForCausalLM, LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    sentence = "Q: I have two blue apples and three yellow bananas. What colorare   my apples? A: "

    inputs = tokenizer(sentence, return_tensors="pt")
    generate_ids = model.generate(inputs.input_ids, max_length=30)
    result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    print(" ".join(result))
    ```
    If needed(hopefully not), it is possible to convert simple Llama-2-7b-chat model-folders to Llama-2-7b-chat-hf with a script `convert_pth_weights_to_hf_format.py` in [poludmik test repo](https://github.com/poludmik/Llama2_tests).


### Converting .pth model-folders to a single .bin file 
It is needed to use with:

* from langchain.llms import LlamaCpp

* from langchain.embeddings import LlamaCppEmbeddings

These modules accept a single **.bin** model file as a parameter. 
To convert a whole model-folder downloaded from huggingface, you can use a `convert.py` script from [this repo](https://github.com/ggerganov/llama.cpp/tree/master). It is possible to pass a quantization level as a parameter (q4_0, q4_1, ...). I used [another repos](https://github.com/ggerganov/llama.cpp/tree/master) script to quantize models to q4_0.
```
$ ./quantize weights/ggml-model-f32.bin weights/ggml-model-q4.bin q4_0
```

### Generate sentence embeddings

I used [this](https://api.python.langchain.com/en/latest/embeddings/langchain.embeddings.llamacpp.LlamaCppEmbeddings.html) as a guideline for embedding generation with langchain.

First, you need to install [llama-cpp-python module](https://github.com/abetlen/llama-cpp-python/tree/main) with (hopefully only)
```
$ pip install llama-cpp-python
```
It could not install for me on cluster, but after **ml load Python/3.10.8** it finally worked. Also, I loaded this module before activating my venv, otherwise didn't work.

After this, you can generate embeddings with
```python
from langchain.embeddings import LlamaCppEmbeddings
llama = LlamaCppEmbeddings(model_path="from_hf_2-7b/ggml-model-q4_0.bin", n_ctx=2048)
query_result = llama.embed_query("Sentence to embed.")
print(len(query_result))
```
You can also check [test.py](https://github.com/poludmik/Llama2_tests/blob/master/test.py) for more examples of Llama applications.

## Notes
I used `from_hf_2-7b/ggml-model-q4_0.bin` and `from_hf_2-7b-chat/ggml-model-q4_0.bin` models to generate FAISS indexes. 














