# 🥐💤 breakfast-embed

yet another hnsw server for embedding, storing and searching vectors written in rust.

built on:

- hnsw
- actix-web
- sqlite3
- ~~openai embedding api~~
- local embeddings [pretty-good-embeddings](https://github.com/drbh/pretty-good-embeddings)

shout out to [Instant Domain](https://github.com/InstantDomain/instant-distance) for their fantastic work on hnsw 🙇‍♂️

shout out to [SentenceTransformers](https://github.com/UKPLab/sentence-transformers) and specifically [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) for their lite and fast sentence embeddings model 🙇‍♂️

## 😅 Why?

There are a lot of great projects out there for embedding, storing and searching vectors. I wanted to build something that was easy to use and easy to deploy.

## 📦 Install

```bash
git clone htt://github.com/drbh/breakfast-embed
cd breakfast-embed
cargo run --release
```

## 🚜 Model

The default model is [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) that has been converted to ONNX and optimized for inference using [onnxruntime-rs](https://github.com/nbigaouette/onnxruntime-rs). This is handled by the [pretty-good-embeddings](https://github.com/drbh/pretty-good-embeddings) crate, and that repo includes the 90MB model file.

## 🚀 Usage

The easiest way to get started using breakfast-embed is via the CLI.

![cli](./assets/breakfast-embed-repl.gif)

```bash
# In one terminal start the server
cargo run --bin breakfast-embed
```

```bash
# In another terminal start the repl
cargo run --bin breakfast-embed-cli
#     Finished dev [unoptimized + debuginfo] target(s) in 0.83s
#      Running `target/debug/breakfast-embed-cli`
# > !help

# The following commands are available:

# !clear - clear the screen
# !drop - drop the database
# !exit - exit the program
# !help - print this help menu
# !store - upload the sentences.txt file to the database
# [sentence] - search for similar sentences

# >
```

A more advanced example is to use the chat client. However, this requires a running instance of a locally running LLM server. Please see the [llm/server.py](./llm/server.py) file for a simple example of how to run a server.


**Example of not the smartest chat bot answering questions using breakfast-embed for memory.**
![chat](./assets/breakfast-embed-chat.gif)

The memory used in the example above can be found in [sentences.txt](./sentences.txt); and is a version of an old folk tale call Stone Soup.

```typescript
import { EmbeddingAPIClient } from "./client/index.ts";

const client = new EmbeddingAPIClient("http://localhost:8080");

client
  .embedSearchInsert(["my super secret sentence to embed"])
  .then((response) => console.log(response))
  .catch((error) => console.error(error));
```
