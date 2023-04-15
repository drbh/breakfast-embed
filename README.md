# 🥐💤 breakfast-embed

yet another hnsw server for embedding, storing and searching vectors written in rust.

built on:

- hnsw
- actix-web
- sqlite3
- openai embedding api

shout out to [Instant Domain](https://github.com/InstantDomain/instant-distance) for their fantastic work on hnsw 🙇‍♂️

## 😅 Why?

There are a lot of great projects out there for embedding, storing and searching vectors. I wanted to build something that was easy to use and easy to deploy.

## 📦 Install

```bash
git clone htt://github.com/drbh/breakfast-embed
cd breakfast-embed
cargo run --release
```

## 🚀 Usage

The easiest way to get started using breakfast-embed is via the TypeScript client library [breakfast-embed-client](client/index.ts).

```typescript
import { EmbeddingAPIClient } from "./client/index.ts";

const client = new EmbeddingAPIClient("http://localhost:8080");

client
  .embedSearchInsert(["my super secret sentence to embed"])
  .then((response) => console.log(response))
  .catch((error) => console.error(error));
```
