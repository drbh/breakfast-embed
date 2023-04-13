This endpoint does most of the functionality of the API. It takes a sentence, calls the embedding API, finds the nearest neighbors, and then inserts the new sentence into the HNSW map.

```bash
curl --request POST \
  --url http://localhost:8080/embed_search_insert \
  --header 'Content-Type: application/json' \
  --data '{
	"sentences": [
		"Initalize the HNSW map with new sentence embeddings."
	]
}'
```

This one is basiclly a pass throuh to the embedding API. It takes a sentence and returns the embedding.

```bash
curl --request POST \
  --url http://localhost:8080/embed \
  --header 'Content-Type: application/json' \
  --data '{
	"sentences": [
		"red"
	]
}'
```

This endpoint takes a sentence and returns the nearest neighbors. Note the array must be 1536 elements long.

```bash
curl --request POST \
  --url http://localhost:8080/search \
  --header 'Content-Type: application/json' \
  --data '[
	-0.00006423246,
	-0.024778806,
	-0.0023977335,
    ...
]'
```

Directly add a sentence to the HNSW map. Note that this does not call the embedding API, so the sentence must be 1536 elements long.

```bash
curl --request POST \
  --url http://localhost:8080/update \
  --header 'Content-Type: application/json' \
  --data '{
	"sentences": [
		"blue"
	],
	"vectors": [
		[
			0.005390428,
			-0.0073414794,
			0.0058082636,
            ...
		]
	]
}'
```

Manually init the HNSW map. This is called with the embed_search_insert endpoint, but can be called manually.

```bash
curl --request POST \
  --url http://localhost:8080/init \
  --header 'Content-Type: application/json' \
  --data '{
	"sentences": [
		"hi my name is"
	],
	"vectors": [
		[
			-0.034310415,
			-0.0060325647,
            ...
	    ]
	]
}'
```

Saves to disk.

```bash
curl --request PATCH \
  --url http://localhost:8080/flush
```

Load from disk. This is called automatically on startup.


```bash
curl --request PATCH \
  --url http://localhost:8080/load
```