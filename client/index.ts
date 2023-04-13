export class EmbeddingAPIClient {
  private apiUrl: string;

  constructor(apiUrl: string) {
    this.apiUrl = apiUrl;
  }

  private async postData(url: string, data: any): Promise<any> {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)
    });
    return response.json();
  }

  public async embedSearchInsert(sentences: string[]): Promise<any> {
    return this.postData(`${this.apiUrl}/embed_search_insert`, { sentences });
  }

  public async embed(sentences: string[]): Promise<any> {
    return this.postData(`${this.apiUrl}/embed`, { sentences });
  }

  public async search(embedding: number[]): Promise<any> {
    return this.postData(`${this.apiUrl}/search`, embedding);
  }

  public async update(sentences: string[], vectors: number[][]): Promise<any> {
    return this.postData(`${this.apiUrl}/update`, { sentences, vectors });
  }

  public async init(sentences: string[], vectors: number[][]): Promise<any> {
    return this.postData(`${this.apiUrl}/init`, { sentences, vectors });
  }

  public async flush(): Promise<any> {
    return fetch(`${this.apiUrl}/flush`, { method: 'PATCH' });
  }

  public async load(): Promise<any> {
    return fetch(`${this.apiUrl}/load`, { method: 'PATCH' });
  }
}
