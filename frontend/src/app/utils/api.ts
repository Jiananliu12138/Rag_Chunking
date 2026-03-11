const API_BASE_URL = (import.meta as { env?: Record<string, string | undefined> }).env?.VITE_API_BASE_URL
  ?? 'http://127.0.0.1:8080/api/v1';

export interface ApiResponse<T = any> {
  success: boolean;
  message: string;
  data: T;
}

async function fetchApi<T>(
  endpoint: string,
  options?: RequestInit
): Promise<ApiResponse<T>> {
  const url = `${API_BASE_URL}${endpoint}`;

  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });
    const contentType = response.headers.get('content-type') || '';
    const isJson = contentType.includes('application/json');
    const payload = isJson ? await response.json() : await response.text();

    if (!response.ok) {
      const message =
        typeof payload === 'object' && payload !== null
          ? String(
              (payload as { message?: unknown; detail?: unknown }).message
              ?? (payload as { message?: unknown; detail?: unknown }).detail
              ?? `HTTP ${response.status}`
            )
          : String(payload || `HTTP ${response.status}`);

      return {
        success: false,
        message,
        data: null as T,
      };
    }

    return isJson
      ? payload as ApiResponse<T>
      : {
          success: true,
          message: 'ok',
          data: payload as T,
        };
  } catch (error) {
    const message = error instanceof DOMException && error.name === 'AbortError'
      ? `Request aborted: ${url}`
      : error instanceof Error
        ? error.message
        : 'Request failed';
    return {
      success: false,
      message,
      data: null as T,
    };
  }
}

export const api = {
  // Chunks
  getChunkMethods: () => fetchApi('/chunks/methods'),
  chunkFile: (data: any) => fetchApi('/chunks/chunk-file', {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  chunkText: (data: any) => fetchApi('/chunks/chunk-text', {
    method: 'POST',
    body: JSON.stringify(data),
  }),

  // Index
  buildIndex: (data: any) => fetchApi('/index/build', {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  addIndex: (data: any) => fetchApi('/index/add', {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  listCollections: () => fetchApi('/index/collections'),
  inspectCollections: () => fetchApi('/index/collections/inspect'),
  deleteCollection: (name: string) => fetchApi(`/index/collections/${name}`, {
    method: 'DELETE',
  }),
  deleteDocumentsByMetadata: (collectionName: string, data: any) =>
    fetchApi(`/index/collections/${collectionName}/delete-by-metadata`, {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  // Retrieval
  search: (data: any) => fetchApi('/retrieval/search', {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  generate: (data: any) => fetchApi('/retrieval/generate', {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  generateFile: (data: any) => fetchApi('/retrieval/generate-file', {
    method: 'POST',
    body: JSON.stringify(data),
  }),

  // Eval
  traditionalEval: (data: any) => fetchApi('/eval/traditional', {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  traditionalEvalFile: (data: any) => fetchApi('/eval/traditional-file', {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  ragasEval: (data: any) => fetchApi('/eval/ragas', {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  ragasEvalFile: (data: any) => fetchApi('/eval/ragas-file', {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  retrievalEval: (data: any) => fetchApi('/eval/retrieval', {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  retrievalEvalFile: (data: any) => fetchApi('/eval/retrieval-file', {
    method: 'POST',
    body: JSON.stringify(data),
  }),

  // Component Eval
  chunkQuality: (data: any) => fetchApi('/component-eval/chunk-quality', {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  chunkQualityFile: (data: any) => fetchApi('/component-eval/chunk-quality-file', {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  chunkStickiness: (data: any) => fetchApi('/component-eval/chunk-stickiness', {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  chunkStickinessFile: (data: any) => fetchApi('/component-eval/chunk-stickiness-file', {
    method: 'POST',
    body: JSON.stringify(data),
  }),
};
