// All calls go through /api, which the Vite dev server proxies to the FastAPI app.
export const API_BASE = "/api";

export interface HealthResponse {
  status: "ok" | "degraded";
  version: string;
  database: "ok" | "error";
  database_error: string | null;
}

export interface Model {
  id: number;
  name: string;
  hf_repo: string | null;
  backend: string;
  endpoint_url: string;
  quantization: string | null;
  params_b: number | null;
  notes: string | null;
  resource_agent_url: string | null;
  created_at: string;
}

export interface Dataset {
  id: number;
  name: string;
  source_url: string;
  license: string;
  language: string;
  n_items: number;
  split: string;
  seed: number;
  materialised: boolean;
}

export interface Progress {
  n_total: number;
  n_done: number;
  n_correct: number;
  n_scored: number;
  n_errors: number;
  n_unparseable: number;
}

export type RunStatus = "pending" | "running" | "completed" | "failed" | "cancelled";

export interface RunSummary {
  id: number;
  model_id: number;
  dataset_id: number;
  status: RunStatus;
  config_json: Record<string, unknown>;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
  model_name: string;
  quantization: string | null;
  dataset_name: string;
  language: string;
  progress: Progress;
  accuracy: number | null;
}

export interface Metric {
  dimension: string;
  name: string;
  value: number | null;
  unit: string | null;
  language: string | null;
  extra_json: Record<string, unknown> | null;
}

export interface RunDetail extends RunSummary {
  metrics: Metric[];
}

export interface ResultItem {
  id: number;
  item_id: string;
  language: string | null;
  subject: string | null;
  prompt: string;
  response: string | null;
  parsed_answer: string | null;
  correct: boolean | null;
  ttft_ms: number | null;
  latency_ms: number | null;
  output_tokens: number | null;
  error: string | null;
  extra_json: Record<string, unknown> | null;
}

export interface Page<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
}

export interface CompareCell {
  run_id: number;
  value: number | null;
  extra_json: Record<string, unknown> | null;
}

export interface CompareRow {
  dimension: string;
  name: string;
  unit: string | null;
  language: string | null;
  cells: CompareCell[];
}

export interface CompareResponse {
  runs: RunSummary[];
  rows: CompareRow[];
}

export interface RunCreate {
  model_id: number;
  dataset_id: number;
  dimensions: string[];
  n_items: number;
  seed: number;
  concurrency: number;
}

export interface ModelCreate {
  name: string;
  hf_repo: string | null;
  backend: "llama_cpp" | "vllm" | "openai_compat";
  endpoint_url: string;
  quantization: string | null;
  params_b: number | null;
  notes: string | null;
  resource_agent_url: string | null;
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "content-type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

export const api = {
  health: () => request<HealthResponse>("/health"),
  models: () => request<Model[]>("/models"),
  createModel: (body: ModelCreate) => request<Model>("/models", { method: "POST", body: JSON.stringify(body) }),
  deleteModel: (id: number) => request<void>(`/models/${id}`, { method: "DELETE" }),
  datasets: () => request<Dataset[]>("/datasets"),
  runs: () => request<RunSummary[]>("/runs"),
  run: (id: number) => request<RunDetail>(`/runs/${id}`),
  createRun: (body: RunCreate) => request<RunSummary>("/runs", { method: "POST", body: JSON.stringify(body) }),
  cancelRun: (id: number) => request<RunSummary>(`/runs/${id}/cancel`, { method: "POST" }),
  deleteRun: (id: number) => request<void>(`/runs/${id}`, { method: "DELETE" }),
  runItems: (id: number, params: { page?: number; page_size?: number; only_failed?: boolean }) => {
    const q = new URLSearchParams();
    if (params.page) q.set("page", String(params.page));
    if (params.page_size) q.set("page_size", String(params.page_size));
    if (params.only_failed) q.set("only_failed", "true");
    return request<Page<ResultItem>>(`/runs/${id}/items?${q}`);
  },
  compare: (ids: number[]) => request<CompareResponse>(`/compare?run_ids=${ids.join(",")}`),
  exportUrl: (id: number, kind: "items" | "metrics") => `${API_BASE}/export/${id}.csv?kind=${kind}`,
};
