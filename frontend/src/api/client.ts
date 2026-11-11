// All calls go through /api, which the Vite dev server proxies to the FastAPI app.
export const API_BASE = "/api";

export interface HealthResponse {
  status: "ok" | "degraded";
  version: string;
  database: "ok" | "error";
  database_error: string | null;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`GET /health failed: ${res.status}`);
  return res.json();
}
