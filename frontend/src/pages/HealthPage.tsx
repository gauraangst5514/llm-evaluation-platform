import { useQuery } from "@tanstack/react-query";
import { fetchHealth } from "../api/client";

export function HealthPage() {
  const { data, error, isPending, dataUpdatedAt } = useQuery({
    queryKey: ["health"],
    queryFn: fetchHealth,
    refetchInterval: 5000,
  });

  return (
    <main style={{ fontFamily: "system-ui, sans-serif", maxWidth: 640, margin: "3rem auto", padding: "0 1rem" }}>
      <h1>LLM Eval Platform</h1>
      <p>Milestone 0: skeleton. This page polls the API every 5 seconds.</p>
      <table style={{ borderCollapse: "collapse", width: "100%" }}>
        <tbody>
          <Row label="API" value={isPending ? "checking…" : error ? `unreachable (${error.message})` : data.status} />
          <Row label="API version" value={data?.version ?? "—"} />
          <Row label="Database" value={data ? (data.database === "ok" ? "ok" : `error: ${data.database_error}`) : "—"} />
          <Row label="Last checked" value={dataUpdatedAt ? new Date(dataUpdatedAt).toLocaleTimeString() : "—"} />
        </tbody>
      </table>
    </main>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <tr>
      <th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #ddd", width: "40%" }}>{label}</th>
      <td style={{ padding: "0.5rem", borderBottom: "1px solid #ddd", fontFamily: "monospace" }}>{value}</td>
    </tr>
  );
}
