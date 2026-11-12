import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import { BrowserRouter, NavLink, Route, Routes } from "react-router-dom";
import { api } from "./api/client";
import { ComparePage } from "./pages/ComparePage";
import { RunDetailPage } from "./pages/RunDetailPage";
import { RunsPage } from "./pages/RunsPage";

const queryClient = new QueryClient({ defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } } });

function HealthIndicator() {
  const { data, error } = useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 15000 });
  const text = error ? "API unreachable" : !data ? "checking…" : data.status === "ok" ? `API ok · db ok · v${data.version}` : `API degraded: ${data.database_error}`;
  return <span className="health" title={data?.database_error ?? ""}>{text}</span>;
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <header className="topbar">
          <span className="brand">LLM Eval Platform</span>
          <nav style={{ display: "flex", gap: "1rem" }}>
            <NavLink to="/" end>Runs</NavLink>
            <NavLink to="/compare">Compare</NavLink>
            <a href={`${window.location.protocol}//${window.location.hostname}:8000/docs`} target="_blank" rel="noreferrer">API docs</a>
          </nav>
          <HealthIndicator />
        </header>
        <Routes>
          <Route path="/" element={<RunsPage />} />
          <Route path="/runs/:id" element={<RunDetailPage />} />
          <Route path="/compare" element={<ComparePage />} />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
