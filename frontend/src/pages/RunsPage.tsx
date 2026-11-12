import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, type Dataset, type Model, type ModelCreate, type RunCreate } from "../api/client";
import { Card, EmptyState, ErrorText, ProgressBar, StatusBadge } from "../components/ui";
import { duration, fmtValue, languageName, quantLabel } from "../lib/format";

const ALL_DIMENSIONS = ["accuracy", "latency", "resources", "tokenizer", "reliability"];

export function RunsPage() {
  const navigate = useNavigate();
  const runs = useQuery({
    queryKey: ["runs"],
    queryFn: api.runs,
    refetchInterval: (q) => (q.state.data?.some((r) => r.status === "running" || r.status === "pending") ? 2000 : 15000),
  });
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: api.datasets });
  const [selected, setSelected] = useState<number[]>([]);

  const toggle = (id: number) =>
    setSelected((s) => (s.includes(id) ? s.filter((x) => x !== id) : s.length < 6 ? [...s, id] : s));

  return (
    <div className="page">
      <h1>Runs</h1>
      <p className="muted">
        Every number here comes from a stored run. Select 2 to 6 completed runs to compare them.
      </p>

      <div className="grid-2">
        <NewRunForm models={models.data ?? []} datasets={datasets.data ?? []} />
        <ModelsPanel models={models.data ?? []} />
      </div>

      <Card
        title="All runs"
        right={
          <button
            className="primary"
            disabled={selected.length < 2}
            onClick={() => navigate(`/compare?runs=${selected.join(",")}`)}
          >
            Compare {selected.length ? `(${selected.length})` : ""}
          </button>
        }
      >
        <ErrorText error={runs.error} />
        {runs.data && runs.data.length === 0 && (
          <EmptyState title="No runs yet">
            Register a model server and start a run above, or use <code>scripts/run_pilot.py</code>.
          </EmptyState>
        )}
        {runs.data && runs.data.length > 0 && (
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th />
                  <th>#</th>
                  <th>Model</th>
                  <th>Quant</th>
                  <th>Dataset</th>
                  <th>Status</th>
                  <th>Progress</th>
                  <th className="num">Accuracy</th>
                  <th className="num">n</th>
                  <th>Duration</th>
                </tr>
              </thead>
              <tbody>
                {runs.data.map((r) => (
                  <tr key={r.id} className="clickable" onClick={() => navigate(`/runs/${r.id}`)}>
                    <td onClick={(e) => e.stopPropagation()}>
                      <input
                        type="checkbox"
                        checked={selected.includes(r.id)}
                        disabled={r.status !== "completed" && !selected.includes(r.id)}
                        onChange={() => toggle(r.id)}
                        aria-label={`select run ${r.id}`}
                      />
                    </td>
                    <td><Link to={`/runs/${r.id}`}>{r.id}</Link></td>
                    <td>{r.model_name}</td>
                    <td className="mono">{quantLabel(r.quantization)}</td>
                    <td>
                      {r.dataset_name} <small>({languageName(r.language)})</small>
                    </td>
                    <td><StatusBadge status={r.status} /></td>
                    <td style={{ minWidth: 120 }}>
                      <ProgressBar done={r.progress.n_done} total={r.progress.n_total} />
                      <small>
                        {r.progress.n_done}/{r.progress.n_total}
                        {r.progress.n_errors ? ` · ${r.progress.n_errors} errors` : ""}
                      </small>
                    </td>
                    <td className="num">{fmtValue(r.accuracy, "fraction")}</td>
                    <td className="num">{r.progress.n_scored || "—"}</td>
                    <td>{duration(r.started_at, r.finished_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}

function NewRunForm({ models, datasets }: { models: Model[]; datasets: Dataset[] }) {
  const qc = useQueryClient();
  const [form, setForm] = useState<RunCreate>({
    model_id: 0, dataset_id: 0, dimensions: ALL_DIMENSIONS, n_items: 50, seed: 42, concurrency: 2,
  });
  const modelId = form.model_id || models[0]?.id || 0;
  const datasetId = form.dataset_id || datasets[0]?.id || 0;
  const dataset = datasets.find((d) => d.id === datasetId);
  const create = useMutation({
    mutationFn: () => api.createRun({ ...form, model_id: modelId, dataset_id: datasetId }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["runs"] }),
  });
  const toggleDim = (d: string) =>
    setForm((f) => ({
      ...f,
      dimensions: f.dimensions.includes(d) ? f.dimensions.filter((x) => x !== d) : [...f.dimensions, d],
    }));

  return (
    <Card title="New run">
      {models.length === 0 ? (
        <EmptyState title="No model servers registered">Register one on the right first.</EmptyState>
      ) : (
        <form
          className="inline"
          onSubmit={(e) => {
            e.preventDefault();
            create.mutate();
          }}
        >
          <label className="field">
            Model
            <select value={modelId} onChange={(e) => setForm({ ...form, model_id: Number(e.target.value) })}>
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} · {quantLabel(m.quantization)}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            Dataset
            <select value={datasetId} onChange={(e) => setForm({ ...form, dataset_id: Number(e.target.value) })}>
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name} ({languageName(d.language)}, {d.n_items} items)
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            Items (max {dataset?.n_items ?? "?"})
            <input
              type="number" min={1} max={dataset?.n_items ?? 5000} value={form.n_items}
              onChange={(e) => setForm({ ...form, n_items: Number(e.target.value) })}
            />
          </label>
          <label className="field">
            Concurrency
            <input type="number" min={1} max={64} value={form.concurrency}
              onChange={(e) => setForm({ ...form, concurrency: Number(e.target.value) })} />
          </label>
          <label className="field">
            Seed
            <input type="number" value={form.seed} onChange={(e) => setForm({ ...form, seed: Number(e.target.value) })} />
          </label>
          <div className="checks" style={{ gridColumn: "1 / -1" }}>
            {ALL_DIMENSIONS.map((d) => (
              <label key={d}>
                <input type="checkbox" checked={form.dimensions.includes(d)} onChange={() => toggleDim(d)} /> {d}
              </label>
            ))}
          </div>
          <div>
            <button className="primary" type="submit" disabled={create.isPending || !form.dimensions.length}>
              {create.isPending ? "Starting…" : "Start run"}
            </button>
          </div>
          <ErrorText error={create.error} />
        </form>
      )}
    </Card>
  );
}

function ModelsPanel({ models }: { models: Model[] }) {
  const qc = useQueryClient();
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState<ModelCreate>({
    name: "", hf_repo: "", backend: "llama_cpp", endpoint_url: "http://host.docker.internal:8080",
    quantization: "", params_b: null, notes: "", resource_agent_url: "",
  });
  const create = useMutation({
    mutationFn: () =>
      api.createModel({
        ...form,
        hf_repo: form.hf_repo || null, quantization: form.quantization || null,
        notes: form.notes || null, resource_agent_url: form.resource_agent_url || null,
      }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["models"] }); setOpen(false); },
  });
  const remove = useMutation({
    mutationFn: (id: number) => api.deleteModel(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["models"] }),
  });
  const set = (k: keyof ModelCreate, v: string) => setForm({ ...form, [k]: v });

  return (
    <Card
      title="Model servers"
      right={<button className="ghost" onClick={() => setOpen((o) => !o)}>{open ? "Close" : "Register"}</button>}
    >
      {open && (
        <form className="inline" style={{ marginBottom: "0.75rem" }} onSubmit={(e) => { e.preventDefault(); create.mutate(); }}>
          <label className="field">Name<input required value={form.name} onChange={(e) => set("name", e.target.value)} placeholder="Qwen2.5-7B-Instruct" /></label>
          <label className="field">Quantization<input value={form.quantization ?? ""} onChange={(e) => set("quantization", e.target.value)} placeholder="Q4_K_M" /></label>
          <label className="field">HF repo (tokenizer)<input value={form.hf_repo ?? ""} onChange={(e) => set("hf_repo", e.target.value)} placeholder="Qwen/Qwen2.5-7B-Instruct" /></label>
          <label className="field">Backend
            <select value={form.backend} onChange={(e) => set("backend", e.target.value)}>
              <option value="llama_cpp">llama.cpp</option><option value="vllm">vLLM</option><option value="openai_compat">other OpenAI-compatible</option>
            </select>
          </label>
          <label className="field">Endpoint URL<input required value={form.endpoint_url} onChange={(e) => set("endpoint_url", e.target.value)} /></label>
          <label className="field">Resource agent URL<input value={form.resource_agent_url ?? ""} onChange={(e) => set("resource_agent_url", e.target.value)} placeholder="http://host.docker.internal:9108" /></label>
          <label className="field" style={{ gridColumn: "1 / -1" }}>Notes (server version, flags)<input value={form.notes ?? ""} onChange={(e) => set("notes", e.target.value)} /></label>
          <div><button className="primary" type="submit" disabled={create.isPending}>Save</button></div>
          <ErrorText error={create.error} />
        </form>
      )}
      {models.length === 0 ? (
        <EmptyState title="None registered">See <code>scripts/start_llama_server.md</code>.</EmptyState>
      ) : (
        <div className="table-wrap">
          <table className="data">
            <thead><tr><th>#</th><th>Model</th><th>Quant</th><th>Backend</th><th>Endpoint</th><th /></tr></thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.id}>
                  <td>{m.id}</td>
                  <td>{m.name}{m.hf_repo ? <><br /><small className="mono">{m.hf_repo}</small></> : null}</td>
                  <td className="mono">{quantLabel(m.quantization)}</td>
                  <td>{m.backend}</td>
                  <td className="mono">{m.endpoint_url}{m.resource_agent_url ? <><br /><small>agent: {m.resource_agent_url}</small></> : null}</td>
                  <td><button className="ghost" title="Delete (refused if runs exist)" onClick={() => remove.mutate(m.id)}>✕</button></td>
                </tr>
              ))}
            </tbody>
          </table>
          <ErrorText error={remove.error} />
        </div>
      )}
    </Card>
  );
}
