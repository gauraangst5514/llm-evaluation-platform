import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, type Metric } from "../api/client";
import { Card, EmptyState, ErrorText, ProgressBar, StatusBadge } from "../components/ui";
import { DIMENSION_TITLES, duration, fmtDate, fmtValue, languageName, modelLabel, sampleSize } from "../lib/format";

export function RunDetailPage() {
  const id = Number(useParams().id);
  const qc = useQueryClient();
  const navigate = useNavigate();
  const run = useQuery({
    queryKey: ["run", id],
    queryFn: () => api.run(id),
    refetchInterval: (q) => (q.state.data && ["pending", "running"].includes(q.state.data.status) ? 2000 : false),
  });
  const failed = useQuery({
    queryKey: ["run-failed", id, run.data?.status],
    queryFn: () => api.runItems(id, { only_failed: true, page_size: 10 }),
    enabled: !!run.data && run.data.progress.n_done > 0,
  });
  const cancel = useMutation({ mutationFn: () => api.cancelRun(id), onSuccess: () => qc.invalidateQueries({ queryKey: ["run", id] }) });
  const remove = useMutation({ mutationFn: () => api.deleteRun(id), onSuccess: () => navigate("/") });

  if (run.error) return <div className="page"><ErrorText error={run.error} /></div>;
  if (!run.data) return <div className="page muted">Loading…</div>;
  const r = run.data;
  const p = r.progress;
  const active = r.status === "pending" || r.status === "running";
  const byDim = groupBy(r.metrics, (m) => m.dimension);
  const cfg = r.config_json as Record<string, unknown>;

  return (
    <div className="page">
      <p><Link to="/">← Runs</Link></p>
      <div className="card-head">
        <h1>Run #{r.id}: {modelLabel(r)} on {r.dataset_name}</h1>
        <div className="pill-row">
          {active && <button onClick={() => cancel.mutate()} disabled={cancel.isPending}>Cancel</button>}
          {!active && (
            <>
              <a className="pill" href={api.exportUrl(r.id, "items")}>items.csv</a>
              <a className="pill" href={api.exportUrl(r.id, "metrics")}>metrics.csv</a>
              <button className="ghost" onClick={() => { if (!remove.isPending) remove.mutate(); }}>Delete</button>
            </>
          )}
        </div>
      </div>
      <p>
        <StatusBadge status={r.status} /> &nbsp; {languageName(r.language)} · seed {String(cfg.seed)} · concurrency{" "}
        {String(cfg.resolved_concurrency ?? cfg.concurrency)} · prompt {String(cfg.prompt_template ?? "—")} · started {fmtDate(r.started_at)} · {duration(r.started_at, r.finished_at)}
      </p>
      {r.error && <p className="error">{r.error}</p>}
      <ErrorText error={cancel.error ?? remove.error} />

      <Card title={active ? "Progress" : "Summary"}>
        <ProgressBar done={p.n_done} total={p.n_total} />
        <div className="stats" style={{ marginTop: "0.75rem" }}>
          <Stat v={`${p.n_done}/${p.n_total}`} l="items done" />
          <Stat v={fmtValue(r.accuracy, "fraction")} l={`accuracy so far (n=${p.n_scored})`} />
          <Stat v={String(p.n_unparseable)} l="unparseable replies" />
          <Stat v={String(p.n_errors)} l="request errors" />
        </div>
      </Card>

      {!active && r.metrics.length === 0 && (
        <EmptyState title="No metrics">
          {r.status === "completed" ? "This run requested no dimensions." : "The run did not reach the metrics stage."}
        </EmptyState>
      )}
      {Object.entries(byDim).map(([dim, metrics]) => (
        <Card key={dim} title={DIMENSION_TITLES[dim] ?? dim}>
          <div className="table-wrap">
            <table className="data">
              <thead><tr><th>Metric</th><th className="num">Value</th><th>Unit</th><th className="num">n</th><th>Notes</th></tr></thead>
              <tbody>
                {metrics.filter((m) => m.unit !== "table").map((m) => (
                  <tr key={m.name}>
                    <td className="mono">{m.name}</td>
                    <td className="num">{fmtValue(m.value, m.unit)}{ci(m)}</td>
                    <td className="muted">{m.unit ?? ""}</td>
                    <td className="num">{sampleSize(m) ?? "—"}</td>
                    <td className="muted">{note(m)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {dim === "accuracy" && <SubjectTable metrics={metrics} />}
        </Card>
      ))}

      {failed.data && (
        <Card title={`Failed items (${failed.data.total} wrong, unparseable or errored; showing ${failed.data.items.length})`}>
          {failed.data.items.length === 0 ? (
            <EmptyState title="Every scored item was correct" />
          ) : (
            <div className="table-wrap">
              <table className="data">
                <thead><tr><th>Item</th><th>Subject</th><th>Expected</th><th>Parsed</th><th>Response</th><th>Error</th><th>Prompt</th></tr></thead>
                <tbody>
                  {failed.data.items.map((it) => (
                    <tr key={it.id}>
                      <td className="mono">{it.item_id}</td>
                      <td>{it.subject ?? "—"}</td>
                      <td className="mono">{String(it.extra_json?.expected ?? "—")}</td>
                      <td className="mono">{it.parsed_answer ?? <span className="muted">unparseable</span>}</td>
                      <td className="mono" style={{ maxWidth: 260, overflowWrap: "anywhere" }}>{it.response ?? "—"}</td>
                      <td className="error" style={{ maxWidth: 260, overflowWrap: "anywhere" }}>{it.error ?? ""}</td>
                      <td><details className="prompt"><summary>show</summary><pre>{it.prompt}</pre></details></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      )}
    </div>
  );
}

function Stat({ v, l }: { v: string; l: string }) {
  return <div className="stat"><span className="v">{v}</span><span className="l">{l}</span></div>;
}

function ci(m: Metric): string {
  const x = m.extra_json ?? {};
  if (typeof x.ci95_low === "number" && typeof x.ci95_high === "number") {
    return ` (${(x.ci95_low * 100).toFixed(0)}–${(x.ci95_high * 100).toFixed(0)}%)`;
  }
  return "";
}

function note(m: Metric): string {
  const x = m.extra_json ?? {};
  if (m.value === null) return String(x.reason ?? x.note ?? x.error ?? "not measured");
  const parts: string[] = [];
  if (x.source) parts.push(`source: ${x.source}`);
  if (typeof x.concurrency === "number") parts.push(`concurrency ${x.concurrency}`);
  if (typeof x.temperature === "number") parts.push(`T=${x.temperature}`);
  if (typeof x.n_repeats === "number") parts.push(`${x.n_repeats} repeats`);
  if (typeof x.n_errors === "number" && x.n_errors > 0) parts.push(`${x.n_errors} errors`);
  if (x.tokenizer_class) parts.push(String(x.tokenizer_class));
  if (x.category) parts.push(String(x.category));
  return parts.join(" · ");
}

function SubjectTable({ metrics }: { metrics: Metric[] }) {
  const subj = metrics.find((m) => m.name === "accuracy_by_subject");
  const dist = metrics.find((m) => m.name === "answer_distribution");
  const rows = (subj?.extra_json?.rows as { subject: string; n: number; n_correct: number; accuracy: number }[] | undefined) ?? [];
  const counts = (dist?.extra_json?.counts as Record<string, number> | undefined) ?? {};
  return (
    <div className="grid-2" style={{ marginTop: "0.75rem" }}>
      <div>
        <h3>By subject <small>(n per subject is small; do not over-read)</small></h3>
        <div className="table-wrap" style={{ maxHeight: 260, overflowY: "auto" }}>
          <table className="data">
            <thead><tr><th>Subject</th><th className="num">correct</th><th className="num">n</th><th className="num">acc.</th></tr></thead>
            <tbody>{rows.map((r) => (
              <tr key={r.subject}><td>{r.subject}</td><td className="num">{r.n_correct}</td><td className="num">{r.n}</td><td className="num">{(r.accuracy * 100).toFixed(0)}%</td></tr>
            ))}</tbody>
          </table>
        </div>
      </div>
      <div>
        <h3>Answer-letter distribution <small>(position bias check)</small></h3>
        <table className="data">
          <thead><tr>{Object.keys(counts).map((k) => <th key={k} className="num">{k}</th>)}</tr></thead>
          <tbody><tr>{Object.values(counts).map((v, i) => <td key={i} className="num">{v}</td>)}</tr></tbody>
        </table>
      </div>
    </div>
  );
}

function groupBy<T>(xs: T[], key: (x: T) => string): Record<string, T[]> {
  return xs.reduce<Record<string, T[]>>((acc, x) => ((acc[key(x)] ??= []).push(x), acc), {});
}
