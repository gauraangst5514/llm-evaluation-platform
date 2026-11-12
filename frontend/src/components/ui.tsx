import type { ReactNode } from "react";
import type { RunStatus } from "../api/client";

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="empty">
      <strong>{title}</strong>
      {children}
    </div>
  );
}

export function StatusBadge({ status }: { status: RunStatus }) {
  return <span className={`badge ${status}`}>{status}</span>;
}

export function ProgressBar({ done, total }: { done: number; total: number }) {
  const pct = total ? Math.min(100, (100 * done) / total) : 0;
  return (
    <div className="progress" role="progressbar" aria-valuenow={done} aria-valuemax={total} title={`${done}/${total}`}>
      <span style={{ width: `${pct}%` }} />
    </div>
  );
}

export function ErrorText({ error }: { error: unknown }) {
  if (!error) return null;
  const msg = error instanceof Error ? error.message : String(error);
  return <p className="error">{msg}</p>;
}

export function Card({ title, right, children, foot }: { title?: ReactNode; right?: ReactNode; children: ReactNode; foot?: ReactNode }) {
  return (
    <section className="card">
      {(title || right) && (
        <div className="card-head">
          {title && <h3>{title}</h3>}
          {right}
        </div>
      )}
      {children}
      {foot && <div className="chart-foot">{foot}</div>}
    </section>
  );
}

export function Legend({ entries }: { entries: { label: string; color: string }[] }) {
  if (entries.length < 2) return null; // a single series is named by the title
  return (
    <div className="legend" aria-label="legend">
      {entries.map((e) => (
        <span key={e.label}>
          <i className="swatch" style={{ background: e.color }} />
          {e.label}
        </span>
      ))}
    </div>
  );
}
