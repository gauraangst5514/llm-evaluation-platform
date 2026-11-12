import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api, type CompareResponse, type RunSummary } from "../api/client";
import { FertilityScatter, QuantLineChart, RunBarChart, languageColor, type BarDatum, type LinePoint, type ScatterPoint } from "../components/charts";
import { Card, EmptyState, ErrorText, Legend } from "../components/ui";
import { HEADLINE_METRICS, axisUnit, fmtValue, languageName, languageSlot, modelLabel, quantLabel, quantRank, toAxisValue } from "../lib/format";

type MetricCell = { value: number | null; unit: string | null; extra: Record<string, unknown> | null };
type Pivot = Map<number, Map<string, MetricCell>>; // run id -> metric name -> cell

function pivot(data: CompareResponse): Pivot {
  const out: Pivot = new Map(data.runs.map((r) => [r.id, new Map()]));
  for (const row of data.rows) {
    for (const c of row.cells) {
      if (c.value === null && c.extra_json === null) continue; // metric absent for this run
      out.get(c.run_id)?.set(row.name, { value: c.value, unit: row.unit, extra: c.extra_json });
    }
  }
  return out;
}

const n_of = (cell?: MetricCell): number | null => {
  const x = cell?.extra ?? {};
  for (const k of ["n", "n_items", "n_texts", "n_samples"]) if (typeof x[k] === "number") return x[k] as number;
  return null;
};

export function ComparePage() {
  const [params, setParams] = useSearchParams();
  const ids = (params.get("runs") ?? "").split(",").map(Number).filter((n) => Number.isInteger(n) && n > 0);
  const all = useQuery({ queryKey: ["runs"], queryFn: api.runs });
  const cmp = useQuery({ queryKey: ["compare", ids], queryFn: () => api.compare(ids), enabled: ids.length >= 1 });
  const completed = (all.data ?? []).filter((r) => r.status === "completed");

  const toggle = (id: number) => {
    const next = ids.includes(id) ? ids.filter((x) => x !== id) : ids.length < 6 ? [...ids, id] : ids;
    setParams(next.length ? { runs: next.join(",") } : {});
  };

  return (
    <div className="page">
      <h1>Compare runs</h1>
      <Card title={`Pick 2 to 6 completed runs (${ids.length} selected)`}>
        <ErrorText error={all.error} />
        {all.data && completed.length === 0 && (
          <EmptyState title="No completed runs to compare">Start runs from the <Link to="/">Runs</Link> page.</EmptyState>
        )}
        <div className="pill-row">
          {completed.map((r) => (
            <button key={r.id} className={`pill ${ids.includes(r.id) ? "on" : ""}`} onClick={() => toggle(r.id)}>
              #{r.id} {modelLabel(r)} · {languageName(r.language)}
            </button>
          ))}
        </div>
      </Card>
      {ids.length < 2 && completed.length > 0 && <EmptyState title="Select at least two runs" />}
      <ErrorText error={cmp.error} />
      {cmp.data && ids.length >= 2 && <CompareBody data={cmp.data} />}
    </div>
  );
}

function CompareBody({ data }: { data: CompareResponse }) {
  const pv = useMemo(() => pivot(data), [data]);
  const runs = data.runs;
  // Fixed language order so a language keeps its colour and position on every chart.
  const languages = [...new Set(runs.map((r) => r.language))].sort((a, b) => languageSlot(a) - languageSlot(b));
  const legend = languages.map((l) => ({ label: languageName(l), color: languageColor(l) }));

  return (
    <>
      <Legend entries={legend} />
      <h2>Headline metrics by dimension</h2>
      <p className="muted">One bar per run, coloured by language. Hover for the exact value and sample size; bars are missing where a metric was not measured.</p>
      <div className="grid-2">
        {HEADLINE_METRICS.map((hm) => {
          const cells = runs.map((r) => pv.get(r.id)?.get(hm.name));
          if (cells.every((c) => !c)) return null;
          const unit = cells.find((c) => c)?.unit ?? null;
          const bars: BarDatum[] = runs.map((r, i) => ({
            key: String(r.id),
            label: `#${r.id} ${quantLabel(r.quantization)}`,
            language: r.language,
            value: toAxisValue(cells[i]?.value ?? null, unit),
            display: fmtValue(cells[i]?.value, unit),
            n: n_of(cells[i]),
          }));
          const ns = bars.map((b) => b.n).filter((n): n is number => n !== null);
          const nText = ns.length ? (new Set(ns).size === 1 ? `n=${ns[0]} per run` : `n=${Math.min(...ns)}–${Math.max(...ns)} per run`) : "";
          return (
            <Card key={hm.name} title={hm.title} right={<small className="mono">{hm.name}</small>}
              foot={`${axisUnit(unit)}${nText ? ` · ${nText}` : ""}${hm.higherIsBetter === null ? "" : hm.higherIsBetter ? " · higher is better" : " · lower is better"}`}>
              <RunBarChart data={bars} unit={axisUnit(unit)} />
            </Card>
          );
        })}
      </div>

      <h2>Accuracy by language</h2>
      <AccuracyTable runs={runs} pv={pv} languages={languages} />

      <h2>Accuracy vs quantization</h2>
      <QuantSection runs={runs} pv={pv} languages={languages} />

      <h2>Fertility vs accuracy drop</h2>
      <ScatterSection runs={runs} pv={pv} />
    </>
  );
}

function AccuracyTable({ runs, pv, languages }: { runs: RunSummary[]; pv: Pivot; languages: string[] }) {
  const models = [...new Map(runs.map((r) => [modelLabel(r), r])).values()].sort(
    (a, b) => a.model_name.localeCompare(b.model_name) || quantRank(b.quantization) - quantRank(a.quantization),
  );
  return (
    <Card foot="Accuracy with 95% Wilson interval and n scored items. Blank cells: no run for that model and language in this selection.">
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr><th>Model · quant</th>{languages.map((l) => <th key={l} className="num">{languageName(l)}</th>)}</tr>
          </thead>
          <tbody>
            {models.map((m) => (
              <tr key={modelLabel(m)}>
                <td>{modelLabel(m)}</td>
                {languages.map((l) => {
                  const r = runs.find((x) => modelLabel(x) === modelLabel(m) && x.language === l);
                  const c = r ? pv.get(r.id)?.get("accuracy") : undefined;
                  if (!r || !c) return <td key={l} className="num muted">—</td>;
                  const lo = c.extra?.ci95_low as number | undefined;
                  const hi = c.extra?.ci95_high as number | undefined;
                  return (
                    <td key={l} className="num">
                      <Link to={`/runs/${r.id}`}>{fmtValue(c.value, "fraction")}</Link>
                      <br />
                      <small>{lo !== undefined && hi !== undefined ? `${(lo * 100).toFixed(0)}–${(hi * 100).toFixed(0)}% · ` : ""}n={n_of(c) ?? "?"}</small>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function QuantSection({ runs, pv, languages }: { runs: RunSummary[]; pv: Pivot; languages: string[] }) {
  // One model family = one model name. Need ≥2 quantization levels for a line to mean anything.
  const families = [...new Set(runs.map((r) => r.model_name))];
  const sections = families.map((name) => {
    const fam = runs.filter((r) => r.model_name === name);
    const quants = [...new Set(fam.map((r) => r.quantization))].sort((a, b) => quantRank(b) - quantRank(a));
    const data: LinePoint[] = quants.map((q) => {
      const pt: LinePoint = { quant: quantLabel(q) };
      for (const l of languages) {
        const r = fam.find((x) => x.quantization === q && x.language === l);
        const v = r ? pv.get(r.id)?.get("accuracy")?.value ?? null : null;
        pt[l] = v === null ? null : v * 100;
      }
      return pt;
    });
    return { name, quants, data };
  });
  const usable = sections.filter((s) => s.quants.length >= 2);
  if (usable.length === 0) {
    return (
      <EmptyState title="Needs at least two quantization levels of the same model">
        Selected: {sections.map((s) => `${s.name} (${s.quants.map(quantLabel).join(", ")})`).join("; ")}.
      </EmptyState>
    );
  }
  return (
    <div className="grid-2">
      {usable.map((s) => (
        <Card key={s.name} title={s.name} foot="x-axis ordered from most to fewest bits per weight; points are absent where no run exists. Accuracy %, n per run as in the table above.">
          <Legend entries={languages.map((l) => ({ label: languageName(l), color: languageColor(l) }))} />
          <QuantLineChart data={s.data} languages={languages} />
        </Card>
      ))}
    </div>
  );
}

function ScatterSection({ runs, pv }: { runs: RunSummary[]; pv: Pivot }) {
  // Drop = accuracy(highest-precision run of same model+language) − accuracy(this run), in pp.
  const points: ScatterPoint[] = [];
  const missingFertility: string[] = [];
  const groups = new Map<string, RunSummary[]>();
  for (const r of runs) (groups.get(`${r.model_name}|${r.language}`) ?? groups.set(`${r.model_name}|${r.language}`, []).get(`${r.model_name}|${r.language}`)!).push(r);
  for (const grp of groups.values()) {
    if (grp.length < 2) continue;
    const base = [...grp].sort((a, b) => quantRank(b.quantization) - quantRank(a.quantization))[0];
    const baseAcc = pv.get(base.id)?.get("accuracy")?.value;
    if (baseAcc === null || baseAcc === undefined) continue;
    for (const r of grp) {
      if (r.id === base.id) continue;
      const acc = pv.get(r.id)?.get("accuracy");
      const fert = pv.get(r.id)?.get("fertility");
      if (!fert || fert.value === null) { missingFertility.push(`#${r.id}`); continue; }
      if (!acc || acc.value === null) continue;
      points.push({ language: r.language, quant: quantLabel(r.quantization), model: r.model_name,
        fertility: fert.value, drop: (baseAcc - acc.value) * 100, n: n_of(acc) });
    }
  }
  if (points.length === 0) {
    return (
      <EmptyState title="Needs the same model at two or more quantization levels, per language, with tokenizer metrics">
        Each point is one (model, quantization, language) run compared with the highest-precision run of the same model and language.
        {missingFertility.length ? ` Runs without fertility (no hf_repo on the model?): ${missingFertility.join(", ")}.` : ""}
      </EmptyState>
    );
  }
  const languages = [...new Set(points.map((p) => p.language))];
  return (
    <Card foot={`${points.length} points. Drop in percentage points relative to the highest-precision run of the same model and language. With n≈${points[0].n ?? "?"} per run, differences under ~10 pp are within noise.`}>
      <Legend entries={languages.map((l) => ({ label: languageName(l), color: languageColor(l) }))} />
      <FertilityScatter points={points} />
    </Card>
  );
}
