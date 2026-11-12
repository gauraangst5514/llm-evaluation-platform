import type { Metric, RunSummary } from "../api/client";

export const LANGUAGE_NAMES: Record<string, string> = { en: "English", hi: "Hindi", bn: "Bengali", ta: "Tamil", te: "Telugu" };
export const languageName = (code: string | null) => (code ? LANGUAGE_NAMES[code] ?? code : "—");

/** Fixed categorical slot per language, so a language keeps its colour on every chart. */
const LANGUAGE_ORDER = ["en", "hi", "bn", "ta", "te"];
export const languageSlot = (code: string | null) => {
  const i = LANGUAGE_ORDER.indexOf(code ?? "");
  return i === -1 ? 4 : i + 1; // slots 1..4; unknown languages share slot 4 (cap: see dataviz)
};

/** Approximate bits per weight, used only to ORDER quantization levels on an axis. */
const QUANT_RANK: Record<string, number> = {
  Q2_K: 2.6, Q3_K_S: 3.4, Q3_K_M: 3.7, Q3_K_L: 4.0, Q4_0: 4.5, Q4_K_S: 4.6, Q4_K_M: 4.8, Q5_0: 5.5,
  Q5_K_S: 5.5, Q5_K_M: 5.7, Q6_K: 6.6, Q8_0: 8.5, int8: 8, int4: 4, awq: 4, gptq: 4,
  fp16: 16, f16: 16, bf16: 16, fp32: 32, none: 16,
};
export const quantRank = (q: string | null) => QUANT_RANK[q ?? "none"] ?? QUANT_RANK[(q ?? "").toUpperCase()] ?? 99;
export const quantLabel = (q: string | null) => q ?? "unquantized";

export const modelLabel = (r: Pick<RunSummary, "model_name" | "quantization">) =>
  `${r.model_name} · ${quantLabel(r.quantization)}`;
export const runLabel = (r: RunSummary) => `#${r.id} ${modelLabel(r)} · ${languageName(r.language)}`;

export function fmtValue(value: number | null | undefined, unit: string | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  switch (unit) {
    case "fraction":
      return `${(value * 100).toFixed(1)}%`;
    case "ms":
      return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${value.toFixed(0)} ms`;
    case "bytes":
      return value >= 1e9 ? `${(value / 1e9).toFixed(2)} GB` : `${(value / 1e6).toFixed(0)} MB`;
    case "tokens/s":
      return `${value.toFixed(1)} tok/s`;
    case "tokens/word":
    case "tokens/char":
      return value.toFixed(3);
    default:
      return value.toFixed(3);
  }
}

/** Display unit for chart axes (bytes become MB). */
export const axisUnit = (unit: string | null) => (unit === "bytes" ? "MB" : unit === "fraction" ? "%" : unit ?? "");
export const toAxisValue = (value: number | null, unit: string | null) =>
  value === null ? null : unit === "bytes" ? value / 1e6 : unit === "fraction" ? value * 100 : value;

export const sampleSize = (m: Metric | undefined): number | null => {
  const x = m?.extra_json ?? {};
  for (const k of ["n", "n_items", "n_texts", "n_samples"]) if (typeof x[k] === "number") return x[k] as number;
  return null;
};

export const fmtDate = (iso: string | null) => (iso ? new Date(iso).toLocaleString() : "—");

export function duration(start: string | null, end: string | null): string {
  if (!start) return "—";
  const ms = (end ? new Date(end).getTime() : Date.now()) - new Date(start).getTime();
  return ms < 60_000 ? `${(ms / 1000).toFixed(0)} s` : `${(ms / 60_000).toFixed(1)} min`;
}

/** Headline metrics shown as grouped bars on the compare page, in display order. */
export const HEADLINE_METRICS: { name: string; title: string; higherIsBetter: boolean | null }[] = [
  { name: "accuracy", title: "Accuracy", higherIsBetter: true },
  { name: "ttft_ms_p50", title: "Time to first token (P50)", higherIsBetter: false },
  { name: "latency_ms_p95", title: "Total latency (P95)", higherIsBetter: false },
  { name: "decode_tokens_per_s_p50", title: "Decode throughput (P50)", higherIsBetter: true },
  { name: "peak_rss_bytes", title: "Peak server RSS", higherIsBetter: false },
  { name: "model_file_bytes", title: "Model file size", higherIsBetter: false },
  { name: "fertility", title: "Tokenizer fertility", higherIsBetter: false },
  { name: "roundtrip_pass_rate", title: "Tokenizer round-trip pass rate", higherIsBetter: true },
  { name: "json_valid_rate", title: "JSON-schema valid rate", higherIsBetter: true },
  { name: "consistency_unanimous_rate", title: "Consistency at T=0.7 (unanimous)", higherIsBetter: true },
];

export const DIMENSION_TITLES: Record<string, string> = {
  accuracy: "Task accuracy",
  latency: "Latency & throughput",
  resources: "Resource usage",
  tokenizer: "Tokenizer efficiency",
  reliability: "Output reliability",
};
