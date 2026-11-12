import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Label,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { languageName, languageSlot } from "../lib/format";

export const seriesColor = (slot: number) => `var(--series-${Math.min(Math.max(slot, 1), 4)})`;
export const languageColor = (code: string | null) => seriesColor(languageSlot(code));

const axisStyle = { fontSize: 11, fill: "var(--text-secondary)" };
const gridStyle = { stroke: "var(--grid)" };

export interface BarDatum {
  key: string;
  label: string; // x-axis label (model · quant)
  language: string;
  value: number | null;
  display: string; // formatted value with unit
  n: number | null;
}

/** One bar per run, coloured by language, 4px rounded tops anchored at the baseline. */
export function RunBarChart({ data, unit, height = 220 }: { data: BarDatum[]; unit: string; height?: number }) {
  const measurable = data.filter((d) => d.value !== null);
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: 4, bottom: 4 }} barCategoryGap="25%">
        <CartesianGrid vertical={false} {...gridStyle} />
        <XAxis dataKey="label" tick={axisStyle} interval={0} tickLine={false} axisLine={{ stroke: "var(--border)" }} />
        <YAxis tick={axisStyle} tickLine={false} axisLine={false} width={48}>
          <Label value={unit} angle={-90} position="insideLeft" style={{ ...axisStyle, textAnchor: "middle" }} />
        </YAxis>
        <Tooltip cursor={{ fill: "var(--surface-2)" }} content={<BarTip />} />
        <Bar dataKey="value" radius={[4, 4, 0, 0]} maxBarSize={48} isAnimationActive={false}>
          {data.map((d) => (
            <Cell key={d.key} fill={languageColor(d.language)} />
          ))}
        </Bar>
        {measurable.length === 0 && (
          <text x="50%" y="45%" textAnchor="middle" style={{ fontSize: 12, fill: "var(--text-muted)" }}>
            not measured for these runs
          </text>
        )}
      </BarChart>
    </ResponsiveContainer>
  );
}

function BarTip({ active, payload }: { active?: boolean; payload?: { payload: BarDatum }[] }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="tooltip">
      <div className="t">{d.label}</div>
      <div>{languageName(d.language)}</div>
      <div>{d.display}{d.n !== null ? ` · n=${d.n}` : ""}</div>
    </div>
  );
}

export interface LinePoint {
  quant: string;
  [language: string]: number | string | null;
}

/** Accuracy (%) vs quantization level; one line per language. */
export function QuantLineChart({ data, languages, height = 260 }: { data: LinePoint[]; languages: string[]; height?: number }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 16, left: 4, bottom: 4 }}>
        <CartesianGrid vertical={false} {...gridStyle} />
        <XAxis dataKey="quant" tick={axisStyle} tickLine={false} axisLine={{ stroke: "var(--border)" }}>
          <Label value="quantization (fewer bits →)" position="insideBottom" offset={-2} style={axisStyle} />
        </XAxis>
        <YAxis domain={[0, 100]} tick={axisStyle} tickLine={false} axisLine={false} width={44}>
          <Label value="accuracy %" angle={-90} position="insideLeft" style={{ ...axisStyle, textAnchor: "middle" }} />
        </YAxis>
        <Tooltip
          content={<GenericTip fmt={(k, v) => `${languageName(k)}: ${typeof v === "number" ? v.toFixed(1) + "%" : "—"}`} />}
        />
        {languages.map((lang) => (
          <Line
            key={lang}
            type="linear"
            dataKey={lang}
            stroke={languageColor(lang)}
            strokeWidth={2}
            dot={{ r: 4, strokeWidth: 2, stroke: "var(--surface-1)", fill: languageColor(lang) }}
            activeDot={{ r: 6 }}
            connectNulls={false}
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

function GenericTip({
  active,
  payload,
  label,
  fmt,
}: {
  active?: boolean;
  payload?: { dataKey: string; value: unknown }[];
  label?: string;
  fmt: (key: string, value: unknown) => string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="tooltip">
      <div className="t">{label}</div>
      {payload.map((p) => (
        <div key={String(p.dataKey)}>{fmt(String(p.dataKey), p.value)}</div>
      ))}
    </div>
  );
}

export interface ScatterPoint {
  language: string;
  quant: string;
  model: string;
  fertility: number;
  drop: number; // percentage points vs the highest-precision run of the same model+language
  n: number | null;
}

/** Fertility vs accuracy drop; one series per language, ≥8px markers with a surface ring. */
export function FertilityScatter({ points, height = 280 }: { points: ScatterPoint[]; height?: number }) {
  const languages = [...new Set(points.map((p) => p.language))];
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ top: 8, right: 16, left: 4, bottom: 8 }}>
        <CartesianGrid {...gridStyle} />
        <XAxis type="number" dataKey="fertility" name="fertility" tick={axisStyle} tickLine={false} domain={["auto", "auto"]}>
          <Label value="fertility (tokens per word)" position="insideBottom" offset={-4} style={axisStyle} />
        </XAxis>
        <YAxis type="number" dataKey="drop" name="accuracy drop" tick={axisStyle} tickLine={false} axisLine={false} width={44}>
          <Label value="accuracy drop (pp)" angle={-90} position="insideLeft" style={{ ...axisStyle, textAnchor: "middle" }} />
        </YAxis>
        <ZAxis range={[80, 80]} />
        <Tooltip cursor={{ strokeDasharray: "3 3" }} content={<ScatterTip />} />
        {languages.map((lang) => (
          <Scatter
            key={lang}
            data={points.filter((p) => p.language === lang)}
            fill={languageColor(lang)}
            stroke="var(--surface-1)"
            strokeWidth={2}
            isAnimationActive={false}
          />
        ))}
      </ScatterChart>
    </ResponsiveContainer>
  );
}

function ScatterTip({ active, payload }: { active?: boolean; payload?: { payload: ScatterPoint }[] }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="tooltip">
      <div className="t">{p.model} · {p.quant}</div>
      <div>{languageName(p.language)}</div>
      <div>fertility {p.fertility.toFixed(2)} tok/word</div>
      <div>drop {p.drop.toFixed(1)} pp{p.n !== null ? ` · n=${p.n}` : ""}</div>
    </div>
  );
}
