import React from "react";
import { useApp } from "../context.js";
import { RUN_STATE } from "../meta.js";
import { gb, interval, num, score } from "../format.js";
import { Chip } from "./ui.jsx";

// A score with its 95 % interval drawn on a 0..1 axis: the band is the interval, the mark the mean.
export function CiBar({ value, ci, width = 160 }) {
  const { t, lang } = useApp();
  if (value === null || value === undefined) return <span className="help">—</span>;
  const lo = Math.max(0, Math.min(1, ci?.[0] ?? value));
  const hi = Math.max(0, Math.min(1, ci?.[1] ?? value));
  const v = Math.max(0, Math.min(1, value));
  const label = `${t("score")} ${score(value, lang)}${ci ? ` (${interval(ci, lang)})` : ""}`;
  return (
    <svg className="ci" viewBox="0 0 100 16" preserveAspectRatio="none" style={{ maxWidth: width }} role="img" aria-label={label}>
      <title>{label}</title>
      <rect className="track" x="0" y="6" width="100" height="4" rx="2" />
      <rect className="band" x={lo * 100} y="3" width={Math.max(1, (hi - lo) * 100)} height="10" rx="2" />
      <line className="mark" x1={v * 100} x2={v * 100} y1="1" y2="15" />
    </svg>
  );
}

export function Progress({ done, total, tone }) {
  const pct = total ? Math.min(100, (100 * done) / total) : 0;
  return (
    <div className={`bar ${tone ? `bar-${tone}` : ""}`} role="progressbar" aria-valuemin={0} aria-valuemax={total || 0} aria-valuenow={done || 0}>
      <span style={{ width: `${pct}%` }} />
    </div>
  );
}

export function StateChip({ state }) {
  const { t } = useApp();
  return <Chip className={RUN_STATE[state] || ""}>{t(`state_${state}`)}</Chip>;
}

// Marks a speed that was measured on the processor: it is not comparable with the speeds measured on a GPU.
export function CpuBadge() {
  const { t } = useApp();
  return <Chip className="chip-amber" title={t("cpu_badge_hint")}>CPU</Chip>;
}

export function Score({ value, ci, n }) {
  const { lang } = useApp();
  if (value === null || value === undefined) return <span className="help">—</span>;
  return (
    <span className="num">
      <b>{score(value, lang)}</b>
      {ci && <span className="help"> [{interval(ci, lang)}]</span>}
      {n !== undefined && n !== null && <span className="help"> · n={n}</span>}
    </span>
  );
}

// One strip of cards, one per GPU: memory bar, allowed or reserved, leases.
export function GpuStrip({ gpus }) {
  const { t, lang } = useApp();
  const list = gpus?.gpus || [];
  if (!list.length) return <div className="help">{t("gpu_none")}</div>;
  return (
    <div className="gpu-strip">
      {list.map((g) => {
        const used = g.total_mb ? Math.min(100, (100 * g.used_mb) / g.total_mb) : 0;
        return (
          <div key={g.index} className={`gpu ${g.allowed ? "" : "gpu-reserved"}`}>
            <div className="flex items-center gap-2">
              <b>GPU {g.index}</b>
              <Chip className={g.allowed ? "chip-ok" : "chip-amber"}>{g.allowed ? t("gpu_allowed") : t("gpu_reserved")}</Chip>
            </div>
            <div className="help trunc" title={g.name}>{g.name}</div>
            <div className="bar" role="progressbar" aria-valuenow={Math.round(used)} aria-valuemin={0} aria-valuemax={100} aria-label={`GPU ${g.index}`}><span style={{ width: `${used}%` }} /></div>
            <div className="help num">{t("gpu_free", { free: gb(g.free_mb, lang), total: gb(g.total_mb, lang) })}</div>
            {(g.leases || []).map((l, i) => <div key={i} className="help trunc">{l.owner || l.who || "—"} · {gb(l.mb ?? l.vram_mb, lang)}</div>)}
          </div>
        );
      })}
    </div>
  );
}

export function MemoryChip({ model }) {
  const { t, lang } = useApp();
  const value = model.memory_gb;
  if (value === null || value === undefined) return <span className="help">—</span>;
  const fits = model.fits_16gb ?? model.fits_one_16gb;
  return (
    <span className="num" title={model.memory_method ? t(`mem_${model.memory_method}`) : ""}>
      {num(value, 1, lang)} GB {fits !== undefined && <Chip className={fits ? "chip-ok" : "chip-amber"}>{fits ? t("fits16") : t("nofits16")}</Chip>}
    </span>
  );
}

export function VerdictChip({ verdict }) {
  const { t } = useApp();
  const cls = { better: "chip-ok", worse: "chip-danger", no_clear_difference: "chip-amber" }[verdict] || "";
  return <Chip className={cls}>{t(`verdict_${verdict}`)}</Chip>;
}

export function ModelName({ model }) {
  return (
    <span className="min-w-0">
      <b style={{ overflowWrap: "anywhere" }}>{model.name}</b>
    </span>
  );
}
