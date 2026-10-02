// Numbers, dates and small helpers. es-ES by default, en-GB when English is chosen.
export const locale = (lang) => (lang === "en" ? "en-GB" : "es-ES");

export function num(value, digits = 0, lang = "es") {
  if (value === null || value === undefined || value === "" || Number.isNaN(Number(value))) return "—";
  return Number(value).toLocaleString(locale(lang), { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export const score = (value, lang = "es") => num(value, 2, lang);

export function interval(ci, lang = "es") {
  if (!Array.isArray(ci) || ci.length < 2) return "";
  return `${num(ci[0], 2, lang)}–${num(ci[1], 2, lang)}`;
}

export function clock(ts, lang) {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleString(locale(lang), { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }).replace(",", "");
}

export function shortClock(ts, lang) {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleString(locale(lang), { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
}

// "hace 5 min" / "dentro de 2 h"
export function rel(ts, lang, nowMs = Date.now()) {
  if (!ts) return "—";
  const diff = ts * 1000 - nowMs;
  const abs = Math.abs(diff);
  const formatter = new Intl.RelativeTimeFormat(locale(lang), { numeric: "auto", style: "short" });
  if (abs < 45_000) return formatter.format(0, "second");
  const units = [["day", 86_400_000], ["hour", 3_600_000], ["minute", 60_000]];
  for (const [unit, size] of units) {
    if (abs >= size || unit === "minute") return formatter.format(Math.round(diff / size), unit);
  }
  return "—";
}

export function duration(ms) {
  if (ms === null || ms === undefined || Number.isNaN(Number(ms))) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 90_000) return `${(ms / 1000).toFixed(1)} s`;
  return `${Math.round(ms / 60_000)} min`;
}

/** A wait of ``seconds`` as "45 s", "5 min 12 s" or "1 h 02 min". */
export function elapsed(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(Number(seconds))) return "—";
  const total = Math.max(0, Math.round(Number(seconds)));
  if (total < 60) return `${total} s`;
  const minutes = Math.floor(total / 60);
  if (total < 3600) return `${minutes} min ${String(total % 60).padStart(2, "0")} s`;
  return `${Math.floor(minutes / 60)} h ${String(minutes % 60).padStart(2, "0")} min`;
}

export function gb(mb, lang = "es") {
  if (mb === null || mb === undefined) return "—";
  return `${num(mb / 1024, 1, lang)} GB`;
}

export function fileSize(bytes) {
  if (!bytes) return "";
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 / 1024).toFixed(0)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

export const splitList = (text) => (text || "").split(/[\n,;]+/).map((s) => s.trim()).filter(Boolean);
export const pct = (value, digits = 0, lang = "es") => (value === null || value === undefined ? "—" : `${num(value * 100, digits, lang)} %`);
