import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import { useApp } from "../context.js";
import { Busy, Chip, Empty, ErrorBox, Field, Icon, ICONS, Rel, Section, Spinner, Switch, Tabs, useBusy, useLoad } from "../components/ui.jsx";
import { CiBar, CpuBadge, MemoryChip, Score, VerdictChip } from "../components/parts.jsx";
import { CATEGORIES, TASKS } from "../meta.js";
import { duration, interval, num, pct } from "../format.js";

function ScopePicker({ value, onChange, suites, includeStale, onStale }) {
  const { t } = useApp();
  return (
    <div className="flex flex-wrap items-end gap-2">
      <div style={{ width: 230 }}>
        <Field label={t("category")}>
          <select className="field" value={value.category} onChange={(e) => onChange({ category: e.target.value, suite: "" })}>
            <option value="">{t("all")}</option>
            <optgroup label={t("tasks")}>{TASKS.map((c) => <option key={`t${c}`} value={c}>{t(`task_${c}`)}</option>)}</optgroup>
            <optgroup label={t("suite_categories")}>{CATEGORIES.map((c) => <option key={`c${c}`} value={c}>{t(`cat_${c}`)}</option>)}</optgroup>
          </select>
        </Field>
      </div>
      <div style={{ width: 230 }}>
        <Field label={t("suite")}>
          <select className="field" value={value.suite} onChange={(e) => onChange({ category: "", suite: e.target.value })}>
            <option value="">{t("all")}</option>
            {suites.map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
          </select>
        </Field>
      </div>
      <label className="flex items-center gap-2" style={{ paddingBottom: 6 }}><Switch checked={includeStale} onChange={onStale} label={t("include_stale")} /><span>{t("include_stale")}</span></label>
    </div>
  );
}

function Leaderboard({ suites }) {
  const { t, lang, version } = useApp();
  const [scope, setScope] = useState({ category: "", suite: "" });
  const [stale, setStale] = useState(false);
  const { data, error, loading } = useLoad(() => api.call("leaderboard", { ...scope, include_stale: stale, limit: 200 }), [scope.category, scope.suite, stale, version]);
  const rows = data?.rows || [];
  return (
    <div className="space-y-3">
      <ScopePicker value={scope} onChange={setScope} suites={suites} includeStale={stale} onStale={setStale} />
      <ErrorBox error={error} />
      {loading && !data && <Spinner />}
      {data && <p className="help">{t("board_note")}</p>}
      {data && !rows.length && <Empty>{t("board_empty")}</Empty>}
      {rows.length > 0 && (
        <div className="panel scroll-x" style={{ padding: 0 }}>
          <table className="tbl">
            <thead>
              <tr><th className="r">#</th><th>{t("model")}</th><th style={{ minWidth: 170 }}>{t("score")} (95 %)</th><th className="r">{t("pass_rate")}</th><th className="r">n</th><th className="r">{t("decode")}</th><th className="r">TTFT</th><th>{t("memory")}</th><th>{t("categories")}</th><th>{t("measured")}</th></tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.contestant} style={r.stale ? { opacity: 0.65 } : undefined}>
                  <td className="r num"><b>{r.rank ?? "—"}</b></td>
                  <td style={{ overflowWrap: "anywhere", minWidth: 170 }}>
                    <b>{r.name}</b>
                    <div className="help">{[r.family, r.params_b ? `${num(r.params_b, 1, lang)} B` : "", r.quant].filter(Boolean).join(" · ")}</div>
                    {r.stale && <Chip className="chip-danger">{t("stale_model")}</Chip>}
                    {Object.keys(r.constraints || {}).length > 0 && (
                      <div className="mt-1 inline-flex flex-wrap gap-1" title={t("cst_hint")}>
                        {Object.entries(r.constraints).map(([k, v]) => (
                          <Chip key={k} className={v.pass_rate == null ? "chip-amber" : v.pass_rate < 0.5 ? "chip-danger" : ""} title={`${t("cst_title")}: n=${v.n}${v.unjudged ? ` · ${t("judge_pending_n", { n: v.unjudged })}` : ""}`}>
                            {t(`cst_${k}`)} {v.pass_rate == null ? "—" : pct(v.pass_rate, 0, lang)}{v.unjudged > 0 && <> · {t("judge_pending_n", { n: v.unjudged })}</>}
                          </Chip>
                        ))}
                      </div>
                    )}
                    {r.warnings?.map((w, i) => <div key={i} className="banner banner-warn mt-1">{t.msg(w)}</div>)}
                  </td>
                  <td><Score value={r.score} ci={r.ci} /><CiBar value={r.score} ci={r.ci} /></td>
                  <td className="r num">{pct(r.pass_rate, 0, lang)}<div className="help">{interval(r.pass_ci, lang)}</div></td>
                  <td className="r num">{r.n}{r.truncated > 0 && <div><Chip className={r.warnings?.length ? "chip-danger" : "chip-amber"} title={t("truncated_hint")}>{t("truncated_n", { n: r.truncated })}</Chip></div>}</td>
                  <td className="r num">{r.decode_tps != null ? `${num(r.decode_tps, 0, lang)} tok/s` : "—"}{r.speed_cpu && r.decode_tps != null && <> <CpuBadge /></>}{r.decode_tps_p90 != null && <div className="help">p90 {num(r.decode_tps_p90, 0, lang)}</div>}{r.decode_tps_cpu != null && <div className="help">{t("cpu_speed_also", { tps: num(r.decode_tps_cpu, 0, lang) })}</div>}</td>
                  <td className="r num">{duration(r.ttft_ms)}</td>
                  <td><MemoryChip model={r} /></td>
                  <td><span className="inline-flex flex-wrap gap-1">{Object.entries(r.by_category || {}).map(([k, v]) => <Chip key={k} className={v.truncated > 0 ? "chip-amber" : ""} title={`n=${v.n}${v.truncated > 0 ? ` · ${t("truncated_n", { n: v.truncated })}` : ""}`}>{t(`cat_${k}`)} {num(v.score, 2, lang)}</Chip>)}</span></td>
                  <td className="help"><Rel ts={r.last_ts} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {data?.unmeasured?.length > 0 && (
        <div className="help">{t("unmeasured")}: {data.unmeasured.map((u) => u.name || u).join(", ")}</div>
      )}
    </div>
  );
}

function Compare({ suites }) {
  const { t, lang, version } = useApp();
  const { data: models } = useLoad(() => api.call("models_list", { limit: 200 }), [version]);
  const [a, setA] = useState("");
  const [b, setB] = useState("");
  const [scope, setScope] = useState({ category: "", suite: "" });
  const [stale, setStale] = useState(false);
  const [busy, run] = useBusy();
  const [out, setOut] = useState(null);
  const list = models?.models || [];
  useEffect(() => {
    if (list.length >= 2 && !a && !b) { setA(list[0].id); setB(list[1].id); }
  }, [list.length]); // eslint-disable-line react-hooks/exhaustive-deps
  const go = () => run("compare", async () => { setOut(null); setOut(await api.call("compare", { a, b, ...scope, include_stale: stale, max_cases: 300 })); });
  const better = (out?.per_case || []).filter((c) => c.diff > 0);
  const worse = (out?.per_case || []).filter((c) => c.diff < 0);
  const nameOf = (s) => out?.[s]?.name || "";
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-end gap-2">
        <div style={{ width: 260 }}><Field label="A"><select className="field" value={a} onChange={(e) => setA(e.target.value)}>{list.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}</select></Field></div>
        <div style={{ width: 260 }}><Field label="B"><select className="field" value={b} onChange={(e) => setB(e.target.value)}>{list.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}</select></Field></div>
      </div>
      <ScopePicker value={scope} onChange={setScope} suites={suites} includeStale={stale} onStale={setStale} />
      <Busy className="btn btn-primary" busy={busy.compare} disabled={!a || !b || a === b} onClick={go}>{t("compare_go")}</Busy>
      {out && (
        <div className="space-y-3">
          <div className="panel space-y-2">
            <div className="flex flex-wrap items-center gap-2"><VerdictChip verdict={out.verdict} /><span className="num help">p = {out.p_value != null ? num(out.p_value, 4, lang) : "—"} · n = {out.n}</span></div>
            <p>{t(`cmp_${out.verdict}`, { a: out.a.name, b: out.b.name, diff: `${out.diff > 0 ? "+" : ""}${num(out.diff, 3, lang)}`, lo: num(out.diff_ci?.[0], 3, lang), hi: num(out.diff_ci?.[1], 3, lang), p: num(out.p_value, 4, lang), n: out.n })}</p>
            {out.warnings?.map((w, i) => <div key={i} className="banner banner-warn">{t.msg(w)}</div>)}
            {(out.only_a > 0 || out.only_b > 0) && <div className="help">{t("only_in", { a: out.only_a, b: out.only_b })}</div>}
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            {["a", "b"].map((side) => (
              <div key={side} className="panel space-y-1">
                <div className="help">{side.toUpperCase()}</div>
                <b style={{ overflowWrap: "anywhere" }}>{out[side].name}</b>
                <div><Score value={out[side].summary?.score} ci={out[side].summary?.ci} n={out[side].summary?.n} /></div>
                <CiBar value={out[side].summary?.score} ci={out[side].summary?.ci} />
                <div className="help num">{t("pass_rate")} {pct(out[side].summary?.pass_rate, 0, lang)} · {out[side].speed?.decode_tps_median != null ? `${num(out[side].speed.decode_tps_median, 0, lang)} tok/s` : "—"}{out[side].speed?.cpu && out[side].speed?.decode_tps_median != null ? " (CPU)" : ""} · TTFT {duration(out[side].speed?.ttft_ms_median)}</div>
              </div>
            ))}
          </div>
          <div className="flex flex-wrap gap-2">
            <Chip className="chip-ok">{t("wins_a", { n: out.wins })}</Chip><Chip className="chip-danger">{t("wins_b", { n: out.losses })}</Chip><Chip>{t("ties", { n: out.ties })}</Chip>
            {out.diff != null && <Chip>{t("diff")} {out.diff > 0 ? "+" : ""}{num(out.diff, 3, lang)} ({out.diff_ci ? `${num(out.diff_ci[0], 3, lang)} / ${num(out.diff_ci[1], 3, lang)}` : ""})</Chip>}
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            {[["a", better], ["b", worse]].map(([side, rows]) => (
              <Section key={side} title={t("better_in", { name: nameOf(side) })} count={rows.length}>
                {!rows.length ? <div className="help">—</div> : (
                  <div className="panel scroll-x" style={{ padding: 0, maxHeight: 360, overflowY: "auto" }}>
                    <table className="tbl"><tbody>
                      {rows.map((c) => <tr key={c.case_id}><td style={{ overflowWrap: "anywhere" }}>{c.title}</td><td className="r num">{num(c.a, 2, lang)} / {num(c.b, 2, lang)}</td></tr>)}
                    </tbody></table>
                  </div>
                )}
              </Section>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default function Clasificacion() {
  const { t, version } = useApp();
  const [tab, setTab] = useState("board");
  const { data } = useLoad(() => api.call("suites_list", {}), [version]);
  const suites = data?.suites || [];
  return (
    <div className="space-y-4">
      <h1>{t("nav_board")}</h1>
      <Tabs value={tab} onChange={setTab} tabs={[{ id: "board", label: t("leaderboard") }, { id: "compare", label: t("compare") }]} />
      {tab === "board" ? <Leaderboard suites={suites} /> : <Compare suites={suites} />}
    </div>
  );
}
