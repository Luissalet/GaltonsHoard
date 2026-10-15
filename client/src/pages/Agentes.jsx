import React, { useMemo, useState } from "react";
import { api } from "../api.js";
import { useApp } from "../context.js";
import { Busy, Chip, Empty, ErrorBox, Failure, Field, Icon, ICONS, Rel, Section, Spinner, useBusy, useLoad } from "../components/ui.jsx";
import { Progress, StateChip } from "../components/parts.jsx";
import { usePoll } from "../components/hooks.js";
import { num, pct } from "../format.js";

const ACTIVE = new Set(["queued", "running"]);
const OUTCOME_CHIP = { pass: "chip-ok", fail: "chip-danger", none: "" };
const VERDICT_CHIP = { first: "chip-ok", tied_first: "chip-accent", behind: "chip-danger", alone: "", not_measured: "" };

// The colour of a cell: from the danger colour at 0 % to the success colour at 100 %, soft enough to read the text over it in any theme.
const heat = (p) => `color-mix(in srgb, color-mix(in srgb, var(--ok) ${Math.round(p * 100)}%, var(--danger)) 34%, transparent)`;

// ------------------------------------------------------------------ the form
function NewCampaign() {
  const { t, version, changed, notify } = useApp();
  const catalog = useLoad(() => api.call("agentbench_catalog", {}), [version]);
  const [benches, setBenches] = useState(null);
  const [harnesses, setHarnesses] = useState(null);
  const [splits, setSplits] = useState(new Set(["dev", "test"]));
  const [form, setForm] = useState({ k: 1, label: "", slots: "" });
  const [busy, run] = useBusy();
  const data = catalog.data;
  const allBenches = data?.benchmarks || [];
  const allHarnesses = (data?.harnesses || []).filter((h) => h.enabled);
  const chosenB = benches || new Set(allBenches.filter((b) => b.downloaded).map((b) => b.id));
  const chosenH = harnesses || new Set(allHarnesses.map((h) => h.name));
  const toggle = (set, setter, id) => setter(() => { const next = new Set(set); next.has(id) ? next.delete(id) : next.add(id); return next; });
  const trials = useMemo(() => allBenches.filter((b) => chosenB.has(b.id)).reduce((acc, b) => acc + (splits.has("dev") ? b.dev : 0) + (splits.has("test") ? b.test : 0), 0)
    * chosenH.size * Math.max(1, Number(form.k) || 1), [allBenches, chosenB, chosenH, splits, form.k]);
  const ready = chosenB.size > 0 && chosenH.size > 0 && splits.size > 0 && data?.defaults?.model;
  const start = () => run("start", async () => {
    const r = await api.call("agentbench_start", { benchmarks: [...chosenB], harnesses: [...chosenH], k: Math.max(1, Number(form.k) || 1), splits: [...splits],
      label: form.label, slots: Number(form.slots) || 0 });
    notify(t("ab_started"));
    changed();
    window.location.hash = `#/agentes/${r.id}`;
  });
  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }));
  const box = (checked, onChange, label, disabled) => (
    <label className="flex items-start gap-2" style={{ cursor: disabled ? "not-allowed" : "pointer", opacity: disabled ? 0.6 : 1 }}>
      <input type="checkbox" checked={checked} disabled={disabled} onChange={onChange} style={{ marginTop: 4 }} /><span className="min-w-0">{label}</span>
    </label>
  );
  if (catalog.error) return <ErrorBox error={catalog.error} />;
  if (!data) return <Spinner />;
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Section title={t("ab_benchmarks")} count={chosenB.size || undefined}>
        <div className="panel space-y-2">
          {allBenches.map((b) => (
            <React.Fragment key={b.id}>{box(chosenB.has(b.id), () => toggle(chosenB, setBenches, b.id),
              <span><b>{b.title}</b> <span className="help num">{t("ab_subset", { dev: b.dev, test: b.test, pool: b.pool })}</span>
                {!b.downloaded && <Chip className="chip-amber">{t("ab_not_downloaded")}</Chip>}{b.needs_mcp && <Chip>{t("ab_needs_mcp")}</Chip>}
                <span className="help block">{b.note}</span></span>)}</React.Fragment>
          ))}
        </div>
      </Section>
      <Section title={t("ab_harnesses")} count={chosenH.size || undefined}>
        <div className="panel space-y-1">
          {!allHarnesses.length && <div className="help">{t("ab_no_harnesses")}</div>}
          {allHarnesses.map((h) => (
            <React.Fragment key={h.name}>{box(chosenH.has(h.name), () => toggle(chosenH, setHarnesses, h.name),
              <span><b>{h.name}</b> <span className="help mono" style={{ overflowWrap: "anywhere" }}>{h.agent}</span>{!h.mcp && <> <Chip>{t("ab_no_mcp")}</Chip></>}</span>)}</React.Fragment>
          ))}
          {(data.unreadable_lines || []).map((line, i) => <div key={i} className="banner banner-warn">{t("ab_bad_line")}: <span className="mono">{line}</span></div>)}
        </div>
      </Section>
      <div className="lg:col-span-2 space-y-3">
        <Section title={t("run_settings")}>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Field label={t("ab_k")} hint={t("ab_k_hint")}><input className="field" type="number" min="1" max="10" value={form.k} onChange={set("k")} /></Field>
            <Field label={t("ab_slots")} hint={t("ab_slots_hint")}><input className="field" type="number" min="1" max="64" placeholder={String(data.defaults.slots)} value={form.slots} onChange={set("slots")} /></Field>
            <Field label={t("ab_label")}><input className="field" value={form.label} onChange={set("label")} /></Field>
            <div role="group" aria-label={t("ab_splits")}><span className="label">{t("ab_splits")}</span>
              {box(splits.has("dev"), () => toggle(splits, setSplits, "dev"), t("ab_split_dev"))}
              {box(splits.has("test"), () => toggle(splits, setSplits, "test"), t("ab_split_test"))}
            </div>
          </div>
          <div className="help num">{t("ab_model_line", { model: data.defaults.model || "—", url: data.defaults.base_url || "—" })}</div>
          {!data.runner_found && <div className="banner banner-warn">{t("ab_no_runner", { runner: data.runner })}</div>}
        </Section>
        <div className="help num">{t("ab_plan", { trials })}</div>
        <Busy className="btn btn-primary" busy={busy.start} disabled={!ready} onClick={start}><Icon d={ICONS.run} size={14} />{t("ab_start")}</Busy>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ history
function History() {
  const { t, version } = useApp();
  const { data, error, loading, reload } = useLoad(() => api.call("agentbench_list", { limit: 30 }), [version]);
  const rows = data?.campaigns || [];
  usePoll(reload, 5000, rows.some((c) => ACTIVE.has(c.state)));
  return (
    <Section title={t("history")} count={rows.length || undefined}>
      <ErrorBox error={error} />
      {loading && !data && <Spinner />}
      {data && !rows.length && <Empty>{t("ab_none")}</Empty>}
      {rows.length > 0 && (
        <div className="panel scroll-x" style={{ padding: 0 }}>
          <table className="tbl">
            <thead><tr><th>{t("run")}</th><th>{t("status")}</th><th>{t("progress")}</th><th>{t("model")}</th><th>{t("created")}</th></tr></thead>
            <tbody>
              {rows.map((c) => (
                <tr key={c.id}>
                  <td style={{ overflowWrap: "anywhere" }}><a href={`#/agentes/${c.id}`}>{c.label || c.name}</a><div className="help">{c.benchmarks.join(", ")} · {c.harnesses.length} {t("ab_harnesses_n")}</div></td>
                  <td><StateChip state={c.state} /></td>
                  <td style={{ minWidth: 140 }}><div className="help num">{c.trials.verified}/{c.trials.expected}{c.trials.none ? ` · None ${c.trials.none}` : ""}</div><Progress done={c.trials.verified} total={c.trials.expected} /></td>
                  <td className="help" style={{ overflowWrap: "anywhere" }}>{c.model}</td>
                  <td className="help"><Rel ts={c.created_ts} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Section>
  );
}

// ------------------------------------------------------------------ one trial: logs the harness and the verifier left
function TrialLogs({ id }) {
  const { t } = useApp();
  const { data, error, loading } = useLoad(() => api.call("agentbench_trial", { trial: id, limit: 12000 }), [id]);
  const [file, setFile] = useState(null);
  if (error) return <ErrorBox error={error} />;
  if (loading || !data) return <Spinner />;
  const files = Object.keys(data.logs || {});
  const shown = file || files.find((f) => f.startsWith("verifier/test-stdout")) || files[0];
  return (
    <div className="space-y-2">
      {data.error && <div className="banner banner-danger"><b>{data.error_type}</b> {data.error.slice(0, 600)}</div>}
      {!files.length && <div className="help">{t("ab_no_logs")}</div>}
      <div className="flex flex-wrap gap-1">{files.map((f) => <button key={f} type="button" className={`btn btn-sm${f === shown ? " btn-primary" : ""}`} onClick={() => setFile(f)}>{f}</button>)}</div>
      {shown && <pre className="pre" style={{ maxHeight: 420, overflow: "auto" }}>{data.logs[shown]}</pre>}
    </div>
  );
}

function CellDetail({ campaign, bench, harness, onClose }) {
  const { t, lang, version } = useApp();
  const { data, error, loading } = useLoad(() => api.call("agentbench_cell", { campaign, benchmark: bench, harness }), [campaign, bench, harness, version]);
  const [open, setOpen] = useState(null);
  const c = data?.cell;
  return (
    <div className="panel space-y-3" aria-live="polite">
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="mr-auto">{bench} · {harness}</h2>
        <button type="button" className="btn btn-sm" onClick={onClose}><Icon d={ICONS.close} size={14} />{t("close")}</button>
      </div>
      <ErrorBox error={error} />
      {loading && !data && <Spinner />}
      {c && (
        <>
          <div className="kpis">
            <div className="kpi"><b>{c.rate == null ? "—" : pct(c.rate, 0, lang)}</b><span>{t("ab_rate")} · n={c.n}</span></div>
            <div className="kpi"><b>{c.ci[0] == null ? "—" : `${pct(c.ci[0], 0, lang)}–${pct(c.ci[1], 0, lang)}`}</b><span>{t("ab_ci")}</span></div>
            <div className="kpi"><b>{c.tasks_solved}/{c.tasks}</b><span>{t("ab_tasks_solved")}</span></div>
            <div className="kpi"><b>{c.none_n}</b><span>None</span></div>
            <div className="kpi"><b>{c.tokens_out_mean == null ? "—" : num(c.tokens_out_mean, 0, lang)}</b><span>{t("hb_tokens_out")}</span></div>
            <div className="kpi"><b>{c.agent_s_mean == null ? "—" : `${num(c.agent_s_mean / 60, 1, lang)} min`}</b><span>{t("ab_agent_time")}</span></div>
          </div>
          <div className="panel scroll-x" style={{ padding: 0 }}>
            <table className="tbl">
              <thead><tr><th>{t("case")}</th><th>{t("ab_split")}</th><th>{t("status")}</th><th className="r">{t("hb_tokens_out")}</th><th className="r">{t("ab_agent_time")}</th></tr></thead>
              <tbody>
                {data.trials.map((x) => (
                  <React.Fragment key={x.id}>
                    <tr>
                      <td><button type="button" className="btn-link" onClick={() => setOpen(open === x.id ? null : x.id)} aria-expanded={open === x.id}>{x.task}</button></td>
                      <td className="help">{x.split}</td>
                      <td><Chip className={OUTCOME_CHIP[x.outcome]}>{t(`ab_out_${x.outcome}`)}</Chip>{x.reason && <> <Chip>{x.reason}</Chip></>}</td>
                      <td className="r num">{x.tokens_out == null ? "—" : num(x.tokens_out, 0, lang)}</td>
                      <td className="r num">{x.agent_s == null ? "—" : `${num(x.agent_s / 60, 1, lang)} min`}</td>
                    </tr>
                    {open === x.id && <tr><td colSpan={5}><TrialLogs id={x.id} /></td></tr>}
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ the board
function CampaignDetail({ id }) {
  const { t, lang, version, changed, notify, confirm } = useApp();
  const [split, setSplit] = useState("");
  const { data, error, loading, reload } = useLoad(() => api.call("agentbench_result", { campaign: id, split }), [id, split, version]);
  const [pick, setPick] = useState(null);
  const [busy, act] = useBusy();
  const c = data?.campaign;
  usePoll(reload, 10000, !c || ACTIVE.has(c.state));
  if (!data) return <div className="space-y-3"><a href="#/agentes" className="btn btn-sm"><Icon d={ICONS.back} size={14} />{t("nav_agents")}</a><ErrorBox error={error} />{loading && !error && <Spinner />}</div>;
  const b = data.board;
  const harnesses = b.harnesses;
  const cancel = () => act("cancel", async () => {
    if (!(await confirm({ title: t("ab_cancel"), message: t("ab_cancel_msg"), confirmLabel: t("ab_cancel") }))) return;
    await api.call("agentbench_cancel", { campaign: id });
    notify(t("cancelled"));
    changed();
    reload();
  });
  const sync = () => act("sync", async () => { await api.call("agentbench_sync", { campaign: id }); reload(); });
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <a href="#/agentes" className="btn btn-sm"><Icon d={ICONS.back} size={14} />{t("nav_agents")}</a>
        <h1 className="mr-auto" style={{ overflowWrap: "anywhere" }}>{c.label || c.name}</h1>
        <StateChip state={c.state} />
        <Busy className="btn" busy={busy.sync} onClick={sync}><Icon d={ICONS.refresh} size={14} />{t("ab_sync")}</Busy>
        {ACTIVE.has(c.state) && <Busy className="btn btn-danger" busy={busy.cancel} onClick={cancel}><Icon d={ICONS.stop} size={14} />{t("ab_cancel")}</Busy>}
      </div>
      {c.error && <Failure m={c.error} />}
      <div className="panel space-y-2">
        <div className="flex flex-wrap items-center gap-3 help num">
          <span>{c.model}</span><span>k={c.k}</span>
          <span>{t("ab_progress", { verified: c.trials.verified, expected: c.trials.expected, passed: c.trials.passed, none: c.trials.none })}</span>
        </div>
        <Progress done={c.trials.verified} total={c.trials.expected} tone={c.state === "failed" ? "bad" : c.state === "done" ? "ok" : undefined} />
        <div className="flex flex-wrap gap-2" role="group" aria-label={t("ab_split")}>
          {["", "dev", "test"].map((s) => <button key={s || "all"} type="button" className={`btn btn-sm${split === s ? " btn-primary" : ""}`} onClick={() => setSplit(s)}>{t(`ab_split_${s || "all"}`)}</button>)}
        </div>
      </div>
      {Object.keys(b.verdict || {}).length > 0 && (
        <div className="panel space-y-2" aria-live="polite">
          <h2>{t("ab_verdict")}</h2>
          <div className="flex flex-wrap gap-2">
            {Object.entries(b.verdict).map(([bench, v]) => (
              <Chip key={bench} className={VERDICT_CHIP[v.state]}>{bench}: {t(`ab_v_${v.state}`, { rank: v.rank ?? "—", lead: v.lead == null ? "—" : `${v.lead > 0 ? "+" : ""}${num(v.lead * 100, 1, lang)}` })}{v.weak_evidence ? ` · ${t("ab_weak")}` : ""}</Chip>
            ))}
          </div>
        </div>
      )}
      <Section title={t("ab_board")}>
        {!b.trials ? <Empty>{ACTIVE.has(c.state) ? t("waiting_results") : t("no_results")}</Empty> : (
          <div className="panel scroll-x" style={{ padding: 0 }}>
            <table className="tbl hb-table">
              <thead><tr><th>{t("ab_benchmark")}</th>{harnesses.map((h) => <th key={h} scope="col">{h}</th>)}</tr></thead>
              <tbody>
                {b.benchmarks.map((bench) => (
                  <tr key={bench}>
                    <th scope="row" style={{ color: "var(--ink)" }}>{bench}</th>
                    {harnesses.map((h) => {
                      const cell = b.cells[`${bench}|${h}`];
                      if (!cell) return <td key={h} className="hb-cell hb-empty">—</td>;
                      const rank = (b.ranking[bench] || []).find((r) => r.harness === h)?.rank;
                      const selected = pick?.bench === bench && pick?.harness === h;
                      return (
                        <td key={h} className="hb-cell" style={{ background: cell.rate == null ? undefined : heat(cell.rate) }}>
                          <button type="button" className="hb-pick" aria-pressed={selected} onClick={() => setPick(selected ? null : { bench, harness: h })}>
                            <span className="hb-main num">{cell.rate == null ? "—" : pct(cell.rate, 0, lang)}{rank === 1 ? " ★" : ""}</span>
                            <span className="hb-sub num">n={cell.n}{cell.none_n ? ` · None ${cell.none_n}` : ""}</span>
                          </button>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <p className="help">{t("ab_legend")}</p>
      </Section>
      {Object.keys(b.duels || {}).length > 0 && (
        <Section title={t("ab_duels")}>
          <div className="panel scroll-x" style={{ padding: 0 }}>
            <table className="tbl">
              <thead><tr><th>{t("ab_benchmark")}</th><th>{t("ab_vs")}</th><th className="r">{t("ab_wins")}</th><th className="r">{t("ab_losses")}</th><th className="r">{t("ab_shared")}</th><th className="r">p</th></tr></thead>
              <tbody>
                {Object.entries(b.duels).flatMap(([bench, rows]) => rows.map((d) => (
                  <tr key={`${bench}/${d.vs}`}><td>{bench}</td><td>{d.vs}</td><td className="r num">{d.wins}</td><td className="r num">{d.losses}</td><td className="r num">{d.shared}</td><td className="r num">{num(d.p, 3, lang)}</td></tr>
                )))}
              </tbody>
            </table>
          </div>
        </Section>
      )}
      {pick && <CellDetail key={`${pick.bench}/${pick.harness}`} campaign={id} bench={pick.bench} harness={pick.harness} onClose={() => setPick(null)} />}
    </div>
  );
}

export default function Agentes({ param }) {
  const { t } = useApp();
  if (param) return <CampaignDetail key={param} id={param} />;
  return (
    <div className="space-y-5">
      <h1>{t("nav_agents")}</h1>
      <p className="help">{t("ab_help")}</p>
      <NewCampaign />
      <History />
    </div>
  );
}
