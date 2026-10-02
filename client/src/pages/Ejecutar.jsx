import React, { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api.js";
import { useApp } from "../context.js";
import { Busy, Chip, Empty, ErrorBox, Failure, Field, Icon, ICONS, Modal, Problem, Rel, Section, Spinner, Switch, useBusy, useLoad } from "../components/ui.jsx";
import { CpuBadge, Progress, StateChip } from "../components/parts.jsx";
import { usePoll } from "../components/hooks.js";
import { ACTIVE_STATES, CATEGORIES, EFFORTS, RESUMABLE_STATES as RESUMABLE } from "../meta.js";
import { duration, elapsed, gb, num, shortClock } from "../format.js";

// ------------------------------------------------------------------ the form
function NewRun({ query }) {
  const { t, lang, version, changed, notify } = useApp();
  const suitesLoad = useLoad(() => api.call("suites_list", {}), [version]);
  const modelsLoad = useLoad(() => api.call("models_list", { enabled: true, limit: 200 }), [version]);
  const [suites, setSuites] = useState(() => new Set(query.get("suite") ? [query.get("suite")] : []));
  const [models, setModels] = useState(() => new Set(query.get("model") ? [query.get("model")] : []));
  const [settings, setSettings] = useState({ repeats: 1, temperature: 0, effort: "", device: "auto", context: "", max_tokens: "", timeout_s: "" });
  const [plan, setPlan] = useState(null);
  const [planError, setPlanError] = useState(null);
  const [busy, run] = useBusy();
  const allSuites = suitesLoad.data?.suites || [];
  const allModels = modelsLoad.data?.models || [];

  // The query may carry a suite slug (rapida) instead of its id: resolve once the list is there.
  useEffect(() => {
    const wanted = query.get("suite");
    if (wanted && allSuites.length) {
      const found = allSuites.find((s) => s.id === wanted || s.id === `s_${wanted}` || s.name === wanted);
      if (found) setSuites(new Set([found.id]));
    }
  }, [allSuites.length]); // eslint-disable-line react-hooks/exhaustive-deps

  // A server entry whose address now serves another model cannot be measured: it stays in the list with the reason and cannot be ticked.
  const blocked = useMemo(() => new Set(allModels.filter((m) => m.not_served).map((m) => m.id)), [allModels]);
  const args = useMemo(() => {
    const s = {};
    if (settings.repeats && Number(settings.repeats) !== 1) s.repeats = Number(settings.repeats);
    if (settings.temperature !== "" && Number(settings.temperature) !== 0) s.temperature = Number(settings.temperature);
    if (settings.effort) s.effort = settings.effort;
    if (settings.device && settings.device !== "auto") s.device = settings.device;
    for (const k of ["context", "max_tokens", "timeout_s"]) if (settings[k] !== "") s[k] = Number(settings[k]);
    return { suites: [...suites], contestants: [...models].filter((id) => !blocked.has(id)), settings: s };
  }, [suites, models, settings, blocked]);

  // Estimate (memory, where each model would run, how many cases) whenever the selection changes.
  useEffect(() => {
    if (!args.suites.length || !args.contestants.length) { setPlan(null); setPlanError(null); return undefined; }
    let stale = false;
    const timer = setTimeout(() => {
      api.call("run_plan", args).then((p) => { if (!stale) { setPlan(p); setPlanError(null); } }).catch((e) => { if (!stale) { setPlan(null); setPlanError(e); } });
    }, 250);
    return () => { stale = true; clearTimeout(timer); };
  }, [args]);

  const toggle = (set, setter, id) => setter((prev) => { const next = new Set(prev); next.has(id) ? next.delete(id) : next.add(id); return next; });
  const planFor = (id) => plan?.contestants?.find((c) => c.id === id);
  const start = () => run("start", async () => {
    const r = await api.call("run_start", { ...args, wait_s: 0 });
    notify(t("run_started"));
    changed();
    window.location.hash = `#/ejecutar/${r.run.id}`;
  });
  const set = (k) => (e) => setSettings((s) => ({ ...s, [k]: e.target.value }));
  const totalCases = plan?.total_cases;

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Section title={t("suites")} count={suites.size || undefined}>
        <div className="panel space-y-1" style={{ maxHeight: 360, overflowY: "auto" }}>
          {suitesLoad.loading && !suitesLoad.data && <Spinner />}
          {allSuites.map((s) => (
            <label key={s.id} className="flex items-start gap-2" style={{ cursor: "pointer" }}>
              <input type="checkbox" checked={suites.has(s.id)} onChange={() => toggle(suites, setSuites, s.id)} style={{ marginTop: 4 }} />
              <span className="min-w-0 flex-1"><b>{s.name}</b> <span className="help">{t(`cat_${s.category}`)} · {t("n_cases", { n: s.cases })}</span></span>
            </label>
          ))}
        </div>
      </Section>
      <Section title={t("models")} count={models.size || undefined}>
        <div className="panel space-y-1" style={{ maxHeight: 360, overflowY: "auto" }}>
          {modelsLoad.loading && !modelsLoad.data && <Spinner />}
          {!allModels.length && modelsLoad.data && <div className="help">{t("models_empty")}</div>}
          {allModels.map((m) => {
            const p = planFor(m.id);
            return (
              <label key={m.id} className="flex items-start gap-2" style={{ cursor: m.not_served ? "not-allowed" : "pointer", opacity: m.not_served ? 0.7 : 1 }}>
                <input type="checkbox" checked={models.has(m.id) && !m.not_served} disabled={m.not_served} onChange={() => toggle(models, setModels, m.id)} style={{ marginTop: 4 }} />
                <span className="min-w-0 flex-1">
                  <b style={{ overflowWrap: "anywhere" }}>{m.name}</b>{" "}
                  <span className="help">{m.kind === "gguf" ? "GGUF" : m.provider}{m.memory_gb != null ? ` · ~${num(m.memory_gb, 1, lang)} GB` : ""}</span>
                  {m.not_served && <Chip className="chip-danger">{t("not_served")}</Chip>}
                  {m.not_served && <span className="help block" style={{ color: "var(--warn)" }}>{t("not_served_excluded", { reason: t.msg(m.not_served_reason) })}</span>}
                  {p && (
                    <span className="help block">
                      {t.msg(p.where) || p.runs_on}{p.vram_mb ? ` · ${gb(p.vram_mb, lang)}` : p.ram_mb ? ` · ${gb(p.ram_mb, lang)} RAM` : ""} · {t("n_cases", { n: p.cases })}
                      {p.warnings?.map((w, i) => <span key={i} className="block" style={{ color: "var(--warn)" }}>{t.msg(w)}</span>)}
                      {p.notes?.map((w, i) => <span key={i} className="block">{t.msg(w)}</span>)}
                    </span>
                  )}
                </span>
              </label>
            );
          })}
        </div>
      </Section>
      <div className="lg:col-span-2 space-y-3">
        <Section title={t("run_settings")}>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4 lg:grid-cols-7">
            <Field label={t("repeats")}><input className="field" type="number" min="1" max="20" value={settings.repeats} onChange={set("repeats")} /></Field>
            <Field label={t("temperature")}><input className="field" type="number" min="0" max="2" step="0.1" value={settings.temperature} onChange={set("temperature")} /></Field>
            <Field label={t("effort")}><select className="field" value={settings.effort} onChange={set("effort")}>{EFFORTS.map((e) => <option key={e} value={e}>{e ? t(`effort_${e}`) : t("effort_default")}</option>)}</select></Field>
            <Field label={t("device")}><select className="field" value={settings.device} onChange={set("device")}>{["auto", "gpu", "cpu"].map((d) => <option key={d} value={d}>{t(`device_${d}`)}</option>)}</select></Field>
            <Field label={t("context_gguf")}><input className="field" type="number" min="512" placeholder="8192" value={settings.context} onChange={set("context")} /></Field>
            <Field label={t("max_tokens")}><input className="field" type="number" min="16" value={settings.max_tokens} onChange={set("max_tokens")} placeholder={t("per_suite")} /></Field>
            <Field label={t("timeout_s")}><input className="field" type="number" min="5" value={settings.timeout_s} onChange={set("timeout_s")} placeholder="120" /></Field>
          </div>
        </Section>
        <ErrorBox error={planError} />
        {plan?.problems?.map((p, i) => <Problem key={i} p={p} />)}
        <div className="flex flex-wrap items-center gap-3">
          <Busy className="btn btn-primary" busy={busy.start} disabled={!suites.size || !args.contestants.length || !!planError} onClick={start}><Icon d={ICONS.run} size={14} />{t("start_run")}</Busy>
          {totalCases != null && <span className="help num">{t("total_cases", { n: totalCases })}</span>}
        </div>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ history
function History() {
  const { t, version } = useApp();
  const { data, error, loading, reload } = useLoad(() => api.call("runs_list", { limit: 30 }), [version]);
  const active = (data?.runs || []).some((r) => ACTIVE_STATES.has(r.state));
  usePoll(reload, 3000, active);
  const runs = data?.runs || [];
  return (
    <Section title={t("history")} count={runs.length || undefined}>
      <ErrorBox error={error} />
      {loading && !data && <Spinner />}
      {data && !runs.length && <Empty>{t("no_runs")}</Empty>}
      {runs.length > 0 && (
        <div className="panel scroll-x" style={{ padding: 0 }}>
          <table className="tbl">
            <thead><tr><th>{t("run")}</th><th>{t("status")}</th><th>{t("progress")}</th><th>{t("models")}</th><th>{t("created")}</th></tr></thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.id} style={r.discarded ? { opacity: 0.65 } : undefined}>
                  <td style={{ overflowWrap: "anywhere" }}><a href={`#/ejecutar/${r.id}`}>{t.msg(r.label)}</a></td>
                  <td><StateChip state={r.state} />{r.discarded && <Chip className="chip-danger" title={r.discard_reason}>{t("discarded_run")}</Chip>}{r.pending_judge > 0 && <Chip className="chip-amber">{t("judge_pending_n", { n: r.pending_judge })}</Chip>}</td>
                  <td style={{ minWidth: 120 }}><div className="help num">{r.progress.done}/{r.progress.total}</div><Progress done={r.progress.done} total={r.progress.total} /></td>
                  <td className="help">{r.contestants.map((c) => c.name).join(", ")}</td>
                  <td className="help"><Rel ts={r.created_ts} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Section>
  );
}

// ------------------------------------------------------------------ one run
function ResultRow({ r, runId }) {
  const { t, lang } = useApp();
  const [open, setOpen] = useState(false);
  const [full, setFull] = useState(null);
  const [error, setError] = useState(null);
  const expand = async () => {
    const next = !open;
    setOpen(next);
    if (next && !full) {
      try {
        const res = await api.call("run_results", { run: runId, model: r.contestant, case: r.case, include_output: true, limit: 50 });
        setFull(res.results.find((x) => x.id === r.id) || res.results[0] || null);
      } catch (e) {
        setError(e);
      }
    }
  };
  const verdict = r.skipped ? "skip" : r.unavailable ? "unavailable" : r.judge_pending ? "judge" : r.error ? "error" : r.truncated ? "truncated" : r.passed ? "pass" : "fail";
  const chip = { pass: "chip-ok", fail: "chip-danger", error: "chip-danger", skip: "", unavailable: "chip-amber", judge: "chip-amber", truncated: "chip-amber" }[verdict];
  return (
    <>
      <tr>
        <td style={{ overflowWrap: "anywhere" }}><button type="button" className="btn-link" style={{ textAlign: "left" }} onClick={expand} aria-expanded={open}>{r.title}</button></td>
        <td className="help">{r.contestant_name}</td>
        <td><Chip className={chip} title={r.truncated ? t("truncated_hint") : undefined}>{t(`res_${verdict}`)}</Chip>{r.self_judged && <Chip className="chip-amber">{t("self_judged")}</Chip>}</td>
        <td className="r num">{r.skipped ? "—" : num(r.score, 2, lang)}</td>
        <td className="r num">{duration(r.latency_ms)}</td>
        <td className="r num">{r.decode_tps ? num(r.decode_tps, 1, lang) : "—"}{r.cpu && r.decode_tps ? <> <CpuBadge /></> : null}</td>
        <td className="r num">{r.completion_tokens ?? "—"}</td>
      </tr>
      {open && (
        <tr>
          <td colSpan={7}>
            <div className="space-y-2">
              <ErrorBox error={error} />
              {r.error && <Failure m={r.error} className="banner banner-danger" />}
              {!full && !error && <Spinner />}
              {full && (
                <>
                  <div className="pre">{full.output || t("empty_output")}{full.output_truncated ? "\n…" : ""}</div>
                  {full.reasoning && <details><summary className="help">{t("reasoning")}</summary><div className="pre mt-1">{full.reasoning}</div></details>}
                  {full.tool_calls?.length > 0 && <details open><summary className="help">tool_calls</summary><pre className="pre mt-1">{JSON.stringify(full.tool_calls, null, 2)}</pre></details>}
                  <details><summary className="help">{t("checker_detail")}</summary><pre className="pre mt-1">{JSON.stringify(full.detail, null, 2)}</pre></details>
                  <div className="help num">{t("tokens_in_out", { a: full.prompt_tokens ?? "—", b: full.completion_tokens ?? "—" })} · TTFT {duration(full.ttft_ms)}</div>
                </>
              )}
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

// What cancelling does depends on the contestants the run has still to go through: only a llama-server that Galton started is stopped.
function cancelMessage(run, t) {
  const open = run.contestants.filter((c) => !["done", "failed", "cancelled"].includes(c.state));
  const own = open.some((c) => c.kind === "gguf");
  const shared = open.filter((c) => c.kind !== "gguf");
  const urls = [...new Set(shared.map((c) => (c.runs_on ? t.msg(c.runs_on).replace(/^\S+\s+/, "") : c.name)))].join(", ");
  return t(own && shared.length ? "cancel_run_msg_both" : shared.length ? "cancel_run_msg_shared" : "cancel_run_msg_own", { urls });
}

function DiscardModal({ run, onClose, onDone }) {
  const { t } = useApp();
  const [reason, setReason] = useState("");
  const [busy, act] = useBusy();
  const [error, setError] = useState(null);
  const submit = (e) => {
    e.preventDefault();
    act("discard", async () => {
      try {
        await api.call("run_discard", { run: run.id, reason: reason.trim(), confirm: true });
        onDone();
        onClose();
      } catch (err) {
        setError(err);
      }
    });
  };
  return (
    <Modal title={t("discard_run")} onClose={onClose} wide>
      <form className="space-y-3" onSubmit={submit}>
        <p>{t("discard_run_msg")}</p>
        <Field label={t("discard_reason")} hint={t("discard_reason_hint")}><textarea className="field" rows={3} value={reason} onChange={(e) => setReason(e.target.value)} required minLength={3} maxLength={500} /></Field>
        <ErrorBox error={error} />
        <div className="flex justify-end gap-2">
          <button type="button" className="btn" onClick={onClose}>{t("cancel")}</button>
          <Busy className="btn btn-danger" type="submit" busy={busy.discard} disabled={reason.trim().length < 3}>{t("discard_confirm")}</Busy>
        </div>
      </form>
    </Modal>
  );
}

function RunDetail({ id }) {
  const { t, lang, changed, notify, confirm } = useApp();
  const [discarding, setDiscarding] = useState(false);
  const [run, setRun] = useState(null);
  const [error, setError] = useState(null);
  const [results, setResults] = useState([]);
  const [filter, setFilter] = useState({ contestant: "", failed: false });
  const [busy, act] = useBusy();
  const last = useRef(0);
  const seen = useRef(new Set());
  const finished = useRef(false);

  const pull = async () => {
    try {
      for (let guard = 0; guard < 20; guard += 1) {
        const ev = await api.events(id, last.current);
        setRun(ev.run);
        const fresh = ev.results.filter((r) => !seen.current.has(r.id));
        fresh.forEach((r) => seen.current.add(r.id));
        if (fresh.length) setResults((prev) => [...prev, ...fresh]);
        last.current = ev.last_id;
        finished.current = ev.finished;
        if (ev.results.length < 300) break;
      }
      setError(null);
    } catch (e) {
      setError(e);
    }
  };
  // Poll while the run is live; one extra pull after it finishes picks up the last results and the judge's updates.
  const live = !run || ACTIVE_STATES.has(run.state) || run.pending_judge > 0;
  usePoll(pull, 1500, live);
  useEffect(() => { if (run && !live) pull(); }, [run?.state]); // eslint-disable-line react-hooks/exhaustive-deps

  const cancel = () => act("cancel", async () => {
    if (!(await confirm({ title: t("cancel_run"), message: cancelMessage(run, t), confirmLabel: t("cancel_run") }))) return;
    await api.call("run_cancel", { run: id });
    notify(t("cancelled"));
    changed();
    pull();
  });
  const judge = () => act("judge", async () => { await api.call("judge_run", {}); notify(t("judge_queued")); changed(); });
  const resume = () => act("resume", async () => {
    const r = await api.call("run_resume", { run: id });
    notify(t("run_resumed"));
    changed();
    window.location.hash = `#/ejecutar/${r.run.id}`;
  });
  const restore = () => act("restore", async () => { await api.call("run_restore", { run: id }); notify(t("run_restored")); changed(); pull(); });

  const shown = useMemo(() => results.filter((r) => (!filter.contestant || r.contestant === filter.contestant) && (!filter.failed || (!r.passed && !r.skipped))), [results, filter]);

  if (!run) return <div className="space-y-3"><a href="#/ejecutar" className="btn btn-sm"><Icon d={ICONS.back} size={14} />{t("nav_run")}</a><ErrorBox error={error} />{!error && <Spinner />}</div>;
  const active = ACTIVE_STATES.has(run.state);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <a href="#/ejecutar" className="btn btn-sm"><Icon d={ICONS.back} size={14} />{t("nav_run")}</a>
        <h1 className="mr-auto" style={{ overflowWrap: "anywhere" }}>{t.msg(run.label)}</h1>
        <StateChip state={run.state} />
        {run.discarded && <Chip className="chip-danger">{t("discarded_run")}</Chip>}
        {run.pending_judge > 0 && <Busy className="btn" busy={busy.judge} onClick={judge}>{t("judge_now", { n: run.pending_judge })}</Busy>}
        {active && <Busy className="btn btn-danger" busy={busy.cancel} onClick={cancel}><Icon d={ICONS.stop} size={14} />{t("cancel_run")}</Busy>}
        {RESUMABLE.has(run.state) && !run.discarded && <Busy className="btn btn-primary" busy={busy.resume} onClick={resume}><Icon d={ICONS.run} size={14} />{t("resume_run")}</Busy>}
        {!active && !run.discarded && <button type="button" className="btn" onClick={() => setDiscarding(true)}>{t("discard_run")}</button>}
        {run.discarded && <Busy className="btn" busy={busy.restore} onClick={restore}>{t("restore_run")}</Busy>}
      </div>
      {discarding && <DiscardModal run={run} onClose={() => setDiscarding(false)} onDone={() => { notify(t("run_discarded")); changed(); pull(); }} />}
      {run.continues && <div className="help">{t("continues_run")} <a href={`#/ejecutar/${run.continues}`}>{run.continues_label ? t.msg(run.continues_label) : run.continues}</a></div>}
      {run.discarded && <div className="banner banner-warn">{t("discarded_banner", { reason: run.discard_reason })}</div>}
      <ErrorBox error={error} />
      {run.error && <Failure m={run.error} className="banner banner-danger" />}
      <div className="panel space-y-2">
        <div className="flex flex-wrap items-center gap-3 help num">
          <span>{run.progress.done}/{run.progress.total} · {num(run.progress.pct, 0, lang)} %</span>
          <span>{t("created")} {shortClock(run.created_ts, lang)}</span>
          {run.finished_ts && <span>{t("finished")} {shortClock(run.finished_ts, lang)}</span>}
          <span>{run.suites.map((s) => s.name).join(", ")}</span>
        </div>
        <Progress done={run.progress.done} total={run.progress.total} tone={run.state === "failed" ? "bad" : run.state === "done" ? "ok" : undefined} />
      </div>
      <div className="card-grid">
        {run.contestants.map((c) => (
          <div key={c.id} className="panel space-y-1.5">
            <div className="flex flex-wrap items-center gap-2"><b className="min-w-0 flex-1" style={{ overflowWrap: "anywhere" }}>{c.name}</b>{c.device === "cpu" && <CpuBadge />}<StateChip state={c.state} /></div>
            <Progress done={c.done} total={c.total} />
            <div className="help num">{c.done}/{c.total} · {t("passed")} {c.passed} · {t("res_error")} {c.errors} · {t("res_skip")} {c.skipped}{c.truncated > 0 ? ` · ${t("truncated_n", { n: c.truncated })}` : ""}</div>
            {c.runs_on && <div className="help">{t.msg(c.runs_on)}{c.vram_mb ? ` · ${gb(c.vram_mb, lang)}${c.vram_method ? ` (${c.vram_method})` : ""}` : ""}{c.load_ms != null ? ` · ${t("load")} ${num(c.load_ms / 1000, 1, lang)} s` : ""}</div>}
            {c.waiting_s != null && <div className="banner banner-warn">{t("waiting_server_now", { time: elapsed(c.waiting_s) })}</div>}
            {c.waiting_s == null && c.waited_s > 0 && <div className="help">{t("waited_server", { time: elapsed(c.waited_s) })}</div>}
            {c.spill && <div className="banner banner-warn">{t.msg(c.spill)}</div>}
            {c.warnings?.map((w, i) => <div key={i} className="help" style={{ color: "var(--warn)" }}>{t.msg(w)}</div>)}
            {c.error && <Failure m={c.error} className="banner banner-danger" />}
          </div>
        ))}
      </div>
      <Section title={t("results")} count={results.length}
        actions={(
          <>
            <select className="field" style={{ width: 200 }} aria-label={t("model")} value={filter.contestant} onChange={(e) => setFilter((f) => ({ ...f, contestant: e.target.value }))}>
              <option value="">{t("all_models")}</option>
              {run.contestants.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
            </select>
            <label className="flex items-center gap-2"><Switch checked={filter.failed} onChange={(v) => setFilter((f) => ({ ...f, failed: v }))} label={t("only_failed")} /><span>{t("only_failed")}</span></label>
          </>
        )}>
        {!shown.length ? <Empty>{active ? t("waiting_results") : t("no_results")}</Empty> : (
          <div className="panel scroll-x" style={{ padding: 0 }}>
            <table className="tbl">
              <thead><tr><th>{t("case")}</th><th>{t("model")}</th><th>{t("status")}</th><th className="r">{t("score")}</th><th className="r">{t("latency")}</th><th className="r">tok/s</th><th className="r">tokens</th></tr></thead>
              <tbody>{shown.slice(-600).map((r) => <ResultRow key={r.id} r={r} runId={id} />)}</tbody>
            </table>
          </div>
        )}
        {shown.length > 600 && <div className="help">{t("showing_last", { n: 600, total: shown.length })}</div>}
      </Section>
    </div>
  );
}

export default function Ejecutar({ param, query }) {
  const { t } = useApp();
  if (param) return <RunDetail key={param} id={param} />;
  return (
    <div className="space-y-5">
      <h1>{t("nav_run")}</h1>
      <NewRun query={query} />
      <History />
    </div>
  );
}
