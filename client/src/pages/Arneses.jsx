import React, { useEffect, useMemo, useState } from "react";
import { api } from "../api.js";
import { useApp } from "../context.js";
import { Busy, Chip, Empty, ErrorBox, Failure, Field, Icon, ICONS, Problem, Rel, Section, Spinner, useBusy, useLoad } from "../components/ui.jsx";
import { Progress, StateChip } from "../components/parts.jsx";
import { usePoll } from "../components/hooks.js";
import { ACTIVE_STATES } from "../meta.js";
import { duration, num, pct } from "../format.js";

const BASES = ["native_tools", "json_text", "xml_text", "react_text"];
const OUTCOME_CHIP = { pass: "chip-ok", fail_answer: "chip-danger", fail_format: "chip-amber", none: "" };

// The colour of a cell: from the danger colour at 0 % to the success colour at 100 %, soft enough to read the text over it in any theme.
const heat = (p) => `color-mix(in srgb, color-mix(in srgb, var(--ok) ${Math.round(p * 100)}%, var(--danger)) 34%, transparent)`;

// ------------------------------------------------------------------ the form
function NewMatrix() {
  const { t, version, changed, notify } = useApp();
  const modelsLoad = useLoad(() => api.call("models_list", { enabled: true, limit: 200 }), [version]);
  const catalog = useLoad(() => api.call("harness_catalog", {}), [version]);
  const [models, setModels] = useState(new Set());
  const [bases, setBases] = useState(new Set(BASES));
  const [prompts, setPrompts] = useState(new Set(["short"]));
  const [history, setHistory] = useState(new Set(["none"]));
  const [external, setExternal] = useState(new Set());
  const [kinds, setKinds] = useState(null);                 // null: every kind
  const [form, setForm] = useState({ k: 1, max_turns: "", max_tokens: "" });
  const [plan, setPlan] = useState(null);
  const [planError, setPlanError] = useState(null);
  const [busy, run] = useBusy();

  const allModels = modelsLoad.data?.models || [];
  const blocked = useMemo(() => new Set(allModels.filter((m) => m.not_served).map((m) => m.id)), [allModels]);
  const tasks = catalog.data?.tasks || [];
  const allKinds = catalog.data?.kinds || [];
  const kindCount = (kind) => tasks.filter((x) => x.kind === kind).length;
  const chosenKinds = kinds || new Set(allKinds);
  const configured = (catalog.data?.external?.configured || []).filter((s) => s.enabled);
  const externalOn = !!catalog.data?.external?.allowed;

  const harnesses = useMemo(() => {
    const out = [];
    for (const base of BASES) {
      if (!bases.has(base)) continue;
      for (const p of ["short", "long"]) for (const h of ["none", "trim"]) {
        if (prompts.has(p) && history.has(h)) out.push(base + (p === "long" ? ".long" : "") + (h === "trim" ? ".trim" : ""));
      }
    }
    return [...out, ...[...external]];
  }, [bases, prompts, history, external]);

  const args = useMemo(() => {
    const a = { models: [...models].filter((id) => !blocked.has(id)), harnesses, k: Math.max(1, Number(form.k) || 1) };
    if (kinds && kinds.size !== allKinds.length) a.kinds = [...kinds];
    const settings = {};
    if (form.max_turns !== "") settings.max_turns = Number(form.max_turns);
    if (form.max_tokens !== "") settings.max_tokens = Number(form.max_tokens);
    if (Object.keys(settings).length) a.settings = settings;
    return a;
  }, [models, blocked, harnesses, kinds, allKinds.length, form]);

  const ready = args.models.length > 0 && args.harnesses.length > 0 && (!kinds || kinds.size > 0);
  useEffect(() => {
    if (!ready) { setPlan(null); setPlanError(null); return undefined; }
    let stale = false;
    const timer = setTimeout(() => {
      api.call("harness_matrix_start", { ...args, dry_run: true }).then((p) => { if (!stale) { setPlan(p); setPlanError(null); } }).catch((e) => { if (!stale) { setPlan(null); setPlanError(e); } });
    }, 250);
    return () => { stale = true; clearTimeout(timer); };
  }, [args, ready]);

  const toggle = (set, setter, id) => setter(() => { const next = new Set(set); next.has(id) ? next.delete(id) : next.add(id); return next; });
  const toggleKind = (kind) => setKinds(() => { const next = new Set(chosenKinds); next.has(kind) ? next.delete(kind) : next.add(kind); return next; });
  const start = () => run("start", async () => {
    const r = await api.call("harness_matrix_start", { ...args, wait_s: 0 });
    notify(t("hb_started"));
    changed();
    window.location.hash = `#/arneses/${r.matrix.id}`;
  });
  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }));
  const checkbox = (checked, onChange, label, disabled) => (
    <label className="flex items-center gap-2" style={{ cursor: disabled ? "not-allowed" : "pointer" }}>
      <input type="checkbox" checked={checked} disabled={disabled} onChange={onChange} />{label}
    </label>
  );

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Section title={t("models")} count={models.size || undefined}>
        <div className="panel space-y-1" style={{ maxHeight: 300, overflowY: "auto" }}>
          {modelsLoad.loading && !modelsLoad.data && <Spinner />}
          {!allModels.length && modelsLoad.data && <div className="help">{t("models_empty")}</div>}
          {allModels.map((m) => (
            <label key={m.id} className="flex items-start gap-2" style={{ cursor: m.not_served ? "not-allowed" : "pointer", opacity: m.not_served ? 0.7 : 1 }}>
              <input type="checkbox" checked={models.has(m.id) && !m.not_served} disabled={m.not_served} onChange={() => toggle(models, setModels, m.id)} style={{ marginTop: 4 }} />
              <span className="min-w-0 flex-1"><b style={{ overflowWrap: "anywhere" }}>{m.name}</b> <span className="help">{m.kind === "gguf" ? "GGUF" : m.provider}</span>
                {m.not_served && <Chip className="chip-danger">{t("not_served")}</Chip>}</span>
            </label>
          ))}
        </div>
      </Section>
      <Section title={t("hb_harnesses")} count={harnesses.length || undefined}>
        <div className="panel space-y-3">
          <div role="group" aria-label={t("hb_harnesses")} className="grid gap-1 sm:grid-cols-2">
            {BASES.map((b) => <React.Fragment key={b}>{checkbox(bases.has(b), () => toggle(bases, setBases, b), <span><b>{b}</b> <span className="help">{t(`hb_base_${b}`)}</span></span>)}</React.Fragment>)}
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <div role="group" aria-label={t("hb_prompt")}>
              <span className="label">{t("hb_prompt")}</span>
              {checkbox(prompts.has("short"), () => toggle(prompts, setPrompts, "short"), t("hb_prompt_short"))}
              {checkbox(prompts.has("long"), () => toggle(prompts, setPrompts, "long"), t("hb_prompt_long"))}
            </div>
            <div role="group" aria-label={t("hb_history")}>
              <span className="label">{t("hb_history")}</span>
              {checkbox(history.has("none"), () => toggle(history, setHistory, "none"), t("hb_history_none"))}
              {checkbox(history.has("trim"), () => toggle(history, setHistory, "trim"), t("hb_history_trim"))}
            </div>
          </div>
          {configured.length > 0 && (
            <div role="group" aria-label={t("hb_external")}>
              <span className="label">{t("hb_external")}</span>
              {configured.map((s) => <React.Fragment key={s.id}>{checkbox(external.has(s.id), () => toggle(external, setExternal, s.id), <span><b>{s.id}</b></span>, !externalOn)}</React.Fragment>)}
              {!externalOn && <span className="help block">{t("hb_external_off_hint")}</span>}
            </div>
          )}
          {!configured.length && catalog.data && <div className="help">{t("hb_external_none")}</div>}
        </div>
      </Section>
      <div className="lg:col-span-2 space-y-3">
        <Section title={t("hb_tasks")} count={tasks.length ? tasks.filter((x) => chosenKinds.has(x.kind)).length : undefined}>
          <div className="panel flex flex-wrap gap-x-5 gap-y-1" role="group" aria-label={t("hb_tasks")}>
            {catalog.loading && !catalog.data && <Spinner />}
            {allKinds.map((kind) => <React.Fragment key={kind}>{checkbox(chosenKinds.has(kind), () => toggleKind(kind), <span>{t(`hb_kind_${kind}`)} <span className="help">{kindCount(kind)}</span></span>)}</React.Fragment>)}
          </div>
        </Section>
        <Section title={t("run_settings")}>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Field label={t("hb_k")} hint={t("hb_k_hint")}><input className="field" type="number" min="1" max="20" value={form.k} onChange={set("k")} /></Field>
            <Field label={t("hb_max_turns")} hint={t("hb_same_cap")}><input className="field" type="number" min="1" max="30" placeholder="8" value={form.max_turns} onChange={set("max_turns")} /></Field>
            <Field label={t("max_tokens")} hint={t("hb_same_cap")}><input className="field" type="number" min="16" placeholder="1024" value={form.max_tokens} onChange={set("max_tokens")} /></Field>
          </div>
        </Section>
        <ErrorBox error={planError} />
        {plan && <div className="help num">{t("hb_plan", { trials: plan.trials, models: plan.models.length, harnesses: plan.harnesses.length, tasks: plan.tasks })}</div>}
        {plan?.problems?.map((p, i) => <Problem key={i} p={p} />)}
        <div className="flex flex-wrap items-center gap-3">
          <Busy className="btn btn-primary" busy={busy.start} disabled={!ready || !!planError} onClick={start}><Icon d={ICONS.run} size={14} />{t("hb_start")}</Busy>
        </div>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ history
function History() {
  const { t, lang, version } = useApp();
  const { data, error, loading, reload } = useLoad(() => api.call("harness_matrix_list", { limit: 30 }), [version]);
  const rows = data?.matrices || [];
  usePoll(reload, 3000, rows.some((m) => ACTIVE_STATES.has(m.state)));
  return (
    <Section title={t("history")} count={rows.length || undefined}>
      <ErrorBox error={error} />
      {loading && !data && <Spinner />}
      {data && !rows.length && <Empty>{t("hb_none")}</Empty>}
      {rows.length > 0 && (
        <div className="panel scroll-x" style={{ padding: 0 }}>
          <table className="tbl">
            <thead><tr><th>{t("run")}</th><th>{t("status")}</th><th>{t("progress")}</th><th>{t("hb_conclusion")}</th><th>{t("created")}</th></tr></thead>
            <tbody>
              {rows.map((m) => (
                <tr key={m.id}>
                  <td style={{ overflowWrap: "anywhere" }}><a href={`#/arneses/${m.id}`}>{t.msg(m.label)}</a><div className="help">{m.models.map((x) => x.name).join(", ")}</div></td>
                  <td><StateChip state={m.state} /></td>
                  <td style={{ minWidth: 120 }}><div className="help num">{m.done}/{m.total}</div><Progress done={m.done} total={m.total} /></td>
                  <td className="help" style={{ maxWidth: 420 }}>{m.summary?.summary?.[lang] || "—"}</td>
                  <td className="help"><Rel ts={m.created_ts} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Section>
  );
}

// ------------------------------------------------------------------ one trial: what the model said, what the harness understood, what ran
function Trace({ trial }) {
  const { t } = useApp();
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <Chip className={OUTCOME_CHIP[trial.outcome]}>{t(`hb_out_${trial.outcome}`)}</Chip>
        {trial.reason && <Chip>{t(`hb_reason_${trial.reason}`)}</Chip>}
        <span className="help num">{t("hb_trial_numbers", { turns: trial.turns ?? "—", calls: trial.calls ?? "—", format: trial.format_failures ?? "—", extra: trial.extra_calls ?? "—" })}</span>
      </div>
      {trial.detail && <div className="help">{t.msg(trial.detail)}</div>}
      {(trial.trace || []).map((step) => (
        <div key={step.turn} className="panel panel-tight space-y-1">
          <div className="help">{t("hb_turn", { n: step.turn })}{step.finish ? ` · ${step.finish}` : ""}</div>
          {step.text ? <pre className="pre">{step.text}</pre> : <div className="help">{t("hb_empty_reply")}</div>}
          {step.error && <div className="banner banner-danger">{step.error}</div>}
          {step.format_error && <div className="banner banner-warn"><b>{t("hb_format_error")}</b>: {step.format_error}</div>}
          {(step.rejected || []).map((r, i) => <div key={i} className="banner banner-warn"><b>{t("hb_refused")}</b>{r.name ? ` ${r.name}` : ""}: {r.reason}</div>)}
          {(step.calls || []).map((c, i) => (
            <div key={i} className="space-y-0.5">
              <div className="mono">{c.name}({JSON.stringify(c.args)}) <Chip className={c.ok ? "chip-ok" : "chip-danger"}>{c.ok ? t("hb_call_ok") : t("hb_call_error")}</Chip>{c.extra ? <> <Chip className="chip-amber">{t(`hb_extra_${c.extra}`)}</Chip></> : null}</div>
              <pre className="pre">{c.result}</pre>
            </div>
          ))}
        </div>
      ))}
      {trial.final && <div><span className="label">{t("hb_final")}</span><pre className="pre">{trial.final}</pre></div>}
    </div>
  );
}

function TaskRow({ row, matrix, model, harness }) {
  const { t } = useApp();
  const [open, setOpen] = useState(false);
  const [trials, setTrials] = useState(null);
  const [error, setError] = useState(null);
  const expand = async () => {
    const next = !open;
    setOpen(next);
    if (next && !trials) {
      try {
        setTrials((await api.call("harness_cell", { matrix, model, harness, task: row.task })).trials);
      } catch (e) {
        setError(e);
      }
    }
  };
  const verdict = row.outcomes.includes("pass") && row.outcomes.every((o) => o === "pass") ? "pass" : row.outcomes.every((o) => o === "none") ? "none" : row.outcomes.includes("fail_format") ? "fail_format" : row.outcomes.includes("pass") ? "mixed" : "fail_answer";
  return (
    <>
      <tr>
        <td><button type="button" className="btn-link" style={{ textAlign: "left" }} onClick={expand} aria-expanded={open}>{row.task}</button></td>
        <td className="help">{t(`hb_kind_${row.kind}`)}</td>
        <td><Chip className={OUTCOME_CHIP[verdict]}>{verdict === "mixed" ? `${row.passed}/${row.samples}` : t(`hb_out_${verdict}`)}</Chip>{row.reasons.map((r) => <React.Fragment key={r}> <Chip>{t(`hb_reason_${r}`)}</Chip></React.Fragment>)}</td>
        <td className="r num">{row.format_failures}</td>
        <td className="r num">{row.extra_calls}</td>
        <td className="r num">{row.turns.filter((x) => x != null).join(", ") || "—"}</td>
      </tr>
      {open && (
        <tr><td colSpan={6}>
          <ErrorBox error={error} />
          {!trials && !error && <Spinner />}
          {trials?.map((trial, i) => <div key={i} className="mb-3">{trials.length > 1 && <div className="label">{t("hb_sample", { n: (trial.sample ?? i) + 1 })}</div>}<Trace trial={trial} /></div>)}
        </td></tr>
      )}
    </>
  );
}

function CellDetail({ matrix, model, harness, onClose }) {
  const { t, lang, version } = useApp();
  const { data, error, loading } = useLoad(() => api.call("harness_cell", { matrix, model: model.id, harness }), [matrix, model.id, harness, version]);
  const s = data?.stats;
  return (
    <div className="panel space-y-3" aria-live="polite">
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="mr-auto" style={{ overflowWrap: "anywhere" }}>{model.name} · {harness}</h2>
        <button type="button" className="btn btn-sm" onClick={onClose}><Icon d={ICONS.close} size={14} />{t("close")}</button>
      </div>
      <ErrorBox error={error} />
      {loading && !data && <Spinner />}
      {s && (
        <>
          <div className="kpis">
            <div className="kpi"><b>{pct(s.pass_at_1, 0, lang)}</b><span>pass@1 · n={s.valid}</span></div>
            {s.pass_at_k != null && <div className="kpi"><b>{pct(s.pass_at_k, 0, lang)}</b><span>pass@k</span></div>}
            {s.pass_hat_k != null && <div className="kpi"><b>{pct(s.pass_hat_k, 0, lang)}</b><span>pass^k</span></div>}
            <div className="kpi"><b>{pct(s.format_fail_rate, 0, lang)}</b><span>{t("hb_format_failures")}</span></div>
            <div className="kpi"><b>{pct(s.answer_fail_rate, 0, lang)}</b><span>{t("hb_bad_answers")}</span></div>
            <div className="kpi"><b>{s.extra_calls_mean != null ? num(s.extra_calls_mean, 2, lang) : "—"}</b><span>{t("hb_extra_calls")}</span></div>
            <div className="kpi"><b>{s.turns_mean != null ? num(s.turns_mean, 1, lang) : "—"}</b><span>{t("hb_turns")}</span></div>
            <div className="kpi"><b>{s.tokens_out_mean != null ? num(s.tokens_out_mean, 0, lang) : "—"}</b><span>{t("hb_tokens_out")}</span></div>
            <div className="kpi"><b>{s.latency_ms_mean != null ? duration(s.latency_ms_mean) : "—"}</b><span>{t("latency")}</span></div>
            <div className="kpi"><b>{s.none_n}</b><span>{t("hb_none_n")}</span></div>
          </div>
          {s.none_n > 0 && <div className="banner banner-warn">{t("hb_none_explain", { n: s.none_n })}{Object.keys(s.none_reasons || {}).length > 0 && <span className="help block">{Object.entries(s.none_reasons).map(([why, n]) => `${n} × ${why}`).join(" · ")}</span>}</div>}
          <div className="panel scroll-x" style={{ padding: 0 }}>
            <table className="tbl">
              <thead><tr><th>{t("case")}</th><th>{t("kind")}</th><th>{t("status")}</th><th className="r">{t("hb_format_failures")}</th><th className="r">{t("hb_extra_calls")}</th><th className="r">{t("hb_turns")}</th></tr></thead>
              <tbody>{data.tasks.map((row) => <TaskRow key={row.task} row={row} matrix={matrix} model={model.id} harness={harness} />)}</tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ the table
function HeatCell({ c, k, selected, onPick }) {
  const { t, lang } = useApp();
  if (!c || !c.trials) return <td className="hb-cell hb-empty">—</td>;
  const p = c.pass_at_1;
  return (
    <td className="hb-cell" style={{ background: p == null ? undefined : heat(p) }}>
      <button type="button" className="hb-pick" aria-pressed={selected} onClick={onPick}
        title={t("hb_cell_title", { pass: p == null ? "—" : pct(p, 0, lang), n: c.valid, none: c.none_n })}>
        <span className="hb-main num">{p == null ? "—" : pct(p, 0, lang)}</span>
        <span className="hb-sub num">n={c.valid}{k > 1 && c.pass_at_k != null ? ` · pass@${k} ${pct(c.pass_at_k, 0, lang)}` : ""}</span>
        <span className="hb-tags">
          {c.format_fail_rate > 0 && <Chip className="chip-amber" title={t("hb_format_failures")}>{t("hb_fmt_short")} {pct(c.format_fail_rate, 0, lang)}</Chip>}
          {c.extra_calls_mean > 0 && <Chip title={t("hb_extra_calls")}>{t("hb_extra_short")} {num(c.extra_calls_mean, 1, lang)}</Chip>}
          {c.none_n > 0 && <Chip className="chip-info" title={t("hb_none_n")}>None {c.none_n}</Chip>}
        </span>
      </button>
    </td>
  );
}

function MatrixDetail({ id }) {
  const { t, lang, changed, notify, confirm, version } = useApp();
  const { data, error, loading, reload } = useLoad(() => api.call("harness_matrix_result", { matrix: id }), [id, version]);
  const [pick, setPick] = useState(null);
  const [busy, act] = useBusy();
  const m = data?.matrix;
  const active = !m || ACTIVE_STATES.has(m.state);
  usePoll(reload, 3000, active);
  useEffect(() => { if (m && !ACTIVE_STATES.has(m.state)) reload(); }, [m?.state]); // eslint-disable-line react-hooks/exhaustive-deps

  const cancel = () => act("cancel", async () => {
    if (!(await confirm({ title: t("hb_cancel"), message: t("hb_cancel_msg"), confirmLabel: t("hb_cancel") }))) return;
    await api.call("harness_matrix_cancel", { matrix: id });
    notify(t("cancelled"));
    changed();
    reload();
  });
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(data.markdown);
      notify(t("hb_copied"));
    } catch {
      notify(t("hb_copy_failed"), "error");
    }
  };

  if (!data) return <div className="space-y-3"><a href="#/arneses" className="btn btn-sm"><Icon d={ICONS.back} size={14} />{t("nav_harness")}</a><ErrorBox error={error} />{loading && !error && <Spinner />}</div>;
  const conclusion = data.conclusion;
  const models = data.models;
  const harnesses = data.harnesses.map((h) => h.id);
  const stale = models.filter((x) => x.stale || x.removed);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <a href="#/arneses" className="btn btn-sm"><Icon d={ICONS.back} size={14} />{t("nav_harness")}</a>
        <h1 className="mr-auto" style={{ overflowWrap: "anywhere" }}>{t.msg(m.label)}</h1>
        <StateChip state={m.state} />
        <button type="button" className="btn" onClick={copy}><Icon d={ICONS.copy} size={14} />{t("hb_copy")}</button>
        {ACTIVE_STATES.has(m.state) && <Busy className="btn btn-danger" busy={busy.cancel} onClick={cancel}><Icon d={ICONS.stop} size={14} />{t("hb_cancel")}</Busy>}
      </div>
      <ErrorBox error={error} />
      {m.error && <Failure m={m.error} />}
      <div className="panel space-y-2">
        <div className="flex flex-wrap items-center gap-3 help num"><span>{m.done}/{m.total} · {num(m.total ? (100 * m.done) / m.total : 0, 0, lang)} %</span><span>k={m.k}</span><span>{t("hb_caps", { turns: m.settings.max_turns, tokens: m.settings.max_tokens, temp: m.settings.temperature })}</span></div>
        <Progress done={m.done} total={m.total} tone={m.state === "failed" ? "bad" : m.state === "done" ? "ok" : undefined} />
      </div>
      {m.models.some((x) => x.error || x.state === "running" || (x.warnings || []).length) && (
        <div className="card-grid">
          {m.models.map((x) => (
            <div key={x.id} className="panel space-y-1.5">
              <div className="flex flex-wrap items-center gap-2"><b className="min-w-0 flex-1" style={{ overflowWrap: "anywhere" }}>{x.name}</b><StateChip state={x.state} /></div>
              <Progress done={x.done} total={x.total} />
              {x.runs_on && <div className="help">{t.msg(x.runs_on)}</div>}
              {(x.warnings || []).map((w, i) => <div key={i} className="help" style={{ color: "var(--warn)" }}>{t.msg(w)}</div>)}
              {x.error && <Failure m={x.error} />}
            </div>
          ))}
        </div>
      )}

      {data.trials > 0 && (
        <div className="panel space-y-2" aria-live="polite">
          <h2>{t("hb_conclusion")}{active ? <span className="help"> · {t("hb_so_far")}</span> : null}</h2>
          <p>{conclusion.summary[lang]}</p>
          {Object.keys(conclusion.best_harness_by_model).length > 0 && (
            <div className="flex flex-wrap gap-2">{Object.entries(conclusion.best_harness_by_model).map(([name, h]) => <Chip key={name} className="chip-accent">{t("hb_best", { model: name, harness: h })}</Chip>)}</div>
          )}
        </div>
      )}
      {stale.length > 0 && <div className="banner banner-warn">{stale.map((x) => t(x.removed ? "hb_removed" : "hb_stale", { name: x.name })).join(" ")}</div>}

      <Section title={t("hb_matrix")}>
        {data.trials === 0 ? <Empty>{active ? t("waiting_results") : t("no_results")}</Empty> : (
          <div className="panel scroll-x" style={{ padding: 0 }}>
            <table className="tbl hb-table">
              <thead><tr><th>{t("model")}</th>{harnesses.map((h) => <th key={h} scope="col">{h}</th>)}<th className="r">{t("hb_mean")}</th></tr></thead>
              <tbody>
                {models.map((x) => (
                  <tr key={x.id}>
                    <th scope="row" style={{ overflowWrap: "anywhere", color: "var(--ink)" }}>{x.name}{x.stale && <> <Chip className="chip-amber">{t("hb_stale_chip")}</Chip></>}</th>
                    {harnesses.map((h) => <HeatCell key={h} c={data.cells[x.id]?.[h]} k={m.k} selected={pick?.model === x.id && pick?.harness === h} onPick={() => setPick(pick?.model === x.id && pick?.harness === h ? null : { model: x.id, harness: h })} />)}
                    <td className="r num">{data.by_model[x.id] == null ? "—" : pct(data.by_model[x.id], 0, lang)}</td>
                  </tr>
                ))}
                <tr>
                  <th scope="row">{t("hb_mean")}</th>
                  {harnesses.map((h) => <td key={h} className="num">{data.by_harness[h] == null ? "—" : pct(data.by_harness[h], 0, lang)}</td>)}
                  <td />
                </tr>
              </tbody>
            </table>
          </div>
        )}
        <p className="help">{t("hb_legend")}</p>
      </Section>

      {pick && <CellDetail key={`${pick.model}/${pick.harness}`} matrix={id} model={models.find((x) => x.id === pick.model)} harness={pick.harness} onClose={() => setPick(null)} />}
    </div>
  );
}

export default function Arneses({ param }) {
  const { t } = useApp();
  if (param) return <MatrixDetail key={param} id={param} />;
  return (
    <div className="space-y-5">
      <h1>{t("nav_harness")}</h1>
      <p className="help">{t("hb_help")}</p>
      <NewMatrix />
      <History />
    </div>
  );
}
