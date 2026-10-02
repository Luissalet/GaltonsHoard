import React, { useEffect, useMemo, useState } from "react";
import { api } from "../api.js";
import { useApp } from "../context.js";
import { Busy, Chip, ErrorBox, Field, Icon, ICONS, Section, Spinner, Switch, useBusy, useLoad } from "../components/ui.jsx";
import { GpuStrip } from "../components/parts.jsx";
import { gb } from "../format.js";

const POLICY_FIELDS = ["days", "min_tok_s", "max_vram_gb", "weight_quality", "weight_speed", "min_cases", "top_k"];

// One control per setting kind. `value` is the draft value; `onChange` receives the new one.
function Control({ spec, value, onChange, label, hint }) {
  const { t } = useApp();
  if (spec.kind === "bool") {
    return (
      <label className="flex items-center gap-3">
        <Switch checked={!!value} onChange={onChange} label={label} />
        <span><span className="block">{label}</span>{hint && <span className="help block">{hint}</span>}</span>
      </label>
    );
  }
  if (spec.kind === "enum") {
    return <Field label={label} hint={hint}><select className="field" value={value} onChange={(e) => onChange(e.target.value)}>{spec.choices.map((c) => <option key={c} value={c}>{c}</option>)}</select></Field>;
  }
  if (spec.kind === "int" || spec.kind === "float" || spec.kind === "hour") {
    return <Field label={label} hint={hint}><input className="field" type="number" min={spec.min ?? undefined} max={spec.max ?? undefined} step={spec.kind === "int" || spec.kind === "hour" ? 1 : "any"} value={value ?? ""} onChange={(e) => onChange(e.target.value)} /></Field>;
  }
  if (spec.kind === "str_list") {
    return <Field label={label} hint={hint || t("one_per_line")}><textarea className="field mono" rows={3} value={Array.isArray(value) ? value.join("\n") : value} onChange={(e) => onChange(e.target.value)} /></Field>;
  }
  if (spec.kind === "int_list") {
    return <Field label={label} hint={hint}><input className="field" value={Array.isArray(value) ? value.join(", ") : value} onChange={(e) => onChange(e.target.value)} /></Field>;
  }
  if (spec.kind === "secret") {
    return <Field label={label} hint={hint}><input className="field" type="password" autoComplete="off" value={value ?? ""} placeholder={spec.configured ? t("secret_kept") : ""} onChange={(e) => onChange(e.target.value)} /></Field>;
  }
  return <Field label={label} hint={hint}><input className={`field ${/path|dir|folder/.test(spec.key) ? "mono" : ""}`} value={value ?? ""} onChange={(e) => onChange(e.target.value)} /></Field>;
}

export default function Ajustes() {
  const { t, lang, version, changed, notify, confirm } = useApp();
  const settings = useLoad(() => api.call("settings_get", {}), [version]);
  const gpuLoad = useLoad(() => api.call("gpu_status", {}), [version]);
  const modelsLoad = useLoad(() => api.call("models_list", { limit: 300 }), [version]);
  const statusLoad = useLoad(() => api.call("galton_status", {}), [version]);
  const [draft, setDraft] = useState({});
  const [busy, run] = useBusy();

  const values = settings.data?.values || {};
  const specs = useMemo(() => Object.fromEntries((settings.data?.spec || []).map((s) => [s.key, { ...s, configured: values[s.key]?.configured }])), [settings.data]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { setDraft({}); }, [settings.data]);

  const current = (key) => (key in draft ? draft[key] : (specs[key]?.kind === "secret" ? "" : values[key]));
  const set = (key) => (v) => setDraft((d) => ({ ...d, [key]: v }));
  const setPolicy = (name) => (v) => setDraft((d) => ({ ...d, "routes.policy": { ...(d["routes.policy"] || values["routes.policy"]), [name]: v } }));
  const dirty = (keys) => keys.some((k) => k in draft);

  const save = (keys, extra = {}) => run(`save-${keys[0]}`, async () => {
    const payload = {};
    for (const k of keys) if (k in draft) payload[k] = draft[k];
    if (!Object.keys(payload).length) return;
    await api.call("settings_set", { values: payload, ...extra });
    notify(t("saved"));
    changed();
    settings.reload();
  });

  const gpuList = gpuLoad.data?.gpus || [];
  const reserved = new Set(values["gpus.reserved"] || []);
  const allowedDraft = new Set((current("gpus.allowed") ?? values["gpus.allowed"] ?? []).map?.(Number) ?? []);
  const toggleGpu = (index) => {
    const next = new Set(allowedDraft);
    next.has(index) ? next.delete(index) : next.add(index);
    set("gpus.allowed")([...next].sort((a, b) => a - b));
  };
  const saveGpus = () => run("gpus", async () => {
    const wanted = (draft["gpus.allowed"] || []).map(Number);
    const newlyReserved = wanted.filter((i) => reserved.has(i) && !(values["gpus.allowed"] || []).includes(i));
    let extra = {};
    if (newlyReserved.length) {
      const ok = await confirm({ title: t("gpu_reserved_title"), message: t("gpu_reserved_msg", { list: newlyReserved.join(", ") }), confirmLabel: t("gpu_allow_anyway") });
      if (!ok) return;
      extra = { confirm_reserved: true };
    }
    const payload = { "gpus.allowed": wanted };
    if ("gpus.reserved" in draft) payload["gpus.reserved"] = draft["gpus.reserved"];
    await api.call("settings_set", { values: payload, ...extra });
    notify(t("saved"));
    changed();
    settings.reload();
    gpuLoad.reload();
  });

  const housekeeping = () => run("house", async () => {
    const r = await api.call("housekeeping_run", {});
    notify(t("housekeeping_done", { n: Object.values(r).filter((v) => typeof v === "number").reduce((a, b) => a + b, 0) }));
  });
  const toggleRemote = (m, on) => run(`remote-${m.id}`, async () => { await api.call("model_update", { model: m.id, remote_ok: on }); changed(); modelsLoad.reload(); });

  if (settings.loading && !settings.data) return <div className="space-y-3"><h1>{t("nav_settings")}</h1><Spinner /></div>;
  if (!settings.data) return <div className="space-y-3"><h1>{t("nav_settings")}</h1><ErrorBox error={settings.error} /></div>;

  const field = (key, label, hint) => <Control key={key} spec={specs[key]} value={current(key)} onChange={set(key)} label={label} hint={hint} />;
  const policy = draft["routes.policy"] || values["routes.policy"] || {};
  const judgeModels = (modelsLoad.data?.models || []).filter((m) => m.enabled && !m.missing);
  const remotes = (modelsLoad.data?.models || []).filter((m) => m.remote);
  const llama = statusLoad.data?.llama_server;
  const saveBar = (keys, onClick) => (
    <div className="flex justify-end"><Busy className="btn btn-primary" busy={busy[`save-${keys[0]}`] || (onClick && busy.gpus)} disabled={!dirty(keys)} onClick={onClick || (() => save(keys))}>{t("save")}</Busy></div>
  );

  return (
    <div className="space-y-6">
      <h1>{t("nav_settings")}</h1>
      <ErrorBox error={settings.error} />

      <Section title={t("gpus_section")}>
        <div className="panel space-y-3">
          <p className="help">{t("gpus_help")}</p>
          <GpuStrip gpus={gpuLoad.data} />
          <div className="flex flex-wrap gap-4" role="group" aria-label={t("gpus_allowed")}>
            {(gpuList.length ? gpuList : [0, 1, 2, 3].map((index) => ({ index, name: "" }))).map((g) => (
              <label key={g.index} className="flex items-center gap-2" style={{ cursor: "pointer" }}>
                <input type="checkbox" checked={allowedDraft.has(g.index)} onChange={() => toggleGpu(g.index)} />
                <span>GPU {g.index}</span>
                {reserved.has(g.index) && <Chip className="chip-amber" title={t("gpu_reserved_hint")}>{t("gpu_reserved")}</Chip>}
              </label>
            ))}
          </div>
          {field("gpus.reserved", t("gpus_reserved_label"), t("gpus_reserved_hint"))}
          {saveBar(["gpus.allowed", "gpus.reserved"], saveGpus)}
        </div>
      </Section>

      <Section title={t("llama_section")}>
        <div className="panel space-y-3">
          {llama && <div className="flex flex-wrap items-center gap-2"><Chip className={llama.found ? "chip-ok" : "chip-danger"}>{llama.found ? t("llama_found") : t("llama_missing")}</Chip><span className="mono">{llama.path}</span></div>}
          {field("llama.server_path", t("llama_path"))}
          {field("llama.extra_args", t("llama_args"), t("llama_args_hint"))}
          {field("llama.load_timeout_s", t("llama_timeout"))}
          {saveBar(["llama.server_path", "llama.extra_args", "llama.load_timeout_s"])}
        </div>
      </Section>

      <Section title={t("discovery_section")}>
        <div className="panel space-y-3">
          {field("gguf.folders", t("gguf_folders"))}
          {field("ollama.url", t("ollama_url"))}
          {field("ollama.models_dir", t("ollama_dir"), t("ollama_dir_hint"))}
          {field("faustus.url", t("registry_url"))}
          {field("faustus.token", t("registry_token"))}
          {saveBar(["gguf.folders", "ollama.url", "ollama.models_dir", "faustus.url", "faustus.token"])}
        </div>
      </Section>

      <Section title={t("judge_section")}>
        <div className="panel space-y-3">
          <p className="help">{t("judge_help")}</p>
          <Field label={t("judge_model")}>
            <select className="field" value={current("judge.contestant") ?? ""} onChange={(e) => set("judge.contestant")(e.target.value)}>
              <option value="">{t("judge_none")}</option>
              {judgeModels.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
            </select>
          </Field>
          {saveBar(["judge.contestant"])}
        </div>
      </Section>

      <Section title={t("runner_section")}>
        <div className="panel space-y-3">
          <div className="grid gap-3 md:grid-cols-3">
            {field("runner.timeout_s", t("runner_timeout"))}
            {field("runner.idle_grace_s", t("runner_idle_grace"), t("runner_idle_grace_hint"))}
            {field("runner.wait_idle_max_s", t("runner_wait_idle_max"), t("runner_wait_idle_max_hint"))}
            {field("runner.gpu_wait_s", t("runner_gpu_wait"))}
            {field("runner.context", t("runner_context"), t("runner_context_hint"))}
            {field("runner.reasoning_tokens", t("runner_reasoning"), t("runner_reasoning_hint"))}
            {field("runner.check_threads", t("runner_threads"))}
          </div>
          {field("runner.allow_ollama_load", t("runner_ollama_load"), t("runner_ollama_load_hint"))}
          {field("checks.allow_code_execution", t("code_exec"), t("code_exec_hint"))}
          {field("checks.code_timeout_s", t("code_timeout"))}
          {saveBar(["runner.timeout_s", "runner.idle_grace_s", "runner.wait_idle_max_s", "runner.gpu_wait_s", "runner.context", "runner.reasoning_tokens", "runner.check_threads", "runner.allow_ollama_load", "checks.allow_code_execution", "checks.code_timeout_s"])}
        </div>
      </Section>

      <Section title={t("watch_section")}>
        <div className="panel space-y-3">
          <p className="help">{t("watch_help")}</p>
          {field("watch.enabled", t("watch_enabled"))}
          {field("watch.auto_smoke", t("watch_smoke"), t("watch_smoke_hint"))}
          <div className="grid gap-3 md:grid-cols-4">
            {field("watch.interval_h", t("watch_interval"))}
            {field("watch.idle_min", t("watch_idle"))}
            {field("watch.quiet_from", t("quiet_from"))}
            {field("watch.quiet_to", t("quiet_to"))}
          </div>
          {field("scheduler.paused", t("scheduler_paused"))}
          {saveBar(["watch.enabled", "watch.auto_smoke", "watch.interval_h", "watch.idle_min", "watch.quiet_from", "watch.quiet_to", "scheduler.paused"])}
        </div>
      </Section>

      <Section title={t("policy_section")}>
        <div className="panel space-y-3">
          <p className="help">{t("policy_help")}</p>
          <div className="grid gap-3 md:grid-cols-4">
            {POLICY_FIELDS.map((name) => (
              <Field key={name} label={t(`policy_${name}`)}><input className="field" type="number" step="any" value={policy[name] ?? ""} onChange={(e) => setPolicy(name)(e.target.value)} /></Field>
            ))}
          </div>
          <label className="flex items-center gap-3"><Switch checked={!!policy.include_remote} onChange={setPolicy("include_remote")} label={t("policy_include_remote")} /><span>{t("policy_include_remote")}</span></label>
          {field("arena.min_votes", t("arena_min_votes"))}
          {saveBar(["routes.policy", "arena.min_votes"])}
        </div>
      </Section>

      <Section title={t("remote_section")}>
        <div className="panel space-y-2">
          <p className="help">{t("remote_help")}</p>
          {!remotes.length && <div className="help">{t("remote_none")}</div>}
          {remotes.map((m) => (
            <label key={m.id} className="flex items-center gap-3">
              <Switch checked={m.remote_ok} disabled={busy[`remote-${m.id}`]} onChange={(v) => toggleRemote(m, v)} label={m.name} />
              <span><b>{m.name}</b> <span className="help">{m.url}</span></span>
            </label>
          ))}
        </div>
      </Section>

      <Section title={t("about_section")}>
        <div className="panel space-y-2">
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1">
            <dt className="help">{t("version")}</dt><dd>{statusLoad.data?.version}</dd>
            <dt className="help">{t("data_dir")}</dt><dd className="mono">{statusLoad.data?.data_dir}</dd>
            <dt className="help">{t("routes_file")}</dt><dd className="mono">{statusLoad.data?.routes_path}</dd>
          </dl>
          <Busy className="btn" busy={busy.house} onClick={housekeeping}><Icon d={ICONS.refresh} size={14} />{t("housekeeping")}</Busy>
          <div className="help">{t("housekeeping_help")}</div>
        </div>
      </Section>
    </div>
  );
}
