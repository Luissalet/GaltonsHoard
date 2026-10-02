import React, { useState } from "react";
import { api } from "../api.js";
import { useApp } from "../context.js";
import { Busy, Chip, Drawer, Empty, ErrorBox, Field, Icon, ICONS, Modal, Rel, Section, Spinner, Switch, Tabs, useBusy, useLoad } from "../components/ui.jsx";
import { MemoryChip, Score } from "../components/parts.jsx";
import { fileSize, gb, num, splitList } from "../format.js";

function StateChips({ m }) {
  const { t } = useApp();
  return (
    <span className="inline-flex flex-wrap gap-1">
      {!m.enabled && <Chip>{t("disabled")}</Chip>}
      {m.missing && <Chip className="chip-danger">{t("missing_model")}</Chip>}
      {m.not_chat && <Chip className="chip-amber" title={t.msg(m.not_chat_reason)}>{t("not_chat")}</Chip>}
      {m.not_served && <Chip className="chip-danger" title={t.msg(m.not_served_reason)}>{t("not_served")}</Chip>}
      {m.stale && <Chip className="chip-danger">{t("stale_model")}</Chip>}
      {m.never_measured && !m.missing && <Chip className="chip-amber">{t("never_measured")}</Chip>}
      {m.kind === "server" && m.up === true && <Chip className="chip-ok">{m.resident ? t("resident") : t("up")}</Chip>}
      {m.kind === "server" && m.up === false && !m.not_served && <Chip className="chip-danger">{t("down")}</Chip>}
      {m.busy && <Chip className="chip-amber">{t("busy")}</Chip>}
      {m.remote && <Chip className="chip-amber">{t("remote")}</Chip>}
      {m.demo && <Chip>{t("demo")}</Chip>}
    </span>
  );
}

function whereText(m, t) {
  if (m.kind === "gguf") return t("where_own");
  try {
    return new URL(m.url).host;
  } catch {
    return m.url || "—";
  }
}

function AddModal({ onClose, onDone }) {
  const { t, notify } = useApp();
  const [tab, setTab] = useState("server");
  const [form, setForm] = useState({ url: "", model: "", api: "", path: "", mmproj: "", name: "", remote_ok: false });
  const [busy, run] = useBusy();
  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }));
  const submit = (e) => {
    e.preventDefault();
    run("add", async () => {
      const args = tab === "server"
        ? { url: form.url.trim(), model: form.model.trim(), api: form.api, name: form.name.trim(), remote_ok: form.remote_ok }
        : { path: form.path.trim(), mmproj: form.mmproj.trim(), name: form.name.trim() };
      const r = await api.call("model_add", args);
      notify(t("model_added", { name: r.model?.name || "" }));
      onDone();
      onClose();
    });
  };
  return (
    <Modal title={t("add_model")} onClose={onClose} wide>
      <Tabs value={tab} onChange={setTab} tabs={[{ id: "server", label: t("add_server") }, { id: "gguf", label: t("add_gguf") }]} />
      <form className="space-y-3" onSubmit={submit}>
        {tab === "server" ? (
          <>
            <Field label={t("url")} hint={t("url_hint")}><input className="field" value={form.url} onChange={set("url")} placeholder="http://127.0.0.1:8081" required /></Field>
            <div className="grid grid-cols-2 gap-3">
              <Field label={t("model_name_on_server")}><input className="field" value={form.model} onChange={set("model")} /></Field>
              <Field label={t("api")}>
                <select className="field" value={form.api} onChange={set("api")}>
                  <option value="">{t("api_auto")}</option>
                  <option value="chat">chat completions</option>
                  <option value="ollama">Ollama</option>
                </select>
              </Field>
            </div>
            <label className="flex items-center gap-2"><Switch checked={form.remote_ok} onChange={(v) => setForm((f) => ({ ...f, remote_ok: v }))} label={t("remote_ok")} /><span>{t("remote_ok")}</span></label>
            <div className="help">{t("remote_warning")}</div>
          </>
        ) : (
          <>
            <Field label={t("gguf_path")} hint={t("gguf_path_hint")}><input className="field mono" value={form.path} onChange={set("path")} placeholder="D:\LocalAI\models\modelo.gguf" required /></Field>
            <Field label={t("mmproj_path")} hint={t("mmproj_hint")}><input className="field mono" value={form.mmproj} onChange={set("mmproj")} /></Field>
          </>
        )}
        <Field label={t("display_name")}><input className="field" value={form.name} onChange={set("name")} /></Field>
        <div className="flex justify-end gap-2">
          <button type="button" className="btn" onClick={onClose}>{t("cancel")}</button>
          <Busy type="submit" className="btn btn-primary" busy={busy.add}>{t("add")}</Busy>
        </div>
      </form>
    </Modal>
  );
}

function Detail({ id, onClose, onChanged }) {
  const { t, lang, confirm, notify } = useApp();
  const { data, error, loading, reload } = useLoad(() => api.call("model_get", { model: id }), [id]);
  const [busy, run] = useBusy();
  const [edit, setEdit] = useState(null);
  const m = data?.model;
  const form = edit || (m ? { name: m.name, aliases: m.aliases.join("\n"), vision: m.vision, mmproj: m.mmproj || "", remote_ok: m.remote_ok } : null);
  const setField = (k, v) => setEdit({ ...form, [k]: v });

  const save = () => run("save", async () => {
    await api.call("model_update", { model: id, name: form.name, aliases: splitList(form.aliases), vision: form.vision, mmproj: form.mmproj, remote_ok: form.remote_ok });
    setEdit(null);
    notify(t("saved"));
    await reload();
    onChanged();
  });
  const toggle = (enabled) => run("toggle", async () => { await api.call("model_update", { model: id, enabled }); await reload(); onChanged(); });
  const remove = () => run("remove", async () => {
    if (!(await confirm({ title: t("remove_model"), message: t("remove_model_msg", { name: m.name }), confirmLabel: t("remove") }))) return;
    await api.call("model_remove", { model: id, confirm: true });
    notify(t("deleted"));
    onChanged();
    onClose();
  });

  return (
    <Drawer title={m?.name || "…"} onClose={onClose}>
      {loading && !data && <Spinner />}
      <ErrorBox error={error} />
      {m && (
        <>
          <div className="flex flex-wrap items-center gap-2">
            <StateChips m={m} />
            <span className="ml-auto flex items-center gap-2"><span className="help">{t("enabled")}</span><Switch checked={m.enabled} onChange={toggle} label={t("enabled")} /></span>
          </div>
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1">
            <dt className="help">{t("kind")}</dt><dd>{m.kind} · {m.provider}</dd>
            <dt className="help">{t("family")}</dt><dd>{m.family || "—"}</dd>
            <dt className="help">{t("params")}</dt><dd>{m.params_b ? `${num(m.params_b, 1, lang)} B` : "—"} · {m.quant || "—"}</dd>
            <dt className="help">{t("context")}</dt><dd>{m.context ? num(m.context, 0, lang) : "—"}</dd>
            <dt className="help">{t("memory")}</dt><dd><MemoryChip model={m} /></dd>
            {m.size_bytes ? (<><dt className="help">{t("file_size")}</dt><dd>{fileSize(m.size_bytes)}</dd></>) : null}
            {m.path && (<><dt className="help">{t("path")}</dt><dd className="mono">{m.path}</dd></>)}
            {m.url && (<><dt className="help">URL</dt><dd className="mono">{m.url}</dd></>)}
            <dt className="help">{t("digest")}</dt><dd className="mono">{m.digest || "—"}</dd>
            {data.where && (<><dt className="help">{t("where_runs")}</dt><dd>{t.msg(data.where.where) || data.where.runs_on}{data.where.vram_mb ? ` · ${gb(data.where.vram_mb, lang)}` : ""}</dd></>)}
          </dl>
          {data.where?.warnings?.map((w, i) => <div key={i} className="banner banner-warn">{t.msg(w)}</div>)}
          {m.not_chat && <div className="banner banner-warn">{t.msg(m.not_chat_reason)}</div>}
          {m.not_served && <div className="banner banner-warn">{t.msg(m.not_served_reason)}</div>}
          {data.same_weights?.length > 0 && (
            <div className="help">{t("same_weights_as")}: {data.same_weights.map((s, i) => (
              <span key={s.id}>{i > 0 && ", "}<b>{s.ollama_ref || s.name}</b> ({s.kind === "server" ? s.url : s.source})</span>
            ))}</div>
          )}
          {data.scores && (
            <Section title={t("results")}>
              <div className="flex flex-wrap items-center gap-3"><Score value={data.scores.score} ci={data.scores.ci} n={data.scores.n} /> <a className="btn-link" href={`#/clasificacion`}>{t("nav_board")}</a></div>
              <div className="flex flex-wrap gap-2">
                {Object.entries(data.scores.by_category || {}).map(([k, v]) => <Chip key={k}>{t(`cat_${k}`)} {num(v.score, 2, lang)} · {v.n}</Chip>)}
              </div>
            </Section>
          )}
          <Section title={t("edit")}>
            <div className="space-y-3">
              <Field label={t("display_name")}><input className="field" value={form.name} onChange={(e) => setField("name", e.target.value)} /></Field>
              <Field label={t("aliases")} hint={t("aliases_hint")}><textarea className="field" rows={Math.min(8, Math.max(3, form.aliases.split("\n").length + 1))} value={form.aliases} onChange={(e) => setField("aliases", e.target.value)} /></Field>
              <label className="flex items-center gap-2"><Switch checked={!!form.vision} onChange={(v) => setField("vision", v)} label={t("vision")} /><span>{t("vision")}</span></label>
              {m.kind === "gguf" && <Field label={t("mmproj_path")}><input className="field mono" value={form.mmproj} onChange={(e) => setField("mmproj", e.target.value)} /></Field>}
              {m.kind === "server" && <label className="flex items-center gap-2"><Switch checked={!!form.remote_ok} onChange={(v) => setField("remote_ok", v)} label={t("remote_ok")} /><span>{t("remote_ok")}</span></label>}
              <div className="flex flex-wrap gap-2">
                <Busy className="btn btn-primary" busy={busy.save} disabled={!edit} onClick={save}>{t("save")}</Busy>
                {!m.not_served && <a className="btn" href={`#/ejecutar?model=${m.id}`}><Icon d={ICONS.run} size={14} />{t("measure")}</a>}
                <Busy className="btn btn-danger ml-auto" busy={busy.remove} onClick={remove}><Icon d={ICONS.trash} size={14} />{t("remove")}</Busy>
              </div>
            </div>
          </Section>
          {data.measurements?.length > 0 && (
            <Section title={t("measurements")}>
              <div className="scroll-x"><table className="tbl">
                <thead><tr><th>{t("run")}</th><th>{t("runs_on")}</th><th className="r">{t("load")}</th><th className="r">VRAM</th></tr></thead>
                <tbody>{data.measurements.map((x) => (
                  <tr key={x.run_id}><td><a href={`#/ejecutar/${x.run_id}`}>{x.run_id.slice(0, 10)}</a> <Rel ts={x.finished_ts} /></td><td>{t.msg(x.runs_on) || "—"}</td><td className="r num">{x.load_ms != null ? `${num(x.load_ms / 1000, 1, lang)} s` : "—"}</td><td className="r num">{x.vram_mb ? gb(x.vram_mb, lang) : "—"}</td></tr>
                ))}</tbody>
              </table></div>
            </Section>
          )}
          {data.meta && <details><summary className="help">{t("metadata")}</summary><pre className="pre mt-2">{JSON.stringify(data.meta, null, 2)}</pre></details>}
        </>
      )}
    </Drawer>
  );
}

export default function Modelos() {
  const { t, lang, version, changed, notify } = useApp();
  const [filters, setFilters] = useState({ text: "", kind: "", measured: "any", vision: "" });
  const [adding, setAdding] = useState(false);
  const [open, setOpen] = useState(null);
  const [busy, run] = useBusy();
  const args = { text: filters.text, kind: filters.kind, measured: filters.measured, include_missing: true, ...(filters.vision ? { vision: filters.vision === "yes" } : {}) };
  const { data, error, loading, reload } = useLoad(() => api.call("models_list", args), [version, filters.text, filters.kind, filters.measured, filters.vision]);
  const set = (k) => (e) => setFilters((f) => ({ ...f, [k]: e.target.value }));
  const models = data?.models || [];

  const refresh = () => run("refresh", async () => {
    const r = await api.call("models_refresh", {});
    notify(t("refresh_result", { n: r.new?.length ?? 0, m: r.changed?.length ?? 0 }) + (r.errors?.length ? ` · ${t.msg(r.errors[0])}` : ""));
    changed();
  });
  const toggle = (m, enabled) => run(`t${m.id}`, async () => { await api.call("model_update", { model: m.id, enabled }); changed(); });

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <h1 className="mr-auto">{t("nav_models")}</h1>
        <Busy className="btn" busy={busy.refresh} onClick={refresh}><Icon d={ICONS.refresh} size={14} />{t("refresh_models")}</Busy>
        <button type="button" className="btn btn-primary" onClick={() => setAdding(true)}><Icon d={ICONS.plus} size={14} />{t("add_model")}</button>
      </div>
      <div className="flex flex-wrap items-end gap-2">
        <div style={{ width: 220 }}><Field label={t("search")}><input className="field" value={filters.text} onChange={set("text")} placeholder="qwen, Q4_K_M…" /></Field></div>
        <div style={{ width: 140 }}><Field label={t("kind")}><select className="field" value={filters.kind} onChange={set("kind")}><option value="">{t("all")}</option><option value="server">{t("kind_server")}</option><option value="gguf">GGUF</option></select></Field></div>
        <div style={{ width: 160 }}><Field label={t("measured")}><select className="field" value={filters.measured} onChange={set("measured")}>{["any", "never", "stale", "fresh"].map((v) => <option key={v} value={v}>{t(`measured_${v}`)}</option>)}</select></Field></div>
        <div style={{ width: 120 }}><Field label={t("vision")}><select className="field" value={filters.vision} onChange={set("vision")}><option value="">{t("all")}</option><option value="yes">{t("yes")}</option><option value="no">{t("no")}</option></select></Field></div>
      </div>
      <ErrorBox error={error} />
      {loading && !data && <Spinner />}
      {data && !models.length && <Empty>{t("models_empty")}</Empty>}
      {models.length > 0 && (
        <div className="panel scroll-x" style={{ padding: 0 }}>
          <table className="tbl">
            <thead>
              <tr><th>{t("model")}</th><th>{t("kind")}</th><th>{t("family")}</th><th className="r">{t("params")}</th><th>{t("quant")}</th><th className="r">{t("context")}</th><th>{t("vision")}</th><th>{t("where_runs")}</th><th>{t("status")}</th><th>{t("results")}</th><th>{t("enabled")}</th></tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.id} style={m.enabled ? undefined : { opacity: 0.6 }}>
                  <td style={{ overflowWrap: "anywhere" }}><button type="button" className="btn-link" style={{ textAlign: "left" }} onClick={() => setOpen(m.id)}>{m.name}</button></td>
                  <td>{m.kind === "gguf" ? "GGUF" : m.provider}</td>
                  <td>{m.family || "—"}</td>
                  <td className="r num">{m.params_b ? `${num(m.params_b, 1, lang)} B` : "—"}</td>
                  <td>{m.quant || "—"}</td>
                  <td className="r num">{m.context ? num(m.context, 0, lang) : "—"}</td>
                  <td>{m.vision ? <Chip className="chip-info">{t("vision")}</Chip> : "—"}</td>
                  <td className="trunc" style={{ maxWidth: 170 }} title={m.url || m.path}>{whereText(m, t)}</td>
                  <td><StateChips m={m} /></td>
                  <td>{m.measured_n ? <span className="num">{m.measured_n} · <Rel ts={m.last_measured_ts} /></span> : "—"}</td>
                  <td><Switch checked={m.enabled} disabled={busy[`t${m.id}`]} onChange={(v) => toggle(m, v)} label={`${t("enabled")}: ${m.name}`} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {adding && <AddModal onClose={() => setAdding(false)} onDone={() => { changed(); reload(); }} />}
      {open && <Detail id={open} onClose={() => setOpen(null)} onChanged={changed} />}
    </div>
  );
}
