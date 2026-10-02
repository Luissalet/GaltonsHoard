import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import { useApp } from "../context.js";
import { Busy, Chip, Drawer, Empty, ErrorBox, Field, Icon, ICONS, Modal, Section, Spinner, useBusy, useLoad } from "../components/ui.jsx";
import { CATEGORIES } from "../meta.js";
import { duration, num, splitList } from "../format.js";

// ------------------------------------------------------------------ try a case against a model
function TryPanel({ caseId, draft }) {
  const { t, lang, version } = useApp();
  const { data } = useLoad(() => api.call("models_list", { enabled: true, limit: 200 }), [version]);
  const [model, setModel] = useState("");
  const [busy, run] = useBusy();
  const [out, setOut] = useState(null);
  const models = data?.models || [];
  useEffect(() => { if (!model && models.length) setModel(models[0].id); }, [models, model]);
  const go = () => run("try", async () => {
    setOut(null);
    const args = caseId ? { model, case: caseId } : { model, ...draft() };
    setOut(await api.call("case_try", args));
  });
  return (
    <Section title={t("try_with")}>
      <div className="flex flex-wrap items-end gap-2">
        <div className="min-w-0 flex-1"><select className="field" value={model} onChange={(e) => setModel(e.target.value)} aria-label={t("model")}>{models.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}</select></div>
        <Busy className="btn btn-primary" busy={busy.try} disabled={!model} onClick={go}><Icon d={ICONS.run} size={14} />{t("try")}</Busy>
      </div>
      {out && (
        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            {out.verdict ? <Chip className={out.verdict.passed ? "chip-ok" : "chip-danger"}>{out.verdict.passed ? t("passed") : t("not_passed")} · {num(out.verdict.score, 2, lang)}</Chip> : <Chip>{t("unscored")}</Chip>}
            {out.judge_pending && <Chip className="chip-amber">{t("judge_pending")}</Chip>}
            <Chip>{out.checker}</Chip>
            <span className="help num">{duration(out.latency_ms)} · {out.decode_tps ? `${num(out.decode_tps, 1, lang)} tok/s` : "—"}</span>
          </div>
          {out.notes?.map((n, i) => <div key={i} className="banner banner-warn">{t.msg(n)}</div>)}
          <div className="pre">{out.output || t("empty_output")}</div>
          {out.reasoning && <details><summary className="help">{t("reasoning")}</summary><div className="pre mt-1">{out.reasoning}</div></details>}
          {out.verdict && <details><summary className="help">{t("checker_detail")}</summary><pre className="pre mt-1">{JSON.stringify(out.verdict.detail, null, 2)}</pre></details>}
        </div>
      )}
    </Section>
  );
}

// ------------------------------------------------------------------ case editor (also creates)
function promptParts(prompt) {
  if (!prompt) return { text: "", system: "", messages: null };
  if (typeof prompt === "string") return { text: prompt, system: "", messages: null };
  if (prompt.messages) return { text: "", system: "", messages: prompt.messages };
  return { text: prompt.text || "", system: prompt.system || "", messages: null };
}

function CaseEditor({ suite, caseId, onClose, onSaved }) {
  const { t, notify, confirm } = useApp();
  const editable = !suite.builtin;
  const { data, error, loading } = useLoad(() => (caseId ? api.call("case_get", { case: caseId }) : Promise.resolve(null)), [caseId]);
  const [form, setForm] = useState(null);
  const [dirty, setDirty] = useState(false);
  const [busy, run] = useBusy();
  const c = data?.case;
  useEffect(() => {
    if (!caseId) setForm({ title: "", text: "", system: "", expected: "", checker: "", weight: 1, tags: "", notes: "", messages: null });
    else if (c) {
      const p = promptParts(c.prompt);
      setForm({ title: c.title, text: p.text, system: p.system, expected: c.reference || "", checker: JSON.stringify(c.checker), weight: c.weight, tags: (c.tags || []).join(", "), notes: c.notes || "", messages: p.messages });
    }
  }, [caseId, c]);
  if (!form) return <Drawer title={t("case")} onClose={onClose}>{loading ? <Spinner /> : <ErrorBox error={error} />}</Drawer>;
  const set = (k) => (e) => { setDirty(true); setForm((f) => ({ ...f, [k]: e.target.value })); };
  const checkerArg = () => {
    const raw = form.checker.trim();
    if (!raw) return undefined;
    if (raw.startsWith("{")) {
      try { return JSON.parse(raw); } catch { throw new Error(t("checker_json_invalid")); }
    }
    return raw;
  };
  const save = () => run("save", async () => {
    const common = { title: form.title, expected: form.expected, weight: Number(form.weight) || 1, tags: splitList(form.tags), notes: form.notes };
    const checker = checkerArg();
    if (caseId) {
      const args = { case: caseId, ...common, system: form.system };
      if (!form.messages) args.prompt = form.text;
      if (checker !== undefined && form.checker.trim() !== JSON.stringify(c.checker)) args.checker = checker;
      await api.call("case_update", args);
    } else {
      await api.call("case_add", { suite: suite.id, ...common, prompt: form.text, system: form.system, ...(checker !== undefined ? { checker } : {}) });
    }
    notify(t("saved"));
    onSaved();
    onClose();
  });
  const remove = () => run("remove", async () => {
    if (!(await confirm({ title: t("remove_case"), message: t("remove_case_msg", { title: form.title }), confirmLabel: t("delete") }))) return;
    await api.call("case_remove", { case: caseId, confirm: true });
    notify(t("deleted"));
    onSaved();
    onClose();
  });
  const draft = () => ({ prompt: form.text, system: form.system, expected: form.expected, ...(form.checker.trim() ? { checker: checkerArg() } : {}) });
  return (
    <Drawer title={caseId ? form.title : t("new_case")} onClose={onClose}>
      {!editable && <div className="banner banner-info">{t("builtin_readonly")}</div>}
      {c?.generated && <div className="banner banner-info">{t("generated_case")}</div>}
      <fieldset disabled={!editable} className="space-y-3" style={{ border: 0, padding: 0, margin: 0, minWidth: 0 }}>
        <Field label={t("title")}><input className="field" value={form.title} onChange={set("title")} /></Field>
        {form.messages ? <Field label={t("conversation")}><pre className="pre">{JSON.stringify(form.messages, null, 2)}</pre></Field> : <Field label={t("prompt")}><textarea className="field" rows={7} value={form.text} onChange={set("text")} /></Field>}
        <Field label={t("system_prompt")}><textarea className="field" rows={2} value={form.system} onChange={set("system")} /></Field>
        <Field label={t("expected")} hint={t("expected_hint")}><textarea className="field" rows={2} value={form.expected} onChange={set("expected")} /></Field>
        <Field label={t("checker")} hint={t("checker_hint")}><textarea className="field" rows={3} value={form.checker} onChange={set("checker")} placeholder="contains:Madrid" /></Field>
        <div className="grid grid-cols-2 gap-3">
          <Field label={t("weight")}><input className="field" type="number" min="0.1" step="0.1" value={form.weight} onChange={set("weight")} /></Field>
          <Field label={t("tags")}><input className="field" value={form.tags} onChange={set("tags")} /></Field>
        </div>
        <Field label={t("notes")}><input className="field" value={form.notes} onChange={set("notes")} /></Field>
      </fieldset>
      {editable && (
        <div className="flex flex-wrap gap-2">
          <Busy className="btn btn-primary" busy={busy.save} onClick={save}>{t("save")}</Busy>
          {caseId && <Busy className="btn btn-danger ml-auto" busy={busy.remove} onClick={remove}><Icon d={ICONS.trash} size={14} />{t("delete")}</Busy>}
        </div>
      )}
      {c?.results > 0 && <div className="help">{t("case_results", { n: c.results })}</div>}
      <TryPanel caseId={caseId && (!dirty || form.messages || c?.tools?.length) ? caseId : null} draft={draft} />
    </Drawer>
  );
}

// ------------------------------------------------------------------ import
function ImportModal({ suite, onClose, onDone }) {
  const { t, notify } = useApp();
  const [text, setText] = useState("");
  const [format, setFormat] = useState("auto");
  const [checker, setChecker] = useState("");
  const [busy, run] = useBusy();
  const [result, setResult] = useState(null);
  const onFile = async (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setText(await f.text());
    if (/\.csv$/i.test(f.name)) setFormat("csv");
    else if (/\.jsonl?$/i.test(f.name)) setFormat("jsonl");
  };
  const go = () => run("import", async () => {
    const r = await api.call("cases_import", { suite: suite.id, text, format, ...(checker.trim() ? { default_checker: checker.trim() } : {}) });
    setResult(r);
    notify(t("import_result", { n: r.added }));
    onDone();
  });
  return (
    <Modal title={t("import_cases")} onClose={onClose} wide>
      <div className="help">{t("import_help")}</div>
      <Field label={t("file")}><input className="field" type="file" accept=".jsonl,.json,.csv,.tsv,.txt" onChange={onFile} /></Field>
      <Field label={t("import_text")}><textarea className="field" rows={8} value={text} onChange={(e) => setText(e.target.value)} placeholder='{"prompt": "¿Capital de Francia?", "expected": "París"}' /></Field>
      <div className="grid grid-cols-2 gap-3">
        <Field label={t("format")}><select className="field" value={format} onChange={(e) => setFormat(e.target.value)}><option value="auto">auto</option><option value="jsonl">JSONL</option><option value="csv">CSV</option></select></Field>
        <Field label={t("default_checker")} hint={t("default_checker_hint")}><input className="field" value={checker} onChange={(e) => setChecker(e.target.value)} placeholder="judge:Es correcto y claro." /></Field>
      </div>
      {result && (
        <div className="banner banner-info space-y-1">
          <div>{t("import_result", { n: result.added })}{result.unscored ? ` · ${t("import_unscored", { n: result.unscored })}` : ""}</div>
          {result.duplicates_skipped?.length > 0 && <div className="help">{t("import_duplicates", { n: result.duplicates_skipped.length })}</div>}
          {result.errors?.map((e, i) => <div key={i} className="help">{t("row")} {e.row}: {t.msg(e.error)}</div>)}
        </div>
      )}
      <div className="flex justify-end gap-2">
        <button type="button" className="btn" onClick={onClose}>{result ? t("close") : t("cancel")}</button>
        <Busy className="btn btn-primary" busy={busy.import} disabled={!text.trim()} onClick={go}>{t("import")}</Busy>
      </div>
    </Modal>
  );
}

function NewSuiteModal({ onClose, onDone }) {
  const { t, notify } = useApp();
  const [form, setForm] = useState({ name: "", description: "", category: "custom", max_tokens: 512 });
  const [busy, run] = useBusy();
  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }));
  const submit = (e) => {
    e.preventDefault();
    run("create", async () => {
      const r = await api.call("suite_create", { ...form, max_tokens: Number(form.max_tokens) || 512 });
      notify(t("suite_created"));
      onDone(r.suite);
      onClose();
    });
  };
  return (
    <Modal title={t("new_suite")} onClose={onClose}>
      <form className="space-y-3" onSubmit={submit}>
        <Field label={t("name")}><input className="field" value={form.name} onChange={set("name")} required minLength={2} /></Field>
        <Field label={t("description")}><input className="field" value={form.description} onChange={set("description")} /></Field>
        <div className="grid grid-cols-2 gap-3">
          <Field label={t("category")}><select className="field" value={form.category} onChange={set("category")}>{CATEGORIES.map((c) => <option key={c} value={c}>{t(`cat_${c}`)}</option>)}</select></Field>
          <Field label={t("max_tokens")}><input className="field" type="number" min="16" value={form.max_tokens} onChange={set("max_tokens")} /></Field>
        </div>
        <div className="flex justify-end gap-2">
          <button type="button" className="btn" onClick={onClose}>{t("cancel")}</button>
          <Busy type="submit" className="btn btn-primary" busy={busy.create}>{t("create")}</Busy>
        </div>
      </form>
    </Modal>
  );
}

// ------------------------------------------------------------------ suite detail
function SuiteDetail({ id }) {
  const { t, version, changed, notify, confirm } = useApp();
  const [page, setPage] = useState(0);
  const [editing, setEditing] = useState(undefined);
  const [importing, setImporting] = useState(false);
  const [busy, run] = useBusy();
  const limit = 100;
  const { data, error, loading, reload } = useLoad(() => api.call("suite_get", { suite: id, limit, offset: page * limit }), [id, page, version]);
  const suite = data?.suite;
  const cases = data?.cases || [];
  const after = () => { changed(); reload(); };

  const duplicate = () => run("dup", async () => {
    const r = await api.call("suite_duplicate", { suite: id });
    notify(t("suite_duplicated"));
    changed();
    window.location.hash = `#/pruebas/${r.suite.id}`;
  });
  const remove = () => run("remove", async () => {
    if (!(await confirm({ title: t("remove_suite"), message: t("remove_suite_msg", { name: suite.name }), confirmLabel: t("delete") }))) return;
    await api.call("suite_remove", { suite: id, confirm: true });
    notify(t("deleted"));
    changed();
    window.location.hash = "#/pruebas";
  });

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <a href="#/pruebas" className="btn btn-sm"><Icon d={ICONS.back} size={14} />{t("nav_tests")}</a>
        <h1 className="mr-auto" style={{ overflowWrap: "anywhere" }}>{suite?.name || "…"}</h1>
        {suite && (
          <>
            <button type="button" className="btn" onClick={duplicate}><Icon d={ICONS.copy} size={14} />{t("duplicate")}</button>
            {!suite.builtin && <button type="button" className="btn" onClick={() => setImporting(true)}><Icon d={ICONS.upload} size={14} />{t("import_cases")}</button>}
            {!suite.builtin && <button type="button" className="btn btn-primary" onClick={() => setEditing(null)}><Icon d={ICONS.plus} size={14} />{t("add_case")}</button>}
            <a className="btn" href={`#/ejecutar?suite=${suite.id}`}><Icon d={ICONS.run} size={14} />{t("run_this")}</a>
            {!suite.builtin && <Busy className="btn btn-danger" busy={busy.remove} onClick={remove}><Icon d={ICONS.trash} size={14} />{t("delete")}</Busy>}
          </>
        )}
      </div>
      <ErrorBox error={error} />
      {loading && !data && <Spinner />}
      {suite && (
        <>
          <div className="flex flex-wrap items-center gap-2">
            <Chip className="chip-accent">{t(`cat_${suite.category}`)}</Chip>
            <Chip>{t("n_cases", { n: data.total_cases })}</Chip>
            <Chip>{t("max_tokens")} {suite.max_tokens}</Chip>
            {suite.builtin ? <Chip>{t("builtin")}</Chip> : <Chip className="chip-ok">{t("yours")}</Chip>}
            <Chip>v{suite.version}</Chip>
          </div>
          {suite.description && <p className="help" style={{ fontSize: 13 }}>{suite.description}</p>}
          {!cases.length ? <Empty>{t("suite_empty")}</Empty> : (
            <div className="panel scroll-x" style={{ padding: 0 }}>
              <table className="tbl">
                <thead><tr><th>{t("title")}</th><th>{t("prompt")}</th><th>{t("checker")}</th><th className="r">{t("weight")}</th><th>{t("tags")}</th></tr></thead>
                <tbody>
                  {cases.map((k) => (
                    <tr key={k.id}>
                      <td style={{ minWidth: 140 }}><button type="button" className="btn-link" style={{ textAlign: "left" }} onClick={() => setEditing(k.id)}>{k.title}</button></td>
                      <td className="help" style={{ maxWidth: 420 }}><span className="clamp2">{k.prompt_preview}</span></td>
                      <td><Chip>{k.checker_label}</Chip>{k.has_tools && <Chip>tools</Chip>}{k.has_images && <Chip className="chip-info">img</Chip>}</td>
                      <td className="r num">{k.weight}</td>
                      <td className="help">{(k.tags || []).join(", ")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {data.total_cases > limit && (
            <div className="flex items-center gap-2">
              <button type="button" className="btn btn-sm" disabled={page === 0} onClick={() => setPage(page - 1)}>{t("prev")}</button>
              <span className="help">{page * limit + 1}–{Math.min(data.total_cases, (page + 1) * limit)} / {data.total_cases}</span>
              <button type="button" className="btn btn-sm" disabled={(page + 1) * limit >= data.total_cases} onClick={() => setPage(page + 1)}>{t("next")}</button>
            </div>
          )}
          {editing !== undefined && <CaseEditor suite={suite} caseId={editing} onClose={() => setEditing(undefined)} onSaved={after} />}
          {importing && <ImportModal suite={suite} onClose={() => setImporting(false)} onDone={after} />}
        </>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ list
export default function Pruebas({ param }) {
  const { t, version, changed } = useApp();
  const [category, setCategory] = useState("");
  const [creating, setCreating] = useState(false);
  const { data, error, loading } = useLoad(() => api.call("suites_list", {}), [version]);
  if (param) return <SuiteDetail id={param} />;
  const suites = (data?.suites || []).filter((s) => !category || s.category === category);
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <h1 className="mr-auto">{t("nav_tests")}</h1>
        <button type="button" className="btn btn-primary" onClick={() => setCreating(true)}><Icon d={ICONS.plus} size={14} />{t("new_suite")}</button>
      </div>
      <div style={{ width: 220 }}>
        <Field label={t("category")}><select className="field" value={category} onChange={(e) => setCategory(e.target.value)}><option value="">{t("all")}</option>{CATEGORIES.map((c) => <option key={c} value={c}>{t(`cat_${c}`)}</option>)}</select></Field>
      </div>
      <ErrorBox error={error} />
      {loading && !data && <Spinner />}
      <div className="card-grid">
        {suites.map((s) => (
          <a key={s.id} href={`#/pruebas/${s.id}`} className="panel card block space-y-1.5" style={{ textDecoration: "none", color: "inherit" }}>
            <div className="flex flex-wrap items-center gap-2">
              <b className="min-w-0 flex-1" style={{ overflowWrap: "anywhere" }}>{s.name}</b>
              {!s.builtin && <Chip className="chip-ok">{t("yours")}</Chip>}
            </div>
            <div className="flex flex-wrap gap-1.5"><Chip className="chip-accent">{t(`cat_${s.category}`)}</Chip><Chip>{t("n_cases", { n: s.cases })}</Chip>{s.generated && <Chip className="chip-info">{t("generated")}</Chip>}</div>
            <p className="help clamp2">{s.description}</p>
          </a>
        ))}
      </div>
      {creating && <NewSuiteModal onClose={() => setCreating(false)} onDone={(s) => { changed(); window.location.hash = `#/pruebas/${s.id}`; }} />}
    </div>
  );
}
