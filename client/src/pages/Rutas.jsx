import React, { useState } from "react";
import { api } from "../api.js";
import { useApp } from "../context.js";
import { Busy, Chip, Empty, ErrorBox, Field, Icon, ICONS, Rel, Section, Spinner, useBusy, useLoad } from "../components/ui.jsx";
import { CiBar, CpuBadge, Score } from "../components/parts.jsx";
import { num } from "../format.js";

function DiffSummary({ diff }) {
  const { t } = useApp();
  if (!diff) return null;
  if (!diff.has_changes) return <span className="help">{t("routes_nochange")}</span>;
  return (
    <div className="space-y-1">
      {diff.added.length > 0 && <div><Chip className="chip-ok">{t("diff_added")}</Chip> {diff.added.map((c) => t(`task_${c}`)).join(", ")}</div>}
      {diff.removed.length > 0 && <div><Chip className="chip-danger">{t("diff_removed")}</Chip> {diff.removed.map((c) => t(`task_${c}`)).join(", ")}</div>}
      {diff.changed.map((c) => (
        <div key={c.task}><Chip className="chip-amber">{c.numbers_only ? t("diff_numbers") : t("diff_changed")}</Chip> <b>{t(`task_${c.task}`)}</b>: {c.from} {c.numbers_only ? "" : `→ ${c.to}`}</div>
      ))}
      {diff.unchanged > 0 && <div className="help">{t("diff_unchanged", { n: diff.unchanged })}</div>}
    </div>
  );
}

export default function Rutas() {
  const { t, lang, version, changed, notify, confirm } = useApp();
  const { data, error, loading, reload } = useLoad(() => api.call("routes_get", {}), [version]);
  const [note, setNote] = useState("");
  const [busy, run] = useBusy();

  const publish = () => run("publish", async () => {
    if (!data.diff.has_changes && data.published) { notify(t("routes_nochange")); return; }
    if (!(await confirm({ title: t("routes_publish"), message: t("routes_publish_file", { path: data.path }), confirmLabel: t("routes_publish"), danger: false }))) return;
    await api.call("routes_publish", { note });
    setNote("");
    notify(t("routes_published"));
    changed();
    reload();
  });

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-2">
        <h1 className="mr-auto">{t("nav_routes")}</h1>
        <Busy className="btn" busy={false} onClick={reload}><Icon d={ICONS.refresh} size={14} />{t("reload")}</Busy>
      </div>
      <p className="help">{t("routes_help")}</p>
      <ErrorBox error={error} />
      {loading && !data && <Spinner />}
      {data && (
        <>
          <div className="panel space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              <b>{t("routes_file")}</b><span className="mono">{data.path}</span>
              {data.published ? <Chip className="chip-ok">{t("published")} <Rel ts={data.published?.updated_at ? Date.parse(data.published.updated_at) / 1000 : null} /></Chip> : <Chip>{t("not_published")}</Chip>}
            </div>
            <Section title={t("diff_title")}><DiffSummary diff={data.diff} /></Section>
            <div className="flex flex-wrap items-end gap-2">
              <div className="min-w-0 flex-1" style={{ maxWidth: 420 }}><Field label={t("publish_note")}><input className="field" value={note} onChange={(e) => setNote(e.target.value)} maxLength={200} /></Field></div>
              <Busy className="btn btn-primary" busy={busy.publish} onClick={publish} disabled={!Object.keys(data.proposed?.tasks || {}).length}><Icon d={ICONS.publish} size={14} />{t("routes_publish")}</Busy>
            </div>
          </div>

          <Section title={t("routes_table")}>
            <div className="space-y-3">
              {data.tasks.map((r) => {
                const detail = data.detail?.[r.category] || {};
                return (
                  <div key={r.category} className="panel space-y-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <b style={{ fontSize: 14 }}>{t(`task_${r.category}`)}</b>
                      {r.published ? <Chip className={r.published_name === r.winner?.names?.[0] ? "chip-ok" : "chip-amber"}>{r.published_name === r.winner?.names?.[0] ? t("published_same") : t("published_other", { name: r.published_name })}</Chip> : <Chip>{t("not_published")}</Chip>}
                      <span className="help ml-auto">{t("candidates_n", { n: r.candidates })}</span>
                    </div>
                    {r.winner ? (
                      <>
                        <div className="grid gap-x-4 gap-y-1" style={{ gridTemplateColumns: "minmax(0, 1.4fr) minmax(0, 1fr) auto auto" }}>
                          {[r.winner, ...r.alternatives].map((x, i) => (
                            <React.Fragment key={x.names[0]}>
                              <div style={{ overflowWrap: "anywhere" }}>{i === 0 ? <Chip className="chip-accent">1</Chip> : <Chip>{i + 1}</Chip>} {i === 0 ? <b>{x.names[0]}</b> : x.names[0]}</div>
                              <div><Score value={x.score} ci={x.ci} n={x.n} /><CiBar value={x.score} ci={x.ci} /></div>
                              <div className="num help">{x.tok_s != null ? `${num(x.tok_s, 0, lang)} tok/s` : "—"}{x.cpu && x.tok_s != null && <> <CpuBadge /></>}</div>
                              <div className="num help">{x.vram_gb != null ? `${num(x.vram_gb, 1, lang)} GB` : "—"}</div>
                            </React.Fragment>
                          ))}
                        </div>
                        {lang === "en" && <p>{r.explain}</p>}
                        {r.winner.names.length > 1 && <div className="help">{t("aliases")}: {r.winner.names.join(", ")}</div>}
                      </>
                    ) : <div className="help">{t("no_winner", { n: r.candidates })}</div>}
                    {detail.warnings?.map((w) => <div key={w.contestant} className="banner banner-warn">{t.msg(w.why)}</div>)}
                    {r.excluded?.length > 0 && (
                      <details><summary className="help">{t("excluded_n", { n: r.excluded.length })}</summary>
                        <ul className="help mt-1" style={{ paddingLeft: 18, listStyle: "disc" }}>{r.excluded.map((x) => <li key={x.contestant}><b>{x.name}</b>: {t.msg(x.why)}</li>)}</ul>
                      </details>
                    )}
                    {detail.comparison && <div className="help">{t("runner_comparison", { verdict: t(`verdict_${detail.comparison.verdict}`), p: detail.comparison.p_value != null ? num(detail.comparison.p_value, 3, lang) : "—", n: detail.comparison.n })}</div>}
                  </div>
                );
              })}
            </div>
          </Section>

          {data.history?.length > 0 && (
            <Section title={t("history")}>
              <div className="panel scroll-x" style={{ padding: 0 }}>
                <table className="tbl">
                  <thead><tr><th>{t("when")}</th><th>{t("changes")}</th><th>{t("notes")}</th></tr></thead>
                  <tbody>{data.history.map((h) => (
                    <tr key={h.id}><td><Rel ts={h.ts} /></td><td><DiffSummary diff={h.diff} /></td><td className="help">{h.note}</td></tr>
                  ))}</tbody>
                </table>
              </div>
            </Section>
          )}
          {data.proposed && <details><summary className="help">{t("routes_json")}</summary><pre className="pre mt-2">{JSON.stringify(data.proposed, null, 2)}</pre></details>}
          {!Object.keys(data.proposed?.tasks || {}).length && <Empty>{t("routes_empty")}</Empty>}
        </>
      )}
    </div>
  );
}
