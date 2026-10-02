import React from "react";
import { api } from "../api.js";
import { useApp } from "../context.js";
import { Busy, Chip, Empty, Icon, ICONS, Rel, Section, useBusy } from "../components/ui.jsx";
import { CiBar, CpuBadge, GpuStrip, Progress, Score, StateChip } from "../components/parts.jsx";
import { num } from "../format.js";

function RunningCard({ run }) {
  const { t } = useApp();
  if (!run) return null;
  return (
    <a href={`#/ejecutar/${run.id}`} className="panel block space-y-2" style={{ textDecoration: "none", color: "inherit" }}>
      <div className="flex flex-wrap items-center gap-2">
        <StateChip state={run.state} />
        <b className="min-w-0 flex-1" style={{ overflowWrap: "anywhere" }}>{run.label}</b>
        <span className="num help">{run.progress.done}/{run.progress.total} · {num(run.progress.pct, 0)} %</span>
      </div>
      <Progress done={run.progress.done} total={run.progress.total} />
      <div className="help">
        {run.contestants.map((c) => `${c.name} ${c.done}/${c.total}`).join(" · ")}
      </div>
    </a>
  );
}

function Notices({ notices, onSeen }) {
  const { t } = useApp();
  const unseen = notices.filter((n) => !n.seen);
  if (!notices.length) return null;
  return (
    <Section title={t("notices")} count={unseen.length || undefined} actions={unseen.length > 0 && <button type="button" className="btn btn-sm" onClick={onSeen}>{t("mark_seen")}</button>}>
      <div className="space-y-2">
        {notices.slice(0, 8).map((n) => {
          const shown = t.notice(n);
          return (
          <div key={n.id} className={`banner ${n.severity === "high" ? "banner-danger" : "banner-info"}`}>
            <div className="flex flex-wrap items-center gap-2">
              <Chip className={n.severity === "high" ? "chip-danger" : ""}>{t(`notice_${n.kind}`)}</Chip>
              <b>{shown.title}</b>
              <span className="ml-auto help"><Rel ts={n.ts} /></span>
            </div>
            {shown.body && <div className="help mt-1">{shown.body}</div>}
          </div>
          );
        })}
      </div>
    </Section>
  );
}

export default function Panel() {
  const { t, lang, dash, dashError, changed, notify, confirm } = useApp();
  const [busy, run] = useBusy();

  if (!dash) return <div className="help">{dashError ? "" : t("loading")}</div>;

  const measure = () => run("measure", async () => {
    const r = await api.call("measure_new", {});
    notify(r.started ? t("measure_started", { n: r.models?.length ?? 0 }) : t("measure_nothing"));
    changed();
  });
  const refresh = () => run("refresh", async () => {
    const r = await api.call("models_refresh", {});
    notify(t("refresh_result", { n: r.new?.length ?? 0, m: r.changed?.length ?? 0 }));
    changed();
  });
  const publish = () => run("publish", async () => {
    const diff = dash.routes_diff || {};
    if (!diff.has_changes) { notify(t("routes_nochange")); return; }
    const ok = await confirm({
      title: t("routes_publish"),
      message: t("routes_publish_msg", { added: diff.added.length, changed: diff.changed.length, removed: diff.removed.length }),
      confirmLabel: t("routes_publish"),
      danger: false,
    });
    if (!ok) return;
    await api.call("routes_publish", {});
    notify(t("routes_published"));
    changed();
  });
  const seen = () => run("seen", async () => { await api.visit(); changed(); });

  const att = dash.attention || {};
  const empty = !dash.counts.results;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-2">
        <h1 className="mr-auto">{t("nav_panel")}</h1>
        <Busy className="btn" busy={busy.refresh} onClick={refresh}><Icon d={ICONS.refresh} size={14} />{t("refresh_models")}</Busy>
        <Busy className="btn btn-primary" busy={busy.measure} onClick={measure}><Icon d={ICONS.run} size={14} />{t("measure_new")}</Busy>
        <Busy className="btn" busy={busy.publish} onClick={publish}><Icon d={ICONS.publish} size={14} />{t("routes_publish")}</Busy>
      </div>

      {dash.demo && <div className="banner banner-info">{t("demo_banner")}</div>}

      <RunningCard run={dash.running} />
      {dash.queued?.length > 0 && <div className="help">{t("queued_runs", { n: dash.queued.length })}: {dash.queued.map((r) => r.label).join(" · ")}</div>}

      <Notices notices={dash.notices || []} onSeen={seen} />

      {(att.never_measured?.length > 0 || att.stale?.length > 0 || att.missing?.length > 0) && (
        <Section title={t("attention")}>
          <div className="flex flex-wrap gap-2">
            {att.never_measured?.map((c) => <a key={c.id} href="#/modelos" className="chip chip-amber" style={{ textDecoration: "none" }}>{c.name}: {t("never_measured")}</a>)}
            {att.stale?.map((c) => <a key={c.id} href="#/modelos" className="chip chip-danger" style={{ textDecoration: "none" }}>{c.name}: {t("stale_model")}</a>)}
            {att.missing?.map((c) => <span key={c.id} className="chip">{c.name}: {t("missing_model")}</span>)}
          </div>
        </Section>
      )}

      <Section title={t("routes_table")} actions={<a className="btn-link" href="#/rutas">{t("routes_details")}</a>}>
        {empty ? (
          <Empty>{t("routes_empty")}</Empty>
        ) : (
          <div className="panel scroll-x" style={{ padding: 0 }}>
            <table className="tbl">
              <thead>
                <tr><th>{t("task")}</th><th>{t("model")}</th><th>{t("score")}</th><th className="r">{t("speed")}</th><th className="r">{t("memory")}</th><th>{t("measured")}</th><th>{t("published")}</th></tr>
              </thead>
              <tbody>
                {dash.routes.map((r) => (
                  <tr key={r.category}>
                    <td><b>{t(`task_${r.category}`)}</b></td>
                    {r.winner ? (
                      <>
                        <td style={{ minWidth: 150 }}>{r.winner.names[0]}</td>
                        <td><div style={{ minWidth: 150 }}><Score value={r.winner.score} ci={r.winner.ci} n={r.winner.n} /><CiBar value={r.winner.score} ci={r.winner.ci} /></div></td>
                        <td className="r num">{r.winner.tok_s != null ? `${num(r.winner.tok_s, 0, lang)} tok/s` : "—"}{r.winner.cpu && r.winner.tok_s != null && <> <CpuBadge /></>}</td>
                        <td className="r num">{r.winner.vram_gb != null ? `${num(r.winner.vram_gb, 1, lang)} GB` : "—"}</td>
                        <td><Rel ts={r.measured_ts} /></td>
                        <td>
                          {r.published ? <Chip className={r.published_name === r.winner.names[0] ? "chip-ok" : "chip-amber"}>{r.published_name === r.winner.names[0] ? t("published_same") : t("published_other", { name: r.published_name })}</Chip> : <Chip>{t("not_published")}</Chip>}
                        </td>
                      </>
                    ) : (
                      <td colSpan={6} className="help">{t("no_winner", { n: r.candidates })}</td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Section>

      <Section title="GPU">
        <GpuStrip gpus={dash.gpus} />
      </Section>

      {dash.recent_runs?.length > 0 && (
        <Section title={t("recent_runs")} actions={<a className="btn-link" href="#/ejecutar">{t("all_runs")}</a>}>
          <div className="space-y-1">
            {dash.recent_runs.map((r) => (
              <a key={r.id} href={`#/ejecutar/${r.id}`} className="panel panel-tight flex flex-wrap items-center gap-2" style={{ textDecoration: "none", color: "inherit" }}>
                <StateChip state={r.state} />
                <span className="min-w-0 flex-1" style={{ overflowWrap: "anywhere" }}>{r.label}</span>
                <span className="help num">{r.progress.done}/{r.progress.total}</span>
                <span className="help"><Rel ts={r.finished_ts || r.created_ts} /></span>
              </a>
            ))}
          </div>
        </Section>
      )}
    </div>
  );
}
