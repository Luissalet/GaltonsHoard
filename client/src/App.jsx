import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, errorText } from "./api.js";
import { initialLang, makeT, saveLang } from "./i18n.js";
import { AppContext } from "./context.js";
import { ConfirmDialog, Icon, ICONS } from "./components/ui.jsx";
import Panel from "./pages/Panel.jsx";
import Modelos from "./pages/Modelos.jsx";
import Pruebas from "./pages/Pruebas.jsx";
import Ejecutar from "./pages/Ejecutar.jsx";
import Clasificacion from "./pages/Clasificacion.jsx";
import Arena from "./pages/Arena.jsx";
import Rutas from "./pages/Rutas.jsx";
import Ajustes from "./pages/Ajustes.jsx";

export { useApp } from "./context.js";

const PAGES = [
  { path: "", key: "nav_panel", icon: ICONS.panel, component: Panel, badge: "notices" },
  { path: "modelos", key: "nav_models", icon: ICONS.models, component: Modelos, badge: "attention" },
  { path: "pruebas", key: "nav_tests", icon: ICONS.tests, component: Pruebas },
  { path: "ejecutar", key: "nav_run", icon: ICONS.run, component: Ejecutar, badge: "active" },
  { path: "clasificacion", key: "nav_board", icon: ICONS.board, component: Clasificacion },
  { path: "arena", key: "nav_arena", icon: ICONS.arena, component: Arena },
  { path: "rutas", key: "nav_routes", icon: ICONS.routes, component: Rutas },
  { path: "ajustes", key: "nav_settings", icon: ICONS.settings, component: Ajustes },
];

function useHashRoute() {
  const read = () => {
    const [path, query = ""] = window.location.hash.replace(/^#\/?/, "").split("?");
    const parts = path.split("/").filter(Boolean).map(decodeURIComponent);
    return { page: parts[0] || "", param: parts[1] || null, query: new URLSearchParams(query) };
  };
  const [route, setRoute] = useState(read);
  useEffect(() => {
    const onChange = () => { setRoute(read()); window.scrollTo(0, 0); };
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return route;
}

export function Toast({ toast, onClose }) {
  useEffect(() => {
    if (!toast) return undefined;
    const timer = setTimeout(onClose, toast.kind === "error" ? 9000 : 4500);
    return () => clearTimeout(timer);
  }, [toast, onClose]);
  if (!toast) return null;
  return (
    <div className={`toast ${toast.kind === "error" ? "toast-error" : "toast-ok"}`} role={toast.kind === "error" ? "alert" : "status"} onClick={onClose}>
      {toast.message}
    </div>
  );
}

function SchedulerLight({ scheduler, t }) {
  if (!scheduler) return null;
  const lanes = Object.values(scheduler.lanes || {});
  const busy = lanes.some((l) => l.current);
  const queue = lanes.reduce((n, l) => n + (l.queue || 0), 0);
  const state = !scheduler.running ? "off" : scheduler.paused ? "paused" : busy ? "busy" : "idle";
  const color = { off: "var(--muted)", paused: "var(--warn)", busy: "var(--accent)", idle: "var(--ok)" }[state];
  return (
    <div className="space-y-0.5" aria-live="polite">
      <div className="flex items-center gap-2 font-semibold text-[12px]" style={{ color: "var(--ink)" }}>
        <span className="dot" style={{ background: color }} />
        {t(`sched_${state}`)}
      </div>
      <div className="help">{t("sched_queue", { n: queue })} · {t("sched_done", { n: scheduler.jobs_done ?? 0 })}</div>
    </div>
  );
}

export default function App() {
  const route = useHashRoute();
  const [lang, setLang] = useState(initialLang);
  const t = useMemo(() => makeT(lang), [lang]);
  const [health, setHealth] = useState(null);
  const [dash, setDash] = useState(null);
  const [dashError, setDashError] = useState(null);
  const [toast, setToast] = useState(null);
  const [confirmReq, setConfirmReq] = useState(null);
  const [version, setVersion] = useState(0);
  const confirmResolve = useRef(null);

  useEffect(() => { document.documentElement.lang = lang; }, [lang]);
  useEffect(() => {
    const n = dash?.counts?.unseen_notices || 0;
    document.title = n ? `(${n}) Galton's Hoard` : "Galton's Hoard";
  }, [dash]);

  const refreshDash = useCallback(async () => {
    try {
      setDash(await api.dashboard());
      setDashError(null);
    } catch (e) {
      setDashError(e);
    }
  }, []);
  // Something changed (a model, a run, a setting): reload the dashboard and every page that listens to `version`.
  const changed = useCallback(() => { setVersion((v) => v + 1); refreshDash(); }, [refreshDash]);

  // Refresh every 60 s (every 4 s while a run is active) when the tab is visible, and when it becomes visible again.
  const running = !!dash?.running;
  useEffect(() => {
    refreshDash();
    api.health().then(setHealth).catch(() => {});
    const timer = setInterval(() => { if (!document.hidden) refreshDash(); }, running ? 4000 : 60000);
    const onVisible = () => { if (!document.hidden) refreshDash(); };
    document.addEventListener("visibilitychange", onVisible);
    return () => { clearInterval(timer); document.removeEventListener("visibilitychange", onVisible); };
  }, [refreshDash, running]);

  const notify = useCallback((message, kind = "ok") => setToast({ message, kind, id: Math.random() }), []);
  const toastError = useCallback((error) => setToast({ message: errorText(error, t), kind: "error", id: Math.random() }), [t]);
  const confirm = useCallback((request) => new Promise((resolve) => {
    confirmResolve.current = resolve;
    setConfirmReq(request);
  }), []);
  const closeConfirm = useCallback((answer) => {
    setConfirmReq(null);
    if (confirmResolve.current) confirmResolve.current(answer);
    confirmResolve.current = null;
  }, []);

  const changeLang = useCallback((next) => {
    saveLang(next);
    setLang(next);
    api.call("settings_set", { values: { "ui.language": next } }).catch(() => {});
  }, []);

  const value = useMemo(() => ({
    t, lang, setLang: changeLang, health, dash, dashError, refreshDash, changed, version, notify, toastError, confirm, route,
  }), [t, lang, changeLang, health, dash, dashError, refreshDash, changed, version, notify, toastError, confirm, route]);

  const page = PAGES.find((p) => p.path === route.page) || PAGES[0];
  const Component = page.component;
  const attention = dash?.attention || {};
  const badges = {
    notices: dash?.counts?.unseen_notices || 0,
    attention: (attention.never_measured?.length || 0) + (attention.stale?.length || 0),
    active: dash?.counts?.runs_active || 0,
  };

  return (
    <AppContext.Provider value={value}>
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:z-50 skip-link focus:p-2">{t("skip")}</a>
      <div className="min-h-dvh md:grid md:grid-cols-[210px_minmax(0,1fr)]">
        <aside className="sticky top-0 z-10 border-b md:flex md:h-dvh md:flex-col md:self-start md:border-b-0 md:border-r" style={{ background: "var(--sidebar)", borderColor: "var(--line)" }}>
          <div className="flex items-center gap-3 px-4 py-3 md:py-4">
            <img src="/icon-192.png" alt="" width="30" height="30" className="rounded-lg" />
            <div>Galton's Hoard<small>{t("subtitle")}</small></div>
          </div>
          <nav aria-label={t("sections")} className="flex gap-1 overflow-x-auto px-3 pb-2 md:flex-col">
            {PAGES.map((p) => {
              const n = p.badge ? badges[p.badge] : 0;
              return (
                <a key={p.path} href={`#/${p.path}`} className="nav-link shrink-0 text-[13px]" aria-current={p.path === page.path ? "page" : undefined}>
                  <Icon d={p.icon} />
                  {t(p.key)}
                  {n > 0 && <span className="nav-badge" aria-label={t("badge_n", { n })}>{n > 99 ? "99+" : n}</span>}
                </a>
              );
            })}
          </nav>
          <div className="hidden flex-1 md:block" />
          <div className="hidden space-y-3 border-t px-4 py-3 md:block" style={{ borderColor: "var(--line)" }}>
            <SchedulerLight scheduler={dash?.scheduler} t={t} />
            {(health?.demo || dash?.demo) && <span className="chip chip-amber">{t("demo_on")}</span>}
            {health?.offline && <span className="chip chip-amber">{t("offline_on")}</span>}
            <div><button type="button" className="btn btn-sm" onClick={() => changeLang(lang === "es" ? "en" : "es")}>{t("language")}</button></div>
          </div>
        </aside>
        <main id="main" className="min-w-0 px-4 py-4 md:px-7 md:py-6">
          {dashError && !dash && (
            <div className="banner banner-danger mb-4" role="alert">
              {t("unreachable")}: {dashError.message}. <button type="button" className="btn-link" onClick={refreshDash}>{t("retry")}</button>
            </div>
          )}
          {dashError && dash && (
            <div className="banner banner-warn mb-4" role="status">
              {t("stale")}: {dashError.message}. <button type="button" className="btn-link" onClick={refreshDash}>{t("retry")}</button>
            </div>
          )}
          <Component key={`${route.page}/${route.param || ""}`} param={route.param} query={route.query} />
          <div className="mt-8 flex flex-wrap items-center gap-3 border-t pt-3 md:hidden" style={{ borderColor: "var(--line)" }}>
            <SchedulerLight scheduler={dash?.scheduler} t={t} />
            {(health?.demo || dash?.demo) && <span className="chip chip-amber">{t("demo_on")}</span>}
            <button type="button" className="btn btn-sm" onClick={() => changeLang(lang === "es" ? "en" : "es")}>{t("language")}</button>
          </div>
        </main>
      </div>
      <Toast toast={toast} onClose={() => setToast(null)} />
      <ConfirmDialog request={confirmReq} onClose={closeConfirm} />
    </AppContext.Provider>
  );
}
