import React, { useEffect, useMemo, useRef, useState } from "react";
import { useApp } from "../context.js";
import { clock, rel } from "../format.js";

export function Icon({ d, size = 17, color }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color || "currentColor"} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={d} />
    </svg>
  );
}

export const ICONS = {
  panel: "M4 13h6V4H4zM14 20h6V4h-6zM4 20h6v-3H4z",
  models: "M5 4h14v6H5zM5 14h14v6H5zM8 7h.01M8 17h.01",
  tests: "M9 3h6M10 3v6L5 19a1 1 0 001 1h12a1 1 0 001-1l-5-10V3M8 15h8",
  run: "M7 4l13 8-13 8z",
  board: "M4 20V10M10 20V4M16 20v-8M22 20H2",
  arena: "M4 20l6-6M14 10l6-6M4 4l16 16M3 21l2-5 3 3zM21 21l-2-5-3 3z",
  routes: "M5 6a2 2 0 100-4 2 2 0 000 4zM19 22a2 2 0 100-4 2 2 0 000 4zM5 6v5a4 4 0 004 4h6a4 4 0 014 4",
  settings: "M12 15a3 3 0 100-6 3 3 0 000 6zM19 12l2-1-1-3-2 .3-1.4-1.4.3-2-3-1-1 2h-2l-1-2-3 1 .3 2L6.8 7.3 5 7 4 10l2 1v2l-2 1 1 3 2-.3 1.4 1.4-.3 2 3 1 1-2h2l1 2 3-1-.3-2 1.4-1.4 2 .3 1-3-2-1z",
  plus: "M12 5v14M5 12h14",
  refresh: "M3 12a9 9 0 0115-6.7L21 8M21 3v5h-5M21 12a9 9 0 01-15 6.7L3 16M3 21v-5h5",
  copy: "M9 9h11v11H9zM5 15V4h11",
  pencil: "M4 20h4L19 9l-4-4L4 16zM13 7l4 4",
  trash: "M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13",
  check: "M5 12l5 5 9-10",
  close: "M6 6l12 12M18 6L6 18",
  chevron: "M9 6l6 6-6 6",
  back: "M15 6l-6 6 6 6",
  upload: "M12 16V4M7 9l5-5 5 5M4 16v4h16v-4",
  search: "M11 4a7 7 0 100 14 7 7 0 000-14zM21 21l-5-5",
  publish: "M12 16V4M7 9l5-5 5 5M4 16v4h16v-4",
  stop: "M6 6h12v12H6z",
  chip: "M7 7h10v10H7zM10 3v4M14 3v4M10 17v4M14 17v4M3 10h4M3 14h4M17 10h4M17 14h4",
  warn: "M12 3l10 18H2zM12 10v5M12 18h.01",
  eye: "M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12zM12 15a3 3 0 100-6 3 3 0 000 6z",
};

export function Spinner() {
  return <span className="spinner" role="status" aria-label="…" />;
}

export function Empty({ children }) {
  return <div className="panel help text-center">{children}</div>;
}

export function Field({ label, hint, children, className = "" }) {
  return (
    <label className={`block ${className}`}>
      <span className="label">{label}</span>
      {children}
      {hint && <span className="help mt-1 block">{hint}</span>}
    </label>
  );
}

export function Switch({ checked, onChange, disabled, label }) {
  return (
    <label className="switch" title={label}>
      <input type="checkbox" role="switch" checked={!!checked} disabled={disabled} aria-label={label} onChange={(e) => onChange(e.target.checked)} />
      <span />
    </label>
  );
}

export function Chip({ children, className = "", title }) {
  return <span className={`chip ${className}`} title={title}>{children}</span>;
}

export function Section({ title, count, actions, children, id }) {
  return (
    <section className="space-y-2" aria-labelledby={id}>
      <div className="flex flex-wrap items-center gap-2">
        <h2 id={id}>{title}</h2>
        {count !== undefined && count !== null && <span className="chip">{count}</span>}
        <div className="ml-auto flex flex-wrap items-center gap-2">{actions}</div>
      </div>
      {children}
    </section>
  );
}

export function Busy({ busy, children, ...props }) {
  return (
    <button type="button" {...props} disabled={busy || props.disabled}>
      {busy && <span className="spinner" aria-hidden="true" />}
      {children}
    </button>
  );
}

export function ErrorBox({ error }) {
  const { t } = useApp();
  if (!error) return null;
  return (
    <div className="banner banner-danger" role="alert">
      {t.error(error)}
      {t.hint(error) ? <span className="help block">{t("hint")}: {t.hint(error)}</span> : null}
    </div>
  );
}

// A stored failure of the backend (a run, a model in a run or one answer): the message in the UI language and, when it has one, what to do about it.
export function Failure({ m, className = "banner banner-danger" }) {
  const { t } = useApp();
  if (!m) return null;
  const hint = t.hint(m);
  return (
    <div className={className} role="alert">
      {t.msg(m)}
      {hint ? <span className="help block">{t("hint")}: {hint}</span> : null}
    </div>
  );
}

// A problem found before a run starts (`run_plan`): which model, what is wrong and what to do.
export function Problem({ p }) {
  const { t } = useApp();
  if (!p) return null;
  const hint = t.hint(p.problem);
  return (
    <div className="banner banner-warn">
      <b>{p.name}</b>: {t.msg(p.problem)}
      {hint ? <span className="help block">{t("hint")}: {hint}</span> : null}
    </div>
  );
}

export function Rel({ ts }) {
  const { lang } = useApp();
  if (!ts) return <span className="help">—</span>;
  return <time dateTime={new Date(ts * 1000).toISOString()} title={clock(ts, lang)}>{rel(ts, lang)}</time>;
}

// ------------------------------------------------------------------ dialogs
export function Modal({ title, onClose, children, wide }) {
  const ref = useRef(null);
  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    const first = ref.current?.querySelector("input, select, textarea, button");
    first?.focus();
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="modal-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="modal space-y-3" ref={ref} role="dialog" aria-modal="true" aria-label={title} style={wide ? { width: "min(560px, 100%)" } : undefined} onMouseDown={(e) => e.stopPropagation()}>
        <h2>{title}</h2>
        {children}
      </div>
    </div>
  );
}

export function Drawer({ title, onClose, children }) {
  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <>
      <div className="modal-backdrop" style={{ zIndex: 64 }} onMouseDown={onClose} aria-hidden="true" />
      <div className="drawer space-y-3" role="dialog" aria-modal="true" aria-label={title} style={{ background: "var(--hoard-surface)", borderColor: "var(--hoard-border)" }}>
        <div className="flex items-start gap-2">
          <h2 className="min-w-0 flex-1" style={{ overflowWrap: "anywhere" }}>{title}</h2>
          <button type="button" className="btn btn-sm" onClick={onClose} aria-label="×"><Icon d={ICONS.close} size={14} /></button>
        </div>
        {children}
      </div>
    </>
  );
}

export function Tabs({ tabs, value, onChange }) {
  return (
    <div role="tablist" className="flex flex-wrap gap-1 border-b" style={{ borderColor: "var(--line)" }}>
      {tabs.map((tab) => (
        <button key={tab.id} type="button" role="tab" className="tab" aria-selected={tab.id === value} onClick={() => onChange(tab.id)}>{tab.label}</button>
      ))}
    </div>
  );
}

export function ConfirmDialog({ request, onClose }) {
  const { t } = useApp();
  const cancelRef = useRef(null);
  useEffect(() => {
    if (!request) return undefined;
    cancelRef.current?.focus();
    const onKey = (e) => { if (e.key === "Escape") onClose(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [request, onClose]);
  if (!request) return null;
  return (
    <div className="modal-backdrop" onClick={() => onClose(false)}>
      <div className="modal space-y-3" role="alertdialog" aria-modal="true" aria-labelledby="confirm-title" onClick={(e) => e.stopPropagation()}>
        <h2 id="confirm-title">{request.title || t("confirm")}</h2>
        <p>{request.message}</p>
        <div className="flex justify-end gap-2">
          <button type="button" ref={cancelRef} className="btn" onClick={() => onClose(false)}>{t("cancel")}</button>
          <button type="button" className={`btn ${request.danger === false ? "btn-primary" : "btn-danger"}`} onClick={() => onClose(true)}>{request.confirmLabel || t("delete")}</button>
        </div>
      </div>
    </div>
  );
}

// Loads something on mount and when `deps` change. `reload()` runs it again without flashing the placeholder.
export function useLoad(fn, deps) {
  const [state, setState] = useState({ data: null, error: null, loading: true });
  const run = useRef(fn);
  run.current = fn;
  const seq = useRef(0);
  const load = useMemo(() => async () => {
    const mine = ++seq.current;
    try {
      const data = await run.current();
      if (mine === seq.current) setState({ data, error: null, loading: false });
    } catch (error) {
      if (mine === seq.current) setState((s) => ({ data: s.data, error, loading: false }));
    }
  }, []);
  useEffect(() => {
    setState((s) => ({ ...s, loading: true }));
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return { ...state, reload: load };
}

// Busy flags keyed by name; a failing action shows the backend's error and hint in a toast.
export function useBusy() {
  const { toastError } = useApp();
  const [busy, setBusy] = useState({});
  const mounted = useRef(true);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  const run = async (key, fn) => {
    setBusy((b) => ({ ...b, [key]: true }));
    try {
      return await fn();
    } catch (error) {
      toastError(error);
      return undefined;
    } finally {
      if (mounted.current) setBusy((b) => ({ ...b, [key]: false }));
    }
  };
  return [busy, run];
}
