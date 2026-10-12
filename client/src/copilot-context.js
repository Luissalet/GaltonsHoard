// What the shared assistant panel asks of this app: the objects open in the current view (real ids, read through the same
// tool endpoint the UI uses), the unsaved fields of open forms, and a refresh when the assistant finishes a turn.
// The pure helpers at the top are covered by client/copilot-context.test.mjs.

const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

// Keys of `draft` whose value differs from what was loaded.
export function changedFields(initial, draft) {
  return Object.fromEntries(Object.entries(draft || {}).filter(([key, value]) => !same(value, (initial || {})[key])));
}

// A response acknowledges what was submitted, not edits made while waiting.
export function acknowledgeDraft(draft, sent) {
  const out = { ...draft };
  for (const [key, value] of Object.entries(sent || {})) {
    const current = out[key];
    if (same(current, value)) {
      delete out[key];
    } else if (value && current && typeof value === 'object' && typeof current === 'object' && !Array.isArray(value) && !Array.isArray(current)) {
      const remaining = Object.fromEntries(Object.entries(current).filter(([field, pending]) => !(field in value) || !same(pending, value[field])));
      if (Object.keys(remaining).length) out[key] = remaining;
      else delete out[key];
    }
  }
  return out;
}

// A form was loaded from `oldBase`, the user edited it into `form`, and the object has been reloaded as `newBase`
// (the assistant changed it meanwhile): keep what the user touched, take the new value of everything else.
// The base keeps the OLD value of the touched fields, so a later save still sees them as changed and a list merge knows what the user removed.
export function rebase(oldBase, form, newBase) {
  const touched = changedFields(oldBase, form);
  const keep = Object.fromEntries(Object.keys(touched).map((k) => [k, (oldBase || {})[k]]));
  return { form: { ...newBase, ...touched }, base: { ...newBase, ...keep } };
}

// Same idea for a list edited by hand (aliases, tags): remove what the user removed, add what the user added, keep the rest of `current`.
export function mergeList(initial, draft, current) {
  const removed = (initial || []).filter((x) => !(draft || []).includes(x));
  const added = (draft || []).filter((x) => !(initial || []).includes(x));
  const kept = (current || []).filter((x) => !removed.includes(x));
  return [...kept, ...added.filter((x) => !kept.includes(x))];
}

// `#/ejecutar/r_123?x=1` -> { page: "ejecutar", param: "r_123" }
export function parseHash(hash) {
  const [path] = String(hash || "").replace(/^#\/?/, "").split("?");
  const parts = path.split("/").filter(Boolean).map((p) => { try { return decodeURIComponent(p); } catch { return p; } });
  return { page: parts[0] || "", param: parts[1] || null };
}

const PAGE_NAMES = { "": "dashboard", modelos: "models", pruebas: "tests", ejecutar: "run", clasificacion: "leaderboard", arena: "arena", rutas: "routes", ajustes: "settings" };

// ---------------------------------------------------------------- objects open inside a page (drawers and modals keep their id in local state)
const openObjects = new Map();
let openSeq = 0;
export function registerOpen(kind, id) {
  if (!id) return () => {};
  const token = ++openSeq;
  openObjects.set(token, { kind, id });
  return () => { openObjects.delete(token); };
}
export const openList = () => [...openObjects.values()];

// ---------------------------------------------------------------- unsaved form fields
// Every open form that matches answers with `detail.provide({ kind, dirty, fields })`. `objectId` is null for creation forms.
export function editorContext(objectId = null) {
  const snapshots = [];
  if (typeof window !== "undefined") {
    window.dispatchEvent(new CustomEvent("galton:context-request", { detail: { objectId, provide: (value) => { if (value) snapshots.push(value); } } }));
  }
  return snapshots;
}

// A form answers when the request is for its object (editing) or for nothing in particular (creation forms).
export function answers(event, objectId) {
  const wanted = event?.detail?.objectId ?? null;
  return objectId == null ? wanted == null : wanted === objectId;
}

// ---------------------------------------------------------------- object cards
const dropEmpty = (o) => Object.fromEntries(Object.entries(o).filter(([, v]) => v !== undefined && v !== null && v !== ""));

export const CARDS = {
  model: {
    tool: (id) => ["model_get", { model: id }],
    card: (r, id) => r?.model && dropEmpty({ kind: "model", object_id: r.model.id || id, title: r.model.name, enabled: r.model.enabled, model_kind: r.model.kind, family: r.model.family,
      quant: r.model.quant, vision: r.model.vision, aliases: r.model.aliases, measured_n: r.model.measured_n }),
  },
  suite: {
    tool: (id) => ["suite_get", { suite: id, include_cases: false }],
    card: (r, id) => r?.suite && dropEmpty({ kind: "suite", object_id: r.suite.id || id, title: r.suite.name, category: r.suite.category, builtin: r.suite.builtin,
      description: r.suite.description, max_tokens: r.suite.max_tokens, total_cases: r.total_cases }),
  },
  case: {
    tool: (id) => ["case_get", { case: id }],
    card: (r, id) => r?.case && dropEmpty({ kind: "case", object_id: r.case.id || id, title: r.case.title, suite_id: r.case.suite_id, checker: r.case.checker_label,
      weight: r.case.weight, tags: r.case.tags, results: r.case.results }),
  },
  run: {
    tool: (id) => ["run_status", { run: id }],
    card: (r, id) => r?.run && dropEmpty({ kind: "run", object_id: r.run.id || id, title: `Run ${String(r.run.id || id).slice(0, 10)}${r.run.state ? ` (${r.run.state})` : ""}`, state: r.run.state,
      suites: (r.run.suites || []).map((s) => s.name || s.id), models: (r.run.contestants || []).map((c) => c.name || c.id) }),
  },
};

async function fetchObject(call, kind, id) {
  const spec = CARDS[kind];
  if (!spec || !id) return null;
  try {
    const [name, args] = spec.tool(id);
    return spec.card(await call(name, args), id) || null;
  } catch {
    return null;                       // an id that does not exist (any more) is simply not an active object
  }
}

// Route objects (a suite or run in the URL) plus the drawers open on top of them; unique, in that order.
export function candidates(route, open) {
  const out = [];
  if (route.page === "pruebas" && route.param) out.push({ kind: "suite", id: route.param });
  if (route.page === "ejecutar" && route.param) out.push({ kind: "run", id: route.param });
  out.push(...open);
  const seen = new Set();
  return out.filter((c) => { const k = `${c.kind}:${c.id}`; if (seen.has(k)) return false; seen.add(k); return true; });
}

// The whole context. `call(name, args)`, `hash`, `open` and `editors(objectId)` are injected so the function is testable.
export async function buildContext({ hash, call, open = [], editors = () => [] }) {
  const route = parseHash(hash);
  const requested = candidates(route, open);
  const pending = new Map(requested.map((c) => [c.id, editors(c.id)]));
  const creation = editors(null);
  const found = await Promise.all(requested.map((c) => fetchObject(call, c.kind, c.id)));
  const active = found.filter(Boolean);
  const view = { page: PAGE_NAMES[route.page] || route.page || "dashboard" };
  const focus = active[active.length - 1];
  if (focus) view.object_id = focus.object_id;
  const snaps = [...(focus ? pending.get(focus.object_id) || [] : []), ...creation];
  const unique = snaps.filter((s, i) => snaps.findIndex((o) => o.kind === s.kind) === i);
  if (unique.length) {
    view.editor = unique[0];
    if (unique.length > 1) view.editors = unique;
  }
  return { active_objects: active, view };
}

async function uiCall(name, args) {
  const response = await fetch("/api/ui/call", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name, arguments: args || {} }) });
  if (!response.ok) throw new Error(`Error ${response.status}`);
  return response.json();
}

export const context = () => buildContext({ hash: window.location.hash, call: uiCall, open: openList(), editors: editorContext });

// ---------------------------------------------------------------- turn end
let refresher = null;
let timer = null;
export function setRefresher(fn) { refresher = fn; return () => { if (refresher === fn) refresher = null; }; }
export function scheduleRefresh() {                 // the event and onTurnEnd may both fire: one refresh
  clearTimeout(timer);
  timer = setTimeout(() => { try { refresher?.(); } catch { /* the next tick of the app refreshes anyway */ } }, 60);
}

export function installCopilot() {
  window.hoardCopilot = { context, onTurnEnd: scheduleRefresh };
  window.addEventListener("hoard:copilot-turn-end", scheduleRefresh);
}
