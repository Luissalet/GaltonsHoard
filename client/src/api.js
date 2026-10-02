// Thin fetch wrapper: JSON in/out. `{ error, code, hint }` bodies become exceptions that keep code and hint.
async function request(method, path, { params, body } = {}) {
  const url = new URL(path, window.location.origin);
  for (const [key, value] of Object.entries(params || {})) {
    if (value !== undefined && value !== null && value !== "") url.searchParams.set(key, value);
  }
  let response;
  try {
    response = await fetch(url, {
      method,
      headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
  } catch (cause) {
    const error = new Error(cause && cause.message ? cause.message : "Network error");
    error.code = "network";
    throw error;
  }
  const text = await response.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { error: text.slice(0, 300) };
  }
  if (!response.ok) {
    const detail = data && (data.error || data.detail);
    const error = new Error((typeof detail === "string" && detail) || `Error ${response.status}`);
    error.code = data && data.code;
    error.hint = data && data.hint;
    error.key = (data && data.key) || "";          // the catalogue key and parameters of the backend (galton_hoard/messages.py); the UI formats them
    error.params = (data && data.params) || {};
    error.problemItems = (data && data.problem_items) || [];
    error.status = response.status;
    throw error;
  }
  return data;
}

// Every write and most reads go through the same tool handlers the assistant uses.
const call = (name, args) => request("POST", "/api/ui/call", { body: { name, arguments: args || {} } });

export const api = {
  health: () => request("GET", "/api/health"),
  dashboard: () => request("GET", "/api/dashboard"),
  visit: () => request("POST", "/api/dashboard/visit", { body: {} }),
  events: (run, after = 0) => request("GET", `/api/runs/${encodeURIComponent(run)}/events`, { params: { after } }),
  visionUrl: (caseId, n = 1) => `/api/vision/${encodeURIComponent(caseId)}/${n}.png`,
  call,
};

// Message for a toast: the backend's error plus its hint.
// `t` is the translator of the UI (i18n.js); without it the English text of the server is used.
export function errorText(error, t) {
  if (!error) return "";
  const base = (t ? t.error(error) : "") || error.message || String(error);
  const hint = t ? t.hint(error) : error.hint;
  return hint ? `${base} — ${hint}` : base;
}
