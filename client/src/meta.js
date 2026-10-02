// Vocabulary shared with the backend. Labels live in i18n.js.
export const CATEGORIES = ["writing_es", "reasoning", "math", "code", "extraction", "tool_use", "instruction", "long_context", "rag", "vision", "translation", "summary", "custom"];
export const TASKS = ["general", "writing_es", "code", "extraction", "tool_use", "long_context", "rag", "vision", "summary", "translation", "math"];
export const EFFORTS = ["", "off", "low", "medium", "high"];
export const CHECKER_TYPES = ["exact", "contains", "regex", "choice", "number", "math_equiv", "json", "tool_call", "python_tests", "constraints", "needle", "citations", "judge", "family", "none"];
export const VOTES = ["a", "b", "tie", "both_bad"];
export const SHORTHANDS = ["exact", "contains", "any", "number", "choice", "regex", "math", "judge", "none"];

// 24x24 stroke icon paths per page and per run state
export const RUN_STATE = {
  queued: "chip",
  waiting_gpu: "chip-amber",
  waiting_server: "chip-amber",
  running: "chip-accent",
  done: "chip-ok",
  failed: "chip-danger",
  cancelled: "chip",
};
export const ACTIVE_STATES = new Set(["queued", "waiting_gpu", "waiting_server", "running"]);
