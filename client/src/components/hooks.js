import { useEffect, useRef } from "react";

// Calls `fn` now and then every `ms` while `active` is true. The first call is unconditional: a page that only gets its data
// through polling must never wait for a tab to be visible. Later ticks are skipped while the tab is hidden, and one tick runs
// as soon as it becomes visible again. The latest `fn` is always used.
export function usePoll(fn, ms, active = true) {
  const ref = useRef(fn);
  ref.current = fn;
  useEffect(() => {
    if (!active) return undefined;
    let stopped = false;
    const run = () => { if (!stopped) ref.current(); };
    const tick = () => { if (!document.hidden) run(); };
    const onVisible = () => { if (!document.hidden) run(); };
    run();
    const timer = setInterval(tick, ms);
    document.addEventListener("visibilitychange", onVisible);
    return () => { stopped = true; clearInterval(timer); document.removeEventListener("visibilitychange", onVisible); };
  }, [ms, active]);
}
