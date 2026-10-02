import React, { useCallback, useEffect, useState } from "react";
import { api } from "../api.js";
import { useApp } from "../context.js";
import { Busy, Chip, Empty, ErrorBox, Field, Section, Spinner, useBusy, useLoad } from "../components/ui.jsx";
import { CATEGORIES, VOTES } from "../meta.js";
import { num } from "../format.js";

function RatingsTable({ rows, minVotes }) {
  const { t, lang } = useApp();
  if (!rows?.length) return <div className="help">{t("ratings_none")}</div>;
  return (
    <div className="panel scroll-x" style={{ padding: 0 }}>
      <table className="tbl">
        <thead><tr><th>{t("model")}</th><th className="r">{t("rating")}</th><th className="r">95 %</th><th className="r">{t("votes")}</th></tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.contestant}>
              <td style={{ overflowWrap: "anywhere" }}>{r.name}</td>
              <td className="r num">{r.shown && r.rating != null ? <b>{num(r.rating, 0, lang)}</b> : <span className="help">{t("rating_hidden", { n: minVotes })}</span>}</td>
              <td className="r num">{r.shown && r.ci ? `${num(r.ci[0], 0, lang)}–${num(r.ci[1], 0, lang)}` : "—"}</td>
              <td className="r num">{r.votes}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Arena() {
  const { t, version, changed } = useApp();
  const [category, setCategory] = useState("");
  const [pair, setPair] = useState(null);
  const [none, setNone] = useState(false);
  const [revealed, setRevealed] = useState(null);
  const [error, setError] = useState(null);
  const [busy, run] = useBusy();
  const ratings = useLoad(() => api.call("arena_ratings", {}), [version]);

  const next = useCallback(async () => {
    setRevealed(null);
    setError(null);
    try {
      const r = await api.call("arena_next", category ? { category } : {});
      setPair(r.pair ? r : null);
      setNone(!r.pair);
    } catch (e) {
      setError(e);
    }
  }, [category]);
  useEffect(() => { next(); }, [next]);

  const vote = (v) => run("vote", async () => {
    const r = await api.call("arena_vote", { pair: pair.pair, vote: v });
    setRevealed(r);
    ratings.reload();
    changed();
  });

  const labels = { a: t("vote_a"), b: t("vote_b"), tie: t("vote_tie"), both_bad: t("vote_bad") };
  const minVotes = ratings.data?.min_votes ?? 5;
  const cats = Object.keys(ratings.data?.by_category || {});

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end gap-2">
        <h1 className="mr-auto">{t("nav_arena")}</h1>
        <div style={{ width: 220 }}>
          <Field label={t("category")}>
            <select className="field" value={category} onChange={(e) => setCategory(e.target.value)}>
              <option value="">{t("all")}</option>
              {CATEGORIES.map((c) => <option key={c} value={c}>{t(`cat_${c}`)}</option>)}
            </select>
          </Field>
        </div>
      </div>
      <p className="help">{t("arena_help")}</p>
      <ErrorBox error={error} />
      {none && !error && <Empty>{t("arena_none")}</Empty>}
      {pair && (
        <div className="space-y-3">
          <div className="panel space-y-1">
            <div className="flex flex-wrap items-center gap-2"><b>{pair.title}</b><Chip className="chip-accent">{t(`cat_${pair.category}`)}</Chip></div>
            <div className="pre" style={{ maxHeight: 220 }}>{pair.prompt}</div>
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            {["a", "b"].map((side) => (
              <div key={side} className="panel space-y-2">
                <div className="flex items-center gap-2"><b>{t(`answer_${side}`)}</b>{revealed && <Chip className="chip-accent">{revealed[side]}</Chip>}</div>
                <div className="pre" style={{ maxHeight: 360 }}>{pair[side].text}</div>
              </div>
            ))}
          </div>
          {!revealed ? (
            <div className="flex flex-wrap gap-2" role="group" aria-label={t("vote")}>
              {VOTES.map((v) => <Busy key={v} className={`btn ${v === "a" || v === "b" ? "btn-primary" : ""}`} busy={busy.vote} onClick={() => vote(v)}>{labels[v]}</Busy>)}
            </div>
          ) : (
            <div className="flex flex-wrap items-center gap-3">
              <span className="help">{t("voted", { vote: labels[revealed.vote] })}</span>
              <button type="button" className="btn btn-primary" onClick={next}>{t("next_pair")}</button>
            </div>
          )}
        </div>
      )}
      <Section title={t("ratings")} count={ratings.data?.votes}>
        <ErrorBox error={ratings.error} />
        {ratings.loading && !ratings.data && <Spinner />}
        {ratings.data && (
          <div className="space-y-3">
            <div><h3 className="mb-1">{t("overall")}</h3><RatingsTable rows={ratings.data.overall} minVotes={minVotes} /></div>
            {cats.map((c) => <div key={c}><h3 className="mb-1">{t(`cat_${c}`)}</h3><RatingsTable rows={ratings.data.by_category[c]} minVotes={minVotes} /></div>)}
          </div>
        )}
      </Section>
    </div>
  );
}
