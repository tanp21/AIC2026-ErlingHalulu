import { Fragment, useMemo } from "react";

import { highlight } from "./highlight";
import { formatTimestamp } from "./results";
import type { AgentCheckStatus, CheckResult, CheckState } from "./types";

/**
 * The four verdicts, and the words for them. Shared by the Retrieval agent's review workspace
 * and the Search tab's "Check by AI", which read the same checklist from two different models.
 */
export const STATUS_MARK: Record<AgentCheckStatus, string> = {
  supported: "✓", partial: "~", contradicted: "✗", not_mentioned: "·",
};
const STATUS_WORDS: Record<AgentCheckStatus, string> = {
  supported: "supported", partial: "close", contradicted: "contradicted", not_mentioned: "not mentioned",
};
const SOURCE_WORDS: Record<string, string> = {
  vision: "looked at the frames", speech: "speech", ocr: "on screen", caption: "caption",
};
const ANSWER_SOURCE_WORDS: Record<string, string> = {
  frame: "read off the frame", speech: "heard in the speech", ocr: "read on screen",
  unknown: "not answered here",
};

/** The query's words lit up inside a quote, accent- and case-folded like the `vi_folded` analyzer. */
export function Lit({ text, keywords }: { text: string; keywords: string[] }) {
  const { segments } = useMemo(() => highlight(text, keywords), [text, keywords]);
  return <>{segments.map((segment, index) => segment.hit
    ? <mark key={index} className={`hit-${segment.hit}`}>{segment.text}</mark>
    : <Fragment key={index}>{segment.text}</Fragment>)}</>;
}

/** `good` at 6 and up, `bad` at 3 and under -- the same tiers the agent's own verdicts use. */
export function verdictTier(score: number): "good" | "mid" | "bad" {
  return score >= 6 ? "good" : score <= 3 ? "bad" : "mid";
}

/**
 * One check's answer, in whichever surface asked for it.
 *
 * `variant` is only about width: `panel` is the Search tab's right-hand inspector, `inline` is
 * the strip under the video in the player, where it has to stay readable while the video plays.
 */
export function CheckPanel({ state, variant = "panel", keywords = [], onRecheck, onSeek }: {
  state: CheckState;
  variant?: "panel" | "inline";
  /** The query's words, lit up inside the quotes the checker read its verdicts from. */
  keywords?: string[];
  /** Re-read this exact frame instead of reusing the shot's answer. */
  onRecheck?(): void;
  onSeek?(time: number): void;
}) {
  if (state.status === "idle") return null;
  return (
    <section className={`check-panel check-${variant}`} aria-label="AI check">
      {state.status === "loading" && (
        <p className="check-busy" role="status">Looking at the frames around this moment…</p>
      )}
      {state.status === "error" && (
        <p className="check-error" role="alert">{state.error}</p>
      )}
      {state.status === "done" && state.result &&
        <CheckBody result={state.result} keywords={keywords} onRecheck={onRecheck} onSeek={onSeek} />}
    </section>
  );
}

function CheckBody({ result, keywords, onRecheck, onSeek }: {
  result: CheckResult;
  keywords: string[];
  onRecheck?(): void;
  onSeek?(time: number): void;
}) {
  const { support, verdict } = result;
  const tier = verdictTier(verdict.score);
  const width = `${Math.round(support.share * 100)}%`;
  return (
    <>
      <header className="check-header">
        <b className={`check-score check-${tier}`}>{verdict.score.toFixed(0)}/10</b>
        <span className="check-reason">{verdict.reason}</span>
      </header>

      {result.details.length > 0 && <>
        <div className="check-bar" role="img"
          aria-label={`${support.supported} of ${support.total} details supported`}>
          <i style={{ width }} />
        </div>
        <p className="check-tally">
          {support.supported}/{support.total} details
          {support.partial > 0 && <> · {support.partial} close</>}
          {support.contradicted > 0 && <> · {support.contradicted} conflict</>}
        </p>
        <ul className="review-checks check-list">
          {result.details.map((detail, index) => {
            const check = result.checks.find((item) => item.detail === index);
            const status = check?.status ?? "not_mentioned";
            return (
              <li key={index} className={`check-${status}`}>
                <b aria-label={STATUS_WORDS[status]}>{STATUS_MARK[status]}</b> {detail.text}
                {detail.distinctiveness >= 0.7 && (
                  <small className="check-telling" title="Knowing this is true narrows down which moment it is"> telling</small>
                )}
                {check?.note && <em className="review-note">{check.note}</em>}
                {check?.quote && <q>
                  <Lit text={check.quote} keywords={keywords} />
                  <small> {SOURCE_WORDS[check.source ?? ""] ?? check.source}
                    {check.at != null && <> {onSeek
                      ? <button type="button" className="check-seek" onClick={() => onSeek(check.at as number)}>{formatTimestamp(check.at)}</button>
                      : formatTimestamp(check.at)}</>}
                  </small>
                </q>}
                {!check?.quote && check?.source === "vision" && (
                  <small className="check-source">{SOURCE_WORDS.vision}</small>
                )}
              </li>
            );
          })}
        </ul>
      </>}

      {result.answer && <CheckAnswerCard answer={result.answer} keywords={keywords} />}

      <footer className="check-footer">
        <span>
          {result.frames_seen} frame{result.frames_seen === 1 ? "" : "s"} · ±{Math.round(result.window_seconds / 60)} min
          {result.covers && result.shot_index !== null && <> · shot {result.shot_index}, {formatTimestamp(result.covers.start)}–{formatTimestamp(result.covers.end)}</>}
        </span>
        {result.cached && <span className="check-cached" title={`Checked by ${result.by}`}>from the cache</span>}
        {onRecheck && <button type="button" className="check-recheck" onClick={onRecheck}>
          Re-check this exact frame
        </button>}
      </footer>
    </>
  );
}

function CheckAnswerCard({ answer, keywords }: { answer: NonNullable<CheckResult["answer"]>; keywords: string[] }) {
  const found = answer.source !== "unknown" && !!answer.text;
  return (
    <section className={`check-answer${found ? "" : " check-answer-none"}`} aria-label="Answer">
      <h4>Answer</h4>
      {found
        ? <>
          <p className="check-answer-text">{answer.text}</p>
          <small>{ANSWER_SOURCE_WORDS[answer.source] ?? answer.source} · confidence {answer.confidence.toFixed(0)}/10</small>
          {answer.quote && <q><Lit text={answer.quote} keywords={keywords} /></q>}
        </>
        : <p className="check-answer-text"><em>This moment does not answer the question.</em></p>}
    </section>
  );
}
