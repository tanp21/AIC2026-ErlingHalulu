export function LoadingOverlay({ active, sticky }: { active: boolean; sticky?: boolean }) {
  if (!active) return null;
  return <div className={`loading-overlay${sticky ? " loading-overlay-sticky" : ""}`} role="status" aria-label="Searching">
    <span className="spinner" aria-hidden="true" />
  </div>;
}
