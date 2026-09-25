import type { KeyboardEvent, ReactNode } from "react";

interface QueryBoxProps {
  label: string;
  ariaLabel: string;
  value: string;
  onChange(value: string): void;
  onKeyDown?(event: KeyboardEvent<HTMLTextAreaElement>): void;
  placeholder?: string;
  rows?: number;
  maxLength?: number;
  autoFocus?: boolean;
  disabled?: boolean;
  footer?: ReactNode;
}

export function QueryBox(props: QueryBoxProps) {
  return (
    <div className="composer-query-box">
      <label>
        <span>{props.label}</span>
        <textarea
          aria-label={props.ariaLabel}
          value={props.value}
          onChange={(event) => props.onChange(event.target.value)}
          onKeyDown={props.onKeyDown}
          placeholder={props.placeholder}
          rows={props.rows ?? 2}
          maxLength={props.maxLength}
          autoFocus={props.autoFocus}
          disabled={props.disabled}
        />
      </label>
      {props.footer != null && <div className="composer-query-footer">{props.footer}</div>}
    </div>
  );
}
