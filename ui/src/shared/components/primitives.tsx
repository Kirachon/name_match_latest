import * as React from "react";
import { cx } from "@/shared/lib/format";

/* -------------------- Button -------------------- */

export type ButtonTone = "primary" | "secondary" | "ghost" | "danger";
export type ButtonSize = "sm" | "md";

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  tone?: ButtonTone;
  size?: ButtonSize;
  loading?: boolean;
  leadingIcon?: React.ReactNode;
  trailingIcon?: React.ReactNode;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  function Button(
    {
      tone = "secondary",
      size = "md",
      loading,
      leadingIcon,
      trailingIcon,
      className,
      children,
      disabled,
      ...rest
    },
    ref,
  ) {
    return (
      <button
        ref={ref}
        type={rest.type ?? "button"}
        className={cx(
          "btn",
          size === "md" ? "btn-md" : "btn-sm",
          tone === "primary" && "btn-primary",
          tone === "secondary" && "btn-secondary",
          tone === "ghost" && "btn-ghost",
          tone === "danger" && "btn-danger",
          className,
        )}
        disabled={disabled || loading}
        {...rest}
      >
        {loading ? <Spinner /> : leadingIcon}
        {children}
        {!loading && trailingIcon}
      </button>
    );
  },
);

function Spinner() {
  return (
    <svg
      className="animate-spin"
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden
    >
      <circle
        cx="12"
        cy="12"
        r="9"
        stroke="currentColor"
        strokeWidth="3"
        opacity="0.25"
      />
      <path
        d="M21 12a9 9 0 0 0-9-9"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
      />
    </svg>
  );
}

/* -------------------- Field (label + input + help/error) -------------------- */

export interface FieldProps {
  label?: React.ReactNode;
  htmlFor?: string;
  help?: React.ReactNode;
  error?: React.ReactNode;
  required?: boolean;
  className?: string;
  children: React.ReactNode;
}

export function Field({
  label,
  htmlFor,
  help,
  error,
  required,
  className,
  children,
}: FieldProps) {
  return (
    <div className={cx("flex flex-col", className)}>
      {label && (
        <label htmlFor={htmlFor} className="label">
          {label}
          {required && <span className="text-danger-400 ml-0.5">*</span>}
        </label>
      )}
      {children}
      {error ? (
        <span role="alert" className="help-error">
          {error}
        </span>
      ) : help ? (
        <span className="help">{help}</span>
      ) : null}
    </div>
  );
}

/* -------------------- Card / Section header -------------------- */

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  padded?: boolean;
}
export function Card({ padded = true, className, ...rest }: CardProps) {
  return (
    <div className={cx("surface", padded && "p-4", className)} {...rest} />
  );
}

export interface SectionHeaderProps {
  title: string;
  description?: React.ReactNode;
  action?: React.ReactNode;
  className?: string;
}
export function SectionHeader({
  title,
  description,
  action,
  className,
}: SectionHeaderProps) {
  return (
    <div className={cx("flex items-start justify-between gap-3 mb-3", className)}>
      <div>
        <h2 className="section-title">{title}</h2>
        {description && (
          <p className="text-sm text-ink-200 mt-1.5">{description}</p>
        )}
      </div>
      {action}
    </div>
  );
}

/* -------------------- Toggle (accessible switch) -------------------- */

export interface ToggleProps {
  checked: boolean;
  onChange: (next: boolean) => void;
  label?: React.ReactNode;
  description?: React.ReactNode;
  disabled?: boolean;
  reason?: string;
}

export function Toggle({
  checked,
  onChange,
  label,
  description,
  disabled,
  reason,
}: ToggleProps) {
  return (
    <label
      className={cx(
        "flex items-start gap-3 group",
        disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer",
      )}
      title={disabled ? reason : undefined}
    >
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        data-on={checked ? "true" : "false"}
        className="toggle shrink-0 mt-0.5"
        onClick={() => onChange(!checked)}
      />
      {(label || description) && (
        <span className="flex flex-col">
          {label && <span className="text-sm text-ink-100">{label}</span>}
          {description && (
            <span className="text-xs text-ink-400">{description}</span>
          )}
          {reason && disabled && (
            <span className="text-xs text-warn-400">{reason}</span>
          )}
        </span>
      )}
    </label>
  );
}

/* -------------------- Pill / Status indicators -------------------- */

export type PillTone = "info" | "ok" | "warn" | "danger" | "mute";

export function Pill({
  tone = "mute",
  children,
  className,
}: {
  tone?: PillTone;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <span
      className={cx(
        "pill",
        tone === "ok" && "pill-ok",
        tone === "warn" && "pill-warn",
        tone === "danger" && "pill-danger",
        tone === "info" && "pill-info",
        tone === "mute" && "pill-mute",
        className,
      )}
    >
      {children}
    </span>
  );
}

export function StatusDot({
  tone = "mute",
  pulse,
  className,
}: {
  tone?: "ok" | "warn" | "danger" | "info" | "mute";
  pulse?: boolean;
  className?: string;
}) {
  return (
    <span
      className={cx(
        "inline-block h-2 w-2 rounded-full",
        tone === "ok" && "bg-ok-400",
        tone === "warn" && "bg-warn-400",
        tone === "danger" && "bg-danger-400",
        tone === "info" && "bg-accent-400",
        tone === "mute" && "bg-ink-500",
        pulse && "animate-pulse-soft",
        className,
      )}
    />
  );
}
