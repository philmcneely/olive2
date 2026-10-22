import { createContext, useCallback, useContext, useEffect, useState } from "react";
import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactElement, ReactNode, TextareaHTMLAttributes } from "react";
import { Moon, Sun } from "lucide-react";

/* ------------------------------------------------------------------ */
/* Button                                                              */
/* ------------------------------------------------------------------ */

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger" | "dangerSoft" | "warning" | "tonal";
type ButtonSize = "sm" | "md";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  square?: boolean;
  circle?: boolean;
}

const BUTTON_VARIANTS: Record<ButtonVariant, string> = {
  primary: "btn-primary",
  secondary: "btn-secondary",
  ghost: "btn-ghost",
  danger: "btn-error",
  dangerSoft: "btn-error btn-soft",
  warning: "btn-warning",
  tonal: "btn-primary btn-soft",
};

export function Button({
  variant = "secondary",
  size = "md",
  square = false,
  circle = false,
  className = "",
  type = "button",
  ...rest
}: ButtonProps): ReactElement {
  return (
    <button
      type={type}
      className={`btn ${BUTTON_VARIANTS[variant]} ${size === "sm" ? "btn-sm" : ""} ${square ? "btn-square" : ""} ${circle ? "btn-circle" : ""} min-h-11 ${className}`}
      {...rest}
    />
  );
}

/* ------------------------------------------------------------------ */
/* Form controls                                                       */
/* ------------------------------------------------------------------ */

export function Input({ className = "", ...rest }: InputHTMLAttributes<HTMLInputElement>): ReactElement {
  return <input className={`input w-full ${className}`} {...rest} />;
}

export function Textarea({ className = "", ...rest }: TextareaHTMLAttributes<HTMLTextAreaElement>): ReactElement {
  return <textarea className={`textarea w-full ${className}`} {...rest} />;
}

export function Checkbox({ className = "", ...rest }: InputHTMLAttributes<HTMLInputElement>): ReactElement {
  return <input type="checkbox" className={`checkbox ${className}`} {...rest} />;
}

export function Radio({ className = "", ...rest }: InputHTMLAttributes<HTMLInputElement>): ReactElement {
  return <input type="radio" className={`radio ${className}`} {...rest} />;
}

export function Toggle({ className = "", ...rest }: InputHTMLAttributes<HTMLInputElement>): ReactElement {
  return <input type="checkbox" role="switch" className={`toggle ${className}`} {...rest} />;
}

export function Range({ className = "", ...rest }: InputHTMLAttributes<HTMLInputElement>): ReactElement {
  return <input type="range" className={`range range-sm w-full ${className}`} {...rest} />;
}

export function FileInput({ className = "", ...rest }: InputHTMLAttributes<HTMLInputElement>): ReactElement {
  return <input type="file" className={`file-input w-full ${className}`} {...rest} />;
}

export function Field({
  label,
  hint,
  error,
  htmlFor,
  children,
}: {
  label: ReactNode;
  hint?: ReactNode;
  error?: ReactNode;
  htmlFor?: string;
  children: ReactNode;
}): ReactElement {
  const errorId = htmlFor ? `${htmlFor}-error` : undefined;
  return (
    <fieldset className="fieldset">
      <label htmlFor={htmlFor} className="fieldset-legend">
        {label}
      </label>
      {children}
      {hint && !error && <p className="label">{hint}</p>}
      {error && (
        <p id={errorId} role="alert" className="label text-error">
          {error}
        </p>
      )}
    </fieldset>
  );
}

/* ------------------------------------------------------------------ */
/* Badge                                                               */
/* ------------------------------------------------------------------ */

type BadgeVariant = "neutral" | "success" | "warning" | "error" | "info" | "primary" | "ghost" | "outline";

const BADGE_VARIANTS: Record<BadgeVariant, string> = {
  neutral: "badge-neutral",
  success: "badge-success",
  warning: "badge-warning",
  error: "badge-error",
  info: "badge-info",
  primary: "badge-primary",
  ghost: "badge-ghost",
  outline: "badge-outline",
};

export function Badge({
  variant = "neutral",
  soft = false,
  className = "",
  children,
}: {
  variant?: BadgeVariant;
  soft?: boolean;
  className?: string;
  children: ReactNode;
}): ReactElement {
  return <span className={`badge badge-sm ${BADGE_VARIANTS[variant]} ${soft ? "badge-soft" : ""} ${className}`}>{children}</span>;
}

/* ------------------------------------------------------------------ */
/* Card                                                                */
/* ------------------------------------------------------------------ */

export function Card({ className = "", children }: { className?: string; children: ReactNode }): ReactElement {
  return <div className={`card card-border card-sm bg-base-200 ${className}`}>{children}</div>;
}

export function CardBody({ className = "", children }: { className?: string; children: ReactNode }): ReactElement {
  return <div className={`card-body ${className}`}>{children}</div>;
}

export function CardTitle({ className = "", children }: { className?: string; children: ReactNode }): ReactElement {
  return <h3 className={`card-title ${className}`}>{children}</h3>;
}

export function CardActions({ className = "", children }: { className?: string; children: ReactNode }): ReactElement {
  return <div className={`card-actions ${className}`}>{children}</div>;
}

/* ------------------------------------------------------------------ */
/* Modal (same public API as the legacy inline Modal)                  */
/* ------------------------------------------------------------------ */

export function Modal({
  onClose,
  title,
  ariaLabel,
  maxWidth = "max-w-lg",
  stacked = false,
  children,
}: {
  onClose: () => void;
  title: ReactNode;
  ariaLabel: string;
  maxWidth?: string;
  /** Dialog opened on top of another Modal (e.g. a confirm). daisyUI's .modal is z-index 999, so stacked sits above it. */
  stacked?: boolean;
  children: ReactNode;
}): ReactElement {
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      // A stacked dialog handles Escape first (capture) and keeps the modal beneath it open.
      if (stacked) e.stopImmediatePropagation();
      onClose();
    };
    window.addEventListener("keydown", onKeyDown, stacked);
    return () => window.removeEventListener("keydown", onKeyDown, stacked);
  }, [onClose, stacked]);

  return (
    <div
      className="modal modal-open modal-middle"
      style={stacked ? { zIndex: 1000 } : undefined}
      role="dialog"
      aria-modal="true"
      aria-label={ariaLabel}
    >
      <div className={`modal-box ${maxWidth} max-h-[90dvh] overflow-y-auto`}>
        <div className="mb-4 flex items-center justify-between gap-3">
          <h2 className="min-w-0 flex-1 text-lg font-semibold">{title}</h2>
          <button type="button" onClick={onClose} aria-label="Close" className="btn btn-ghost btn-sm btn-square min-h-11 min-w-11">
            ✕
          </button>
        </div>
        {children}
      </div>
      <div className="modal-backdrop">
        <button type="button" tabIndex={-1} aria-label="Close dialog" onClick={onClose} className="cursor-default" />
      </div>
    </div>
  );
}

export function ConfirmDialog({
  open,
  title,
  message,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  danger = false,
  busy = false,
  onConfirm,
  onCancel,
}: {
  open: boolean;
  title: string;
  message: ReactNode;
  confirmLabel?: string;
  cancelLabel?: string;
  danger?: boolean;
  busy?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}): ReactElement | null {
  if (!open) return null;
  return (
    <Modal onClose={onCancel} title={title} ariaLabel={title} stacked>
      <div className="text-sm text-base-content/80">{message}</div>
      <div className="modal-action">
        <Button variant="ghost" onClick={onCancel} disabled={busy}>
          {cancelLabel}
        </Button>
        <Button variant={danger ? "danger" : "primary"} onClick={onConfirm} disabled={busy}>
          {busy && <span className="loading loading-spinner loading-xs" aria-hidden="true" />}
          {confirmLabel}
        </Button>
      </div>
    </Modal>
  );
}

/* ------------------------------------------------------------------ */
/* Toast                                                               */
/* ------------------------------------------------------------------ */

type ToastTone = "info" | "success" | "warning" | "error";

interface ToastItem {
  id: number;
  tone: ToastTone;
  message: string;
}

const ToastContext = createContext<(tone: ToastTone, message: string) => void>(() => {});

export function useToast(): (tone: ToastTone, message: string) => void {
  return useContext(ToastContext);
}

let toastSeq = 1;

type NotifyFn = (tone: ToastTone, message: string) => void;

let globalNotify: NotifyFn = (tone, message) => {
  pendingToasts.push({ tone, message });
};
const pendingToasts: { tone: ToastTone; message: string }[] = [];

export function notify(tone: ToastTone, message: string): void {
  globalNotify(tone, message);
}

export interface ConfirmOptions {
  title: string;
  message: string;
  confirmLabel?: string;
  danger?: boolean;
}

let confirmSeq = 1;
interface ConfirmRequest extends ConfirmOptions {
  id: number;
  resolve: (value: boolean) => void;
}

let requestConfirm: (req: Omit<ConfirmRequest, "id" | "resolve">) => Promise<boolean> = () =>
  Promise.resolve(window.confirm("Confirm?"));

export function confirmAction(opts: ConfirmOptions): Promise<boolean> {
  return requestConfirm(opts);
}

export function ConfirmHost(): ReactElement {
  const [current, setCurrent] = useState<ConfirmRequest | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    requestConfirm = (opts) =>
      new Promise<boolean>((resolve) => {
        setBusy(false);
        setCurrent({ ...opts, id: confirmSeq++, resolve });
      });
    return () => {
      requestConfirm = () => Promise.resolve(window.confirm("Confirm?"));
    };
  }, []);

  if (!current) return <></>;

  const settle = (value: boolean) => {
    setBusy(true);
    const req = current;
    setCurrent(null);
    setBusy(false);
    req.resolve(value);
  };

  return (
    <ConfirmDialog
      open
      title={current.title}
      message={current.message}
      confirmLabel={current.confirmLabel ?? (current.danger ? "Delete" : "Confirm")}
      danger={current.danger}
      busy={busy}
      onConfirm={() => settle(true)}
      onCancel={() => settle(false)}
    />
  );
}

export function ToastProvider({ children }: { children: ReactNode }): ReactElement {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const push = useCallback((tone: ToastTone, message: string) => {
    const id = toastSeq++;
    setToasts((prev) => [...prev.slice(-3), { id, tone, message }]);
    window.setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 6000);
  }, []);

  useEffect(() => {
    globalNotify = push;
    for (const p of pendingToasts.splice(0)) push(p.tone, p.message);
    return () => {
      globalNotify = (tone, message) => {
        pendingToasts.push({ tone, message });
      };
    };
  }, [push]);

  return (
    <ToastContext.Provider value={push}>
      {children}
      <div className="toast toast-top toast-end z-[100]" aria-live="polite">
        {toasts.map((t) => (
          <div key={t.id} role={t.tone === "error" ? "alert" : "status"} className={`alert alert-soft alert-${t.tone} max-w-sm`}>
            <span>{t.message}</span>
            <button
              type="button"
              aria-label="Dismiss notification"
              className="btn btn-ghost btn-xs btn-square min-h-11 min-w-11"
              onClick={() => setToasts((prev) => prev.filter((x) => x.id !== t.id))}
            >
              ✕
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

/* ConfirmHost joins ToastProvider at the app root so confirmAction() works
   from any handler without prop threading. */
export function ConfirmHostRoot(): ReactElement {
  return <ConfirmHost />;
}

/* ------------------------------------------------------------------ */
/* Theme                                                               */
/* ------------------------------------------------------------------ */

export type OliveTheme = "olive-dark" | "olive-light";

const THEME_KEY = "olive-theme";

export function getInitialTheme(): OliveTheme {
  if (typeof window === "undefined") return "olive-dark";
  const stored = window.localStorage.getItem(THEME_KEY);
  if (stored === "olive-light" || stored === "olive-dark") return stored;
  return "olive-dark";
}

export function applyTheme(theme: OliveTheme): void {
  document.documentElement.setAttribute("data-theme", theme);
  window.localStorage.setItem(THEME_KEY, theme);
  const meta = document.querySelector('meta[name="theme-color"]');
  meta?.setAttribute("content", theme === "olive-dark" ? "#0c0a09" : "#ffffff");
}

export function ThemeToggle(): ReactElement {
  const [theme, setTheme] = useState<OliveTheme>(getInitialTheme);

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  const next = theme === "olive-dark" ? "olive-light" : "olive-dark";
  return (
    <button
      type="button"
      onClick={() => setTheme(next)}
      title={theme === "olive-dark" ? "Switch to light theme" : "Switch to dark theme"}
      aria-label={theme === "olive-dark" ? "Switch to light theme" : "Switch to dark theme"}
      className="btn btn-ghost btn-sm btn-square"
    >
      {theme === "olive-dark" ? <Sun className="h-4 w-4" aria-hidden="true" /> : <Moon className="h-4 w-4" aria-hidden="true" />}
    </button>
  );
}
