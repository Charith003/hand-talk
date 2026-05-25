import { Link, useRouterState } from "@tanstack/react-router";
import { Brain, Info, Mic2, Radio, Upload } from "lucide-react";

const navItems = [
  { to: "/", label: "Live", icon: Radio },
  { to: "/train", label: "Train", icon: Brain },
  { to: "/upload", label: "Upload dataset", icon: Upload },
  { to: "/about", label: "About", icon: Info },
] as const;

export function AppHeader({ title = "SignSpeak", subtitle = "live · train · upload" }) {
  const pathname = useRouterState({ select: (state) => state.location.pathname });

  return (
    <header className="border-b border-border bg-card/80 backdrop-blur">
      <div className="mx-auto flex max-w-7xl flex-col gap-4 px-5 py-4 sm:flex-row sm:items-center sm:justify-between sm:px-8">
        <Link to="/" className="flex items-center gap-3">
          <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary text-primary-foreground shadow-sm">
            <Mic2 className="h-5 w-5" />
          </span>
          <span>
            <span className="block text-base font-semibold">{title}</span>
            <span className="block text-xs text-muted-foreground">{subtitle}</span>
          </span>
        </Link>
        <nav className="flex flex-wrap items-center gap-2 text-sm sm:justify-end">
          {navItems.map((item) => {
            const Icon = item.icon;
            const active = pathname === item.to;
            return (
              <Link
                key={item.to}
                to={item.to}
                className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-2 transition ${
                  active
                    ? "bg-primary font-semibold text-primary-foreground shadow-sm"
                    : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
                }`}
              >
                <Icon className="h-4 w-4" /> {item.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}