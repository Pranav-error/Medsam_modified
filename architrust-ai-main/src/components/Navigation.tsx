import { Link, useLocation } from "react-router-dom";
import { useState, useEffect } from "react";
import { Activity } from "lucide-react";
import { cn } from "@/lib/utils";

const navItems = [
  { name: "Home", path: "/" },
  { name: "Hospitals", path: "/hospitals" },
  { name: "Training", path: "/training" },
  { name: "FLED Score", path: "/fled" },
  { name: "Explainability", path: "/xai" },
  { name: "Reports", path: "/reports" },
];

export const Navigation = () => {
  const location = useLocation();
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <nav
      className={cn(
        "fixed top-0 left-0 right-0 z-50 transition-all duration-300",
        scrolled
          ? "bg-background/80 backdrop-blur-xl shadow-md border-b border-border/50"
          : "bg-transparent"
      )}
    >
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-2 group">
            <div className="p-2 bg-gradient-to-br from-primary to-secondary rounded-xl group-hover:scale-110 transition-transform duration-300 glow">
              <Activity className="w-6 h-6 text-white" />
            </div>
            <span className="text-2xl font-bold gradient-text">
              Architrust
            </span>
          </Link>

          {/* Nav Links */}
          <ul className="hidden md:flex items-center gap-1">
            {navItems.map((item) => {
              const isActive = location.pathname === item.path;
              return (
                <li key={item.path}>
                  <Link
                    to={item.path}
                    className={cn(
                      "px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200",
                      isActive
                        ? "bg-primary text-primary-foreground shadow-md"
                        : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                    )}
                  >
                    {item.name}
                  </Link>
                </li>
              );
            })}
          </ul>

          {/* CTA Button */}
          <Link
            to="/training"
            className="hidden lg:block px-6 py-2.5 bg-gradient-to-r from-primary to-secondary text-white rounded-full font-semibold hover:scale-105 transition-transform duration-200 glow"
          >
            Start Evaluation
          </Link>
        </div>
      </div>
    </nav>
  );
};
