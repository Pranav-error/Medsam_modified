import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Navigation } from "@/components/Navigation";
import Home from "./pages/Home";
import Hospitals from "./pages/Hospitals";
import Training from "./pages/Training";
import FledScore from "./pages/FledScore";
import XAI from "./pages/XAI";
import Reports from "./pages/Reports";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <div className="min-h-screen bg-background">
          <Navigation />
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/hospitals" element={<Hospitals />} />
            <Route path="/training" element={<Training />} />
            <Route path="/fled" element={<FledScore />} />
            <Route path="/xai" element={<XAI />} />
            <Route path="/reports" element={<Reports />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
          
          {/* Footer */}
          <footer className="bg-card border-t border-border/50 mt-20">
            <div className="max-w-7xl mx-auto px-6 py-12 text-center">
              <p className="text-muted-foreground mb-2">
                &copy; 2024 Architrust - Trustworthy Federated Learning for Healthcare
              </p>
              <p className="text-sm text-muted-foreground/70">
                Built for Medical AI Innovation
              </p>
            </div>
          </footer>
        </div>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
