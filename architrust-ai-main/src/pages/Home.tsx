import { Link } from "react-router-dom";
import { Shield, TrendingUp, Eye, ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";

const trustPillars = [
  {
    icon: Shield,
    title: "Privacy-First Federated Learning",
    description:
      "Train AI models across multiple hospitals without sharing sensitive patient data. Secure, decentralized, and compliant.",
  },
  {
    icon: TrendingUp,
    title: "FLED Score Analytics",
    description:
      "Comprehensive evaluation of Fairness, Latency, Explainability, and Drift detection for trustworthy AI.",
  },
  {
    icon: Eye,
    title: "Explainable Clinical Insights",
    description:
      "SHAP and GradCAM visualizations help clinicians understand AI decisions and build trust in predictions.",
  },
];

const Home = () => {
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative pt-32 pb-20 px-6 overflow-hidden">
        {/* Animated gradient mesh background */}
        <div className="absolute inset-0 gradient-mesh-bg opacity-60" />
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-background/50 to-background" />

        <div className="relative max-w-6xl mx-auto text-center">
          <div className="animate-fade-in">
            <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
              <span className="gradient-text">Trustworthy</span> Federated
              <br />
              Learning for Medical Diagnostics
            </h1>
            <p className="text-xl md:text-2xl text-muted-foreground mb-12 max-w-3xl mx-auto">
              Evaluate fairness, efficiency & robustness with FLED Score
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <Link to="/training">
                <Button
                  size="lg"
                  className="px-8 py-6 text-lg bg-gradient-to-r from-primary to-secondary hover:scale-105 transition-transform duration-200 glow rounded-full"
                >
                  Start Evaluation
                  <ArrowRight className="ml-2 w-5 h-5" />
                </Button>
              </Link>
              <Link to="/fled">
                <Button
                  size="lg"
                  variant="outline"
                  className="px-8 py-6 text-lg border-2 hover:bg-muted/50 rounded-full"
                >
                  View Demo
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Trust Pillars Section */}
      <section className="py-20 px-6">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-4xl md:text-5xl font-bold text-center mb-16 animate-fade-in">
            Three Pillars of <span className="gradient-text">Trust</span>
          </h2>

          <div className="grid md:grid-cols-3 gap-8">
            {trustPillars.map((pillar, index) => {
              const Icon = pillar.icon;
              return (
                <div
                  key={pillar.title}
                  className="group glass-card rounded-2xl p-8 hover-lift animate-scale-in"
                  style={{ animationDelay: `${index * 0.1}s` }}
                >
                  <div className="mb-6 p-4 bg-gradient-to-br from-primary/10 to-accent/10 rounded-xl w-fit group-hover:scale-110 transition-transform duration-300">
                    <Icon className="w-10 h-10 text-primary" />
                  </div>
                  <h3 className="text-2xl font-bold mb-4 text-card-foreground">
                    {pillar.title}
                  </h3>
                  <p className="text-muted-foreground leading-relaxed">
                    {pillar.description}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-6">
        <div className="max-w-4xl mx-auto text-center glass-card rounded-3xl p-12 hover-lift">
          <h2 className="text-3xl md:text-4xl font-bold mb-6">
            Ready to Build Trust in AI?
          </h2>
          <p className="text-lg text-muted-foreground mb-8">
            Join leading healthcare institutions using Architrust for secure,
            fair, and explainable AI.
          </p>
          <Link to="/hospitals">
            <Button
              size="lg"
              className="px-8 py-6 text-lg bg-gradient-to-r from-primary to-secondary hover:scale-105 transition-transform duration-200 glow rounded-full"
            >
              Explore Hospital Nodes
              <ArrowRight className="ml-2 w-5 h-5" />
            </Button>
          </Link>
        </div>
      </section>
    </div>
  );
};

export default Home;
