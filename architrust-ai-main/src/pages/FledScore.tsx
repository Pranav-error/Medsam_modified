import { TrendingUp, Zap, Eye, Shield } from "lucide-react";

const metrics = [
  {
    title: "Fairness Score",
    value: 92,
    icon: Shield,
    color: "from-primary to-primary-glow",
  },
  {
    title: "Latency/Efficiency",
    value: 85,
    icon: Zap,
    color: "from-accent to-success",
  },
  {
    title: "Explainability",
    value: 88,
    icon: Eye,
    color: "from-secondary to-info",
  },
  {
    title: "Drift/Robustness",
    value: 83,
    icon: TrendingUp,
    color: "from-warning to-destructive",
  },
];

const insights = [
  {
    title: "📈 Fairness Analysis",
    description:
      "Low deviation across hospitals (σ = 0.06). All nodes meet fairness threshold.",
  },
  {
    title: "⚡ Performance Metrics",
    description:
      "Average communication time: 2.1s per round. Model size optimized to 12.3MB.",
  },
  {
    title: "🎯 Model Quality",
    description:
      "Global accuracy: 91.2%. No significant drift detected across participating nodes.",
  },
];

const FledScore = () => {
  return (
    <div className="min-h-screen pt-28 pb-20 px-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-16 animate-fade-in">
          <h1 className="text-5xl font-bold mb-4">
            FLED <span className="gradient-text">Score</span> Dashboard
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Comprehensive evaluation of Fairness, Latency, Explainability, and
            Drift
          </p>
        </div>

        {/* Score Container */}
        <div className="grid lg:grid-cols-3 gap-8 mb-12">
          {/* Main Score */}
          <div className="lg:col-span-1 glass-card rounded-3xl p-10 text-center hover-lift animate-scale-in bg-gradient-to-br from-primary/10 via-secondary/10 to-accent/10 glow">
            <h2 className="text-lg uppercase tracking-wider text-muted-foreground mb-4 font-semibold">
              Overall FLED Score
            </h2>
            <div className="text-8xl font-bold gradient-text my-8">87</div>
            <p className="text-lg font-semibold text-success">
              Excellent Performance
            </p>
            <div className="mt-6 pt-6 border-t border-border/50">
              <p className="text-sm text-muted-foreground">
                Top 10% of evaluated models
              </p>
            </div>
          </div>

          {/* Metrics Breakdown */}
          <div className="lg:col-span-2 grid grid-cols-2 gap-4">
            {metrics.map((metric, index) => {
              const Icon = metric.icon;
              return (
                <div
                  key={metric.title}
                  className="glass-card rounded-2xl p-6 hover-lift animate-scale-in"
                  style={{ animationDelay: `${index * 0.1}s` }}
                >
                  <div className="flex items-center gap-3 mb-4">
                    <div
                      className={`p-3 bg-gradient-to-br ${metric.color} rounded-xl`}
                    >
                      <Icon className="w-5 h-5 text-white" />
                    </div>
                    <h4 className="text-sm font-medium text-muted-foreground">
                      {metric.title}
                    </h4>
                  </div>
                  <div className="text-5xl font-bold gradient-text mb-2">
                    {metric.value}
                  </div>
                  <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
                    <div
                      className={`h-full bg-gradient-to-r ${metric.color} transition-all duration-1000`}
                      style={{ width: `${metric.value}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Insights Grid */}
        <div className="grid md:grid-cols-3 gap-6 mb-12">
          {insights.map((insight, index) => (
            <div
              key={insight.title}
              className="glass-card rounded-2xl p-6 hover-lift animate-slide-in"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <h3 className="text-xl font-bold mb-3">{insight.title}</h3>
              <p className="text-muted-foreground leading-relaxed">
                {insight.description}
              </p>
            </div>
          ))}
        </div>

        {/* Detailed Analysis */}
        <div className="glass-card rounded-2xl p-8 animate-fade-in">
          <h2 className="text-2xl font-bold mb-6">Detailed Analysis</h2>
          
          <div className="space-y-6">
            <div className="border-l-4 border-primary pl-6">
              <h3 className="font-semibold text-lg mb-2">Fairness Assessment</h3>
              <p className="text-muted-foreground">
                All hospital nodes demonstrate consistent performance with minimal
                bias. Standard deviation of 0.06 indicates excellent fairness across
                diverse patient populations.
              </p>
            </div>

            <div className="border-l-4 border-accent pl-6">
              <h3 className="font-semibold text-lg mb-2">Efficiency Metrics</h3>
              <p className="text-muted-foreground">
                Communication overhead optimized with an average round time of 2.1
                seconds. Model compression techniques reduced size to 12.3MB without
                accuracy loss.
              </p>
            </div>

            <div className="border-l-4 border-secondary pl-6">
              <h3 className="font-semibold text-lg mb-2">Robustness Evaluation</h3>
              <p className="text-muted-foreground">
                No significant model drift detected across training rounds. All
                participating nodes maintain stable contribution to global model with
                91.2% accuracy.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FledScore;
