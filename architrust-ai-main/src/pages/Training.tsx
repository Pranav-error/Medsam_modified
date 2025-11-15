import { useState, useEffect } from "react";
import { Activity, Clock, Users, Zap } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";

const Training = () => {
  const { toast } = useToast();
  const [progress, setProgress] = useState(75);
  const [currentRound, setCurrentRound] = useState(3);

  useEffect(() => {
    // Simulate progress animation
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          setCurrentRound((r) => Math.min(r + 1, 4));
          return 0;
        }
        return prev + 1;
      });
    }, 100);

    return () => clearInterval(interval);
  }, []);

  const rounds = [
    {
      round: 1,
      accuracy: "82.3%",
      fairnessDev: "0.12",
      participation: "100%",
      commTime: "2.3s",
    },
    {
      round: 2,
      accuracy: "87.8%",
      fairnessDev: "0.09",
      participation: "100%",
      commTime: "2.1s",
    },
    {
      round: 3,
      accuracy: "91.2%",
      fairnessDev: "0.06",
      participation: "100%",
      commTime: "2.0s",
    },
    {
      round: 4,
      accuracy: "93.5%",
      fairnessDev: "0.04",
      participation: "100%",
      commTime: "1.9s",
    },
  ];

  const currentMetrics = [
    { label: "Global Accuracy", value: "91.2%", icon: Activity, color: "text-primary" },
    { label: "Avg Round Time", value: "2.0s", icon: Clock, color: "text-accent" },
    { label: "Active Nodes", value: "3/3", icon: Users, color: "text-success" },
    { label: "Fairness Score", value: "94/100", icon: Zap, color: "text-secondary" },
  ];

  return (
    <div className="min-h-screen pt-28 pb-20 px-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12 animate-fade-in">
          <h1 className="text-5xl font-bold mb-4">
            Federated <span className="gradient-text">Training</span> Dashboard
          </h1>
          <p className="text-xl text-muted-foreground">
            Real-time monitoring of distributed model training
          </p>
        </div>

        {/* Current Metrics */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {currentMetrics.map((metric, index) => {
            const Icon = metric.icon;
            return (
              <div
                key={metric.label}
                className="glass-card rounded-xl p-6 animate-scale-in hover-lift"
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                <div className="flex items-center justify-between mb-2">
                  <Icon className={`w-5 h-5 ${metric.color}`} />
                </div>
                <div className="text-3xl font-bold mb-1">{metric.value}</div>
                <div className="text-sm text-muted-foreground">{metric.label}</div>
              </div>
            );
          })}
        </div>

        {/* Progress Section */}
        <div className="glass-card rounded-2xl p-8 mb-8 animate-fade-in">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold">Training Progress</h2>
            <div className="text-lg font-semibold text-primary">
              Round {currentRound} of 4
            </div>
          </div>

          <Progress value={progress} className="h-3 mb-6" />

          <div className="text-center py-4">
            <p className="text-muted-foreground flex items-center justify-center gap-2 flex-wrap">
              <span className="px-3 py-1 bg-primary/10 text-primary rounded-full text-sm font-medium">
                Hospital A
              </span>
              <span>→</span>
              <span className="px-3 py-1 bg-secondary/10 text-secondary rounded-full text-sm font-medium">
                Hospital B
              </span>
              <span>→</span>
              <span className="px-3 py-1 bg-accent/10 text-accent rounded-full text-sm font-medium">
                Hospital C
              </span>
              <span>→</span>
              <span className="px-3 py-1 bg-muted text-muted-foreground rounded-full text-sm font-medium">
                Central Aggregator
              </span>
            </p>
          </div>
        </div>

        {/* Round Metrics Table */}
        <div className="glass-card rounded-2xl p-8 animate-fade-in">
          <h2 className="text-2xl font-bold mb-6">Round-by-Round Metrics</h2>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-border/50">
                  <th className="text-left py-4 px-4 text-muted-foreground font-semibold">
                    Round
                  </th>
                  <th className="text-left py-4 px-4 text-muted-foreground font-semibold">
                    Accuracy
                  </th>
                  <th className="text-left py-4 px-4 text-muted-foreground font-semibold">
                    Fairness Dev.
                  </th>
                  <th className="text-left py-4 px-4 text-muted-foreground font-semibold">
                    Participation
                  </th>
                  <th className="text-left py-4 px-4 text-muted-foreground font-semibold">
                    Comm. Time
                  </th>
                </tr>
              </thead>
              <tbody>
                {rounds.map((round, index) => (
                  <tr
                    key={round.round}
                    className="border-b border-border/30 hover:bg-muted/30 transition-colors"
                  >
                    <td className="py-4 px-4 font-semibold">Round {round.round}</td>
                    <td className="py-4 px-4">{round.accuracy}</td>
                    <td className="py-4 px-4">{round.fairnessDev}</td>
                    <td className="py-4 px-4">
                      <span className="px-2 py-1 bg-success/10 text-success rounded-md text-sm">
                        {round.participation}
                      </span>
                    </td>
                    <td className="py-4 px-4">{round.commTime}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Training;
