import { FileText, Download, FileSpreadsheet, FileJson, Shield } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";

const Reports = () => {
  const { toast } = useToast();

  const handleDownload = (format: string) => {
    toast({
      title: "Download Started",
      description: `Your ${format} report is being generated...`,
    });
  };

  const reportCards = [
    {
      icon: FileText,
      title: "Summary Report",
      description: "Comprehensive overview of model performance and metrics",
      metrics: [
        { label: "Global Model Accuracy", value: "91.2%" },
        { label: "FLED Score", value: "87/100" },
        { label: "Fairness Violations", value: "0" },
        { label: "Drift Alerts", value: "0" },
      ],
      actions: [
        { label: "Download PDF Report", format: "PDF", icon: FileText },
      ],
    },
    {
      icon: FileSpreadsheet,
      title: "Metrics Export",
      description: "Detailed training metrics and hospital-specific analytics",
      metrics: [
        { label: "Total Training Rounds", value: "4" },
        { label: "Participating Hospitals", value: "3" },
        { label: "Dataset Size", value: "35.5K" },
        { label: "Model Size", value: "12.3MB" },
      ],
      actions: [
        { label: "Download CSV Metrics", format: "CSV", icon: FileSpreadsheet },
        { label: "Download JSON Audit Log", format: "JSON", icon: FileJson },
      ],
    },
    {
      icon: Shield,
      title: "Compliance Audit",
      description: "Privacy preservation and regulatory compliance trail",
      metrics: [
        { label: "HIPAA Compliance", value: "✓ Pass" },
        { label: "Data Encryption", value: "AES-256" },
        { label: "Privacy Score", value: "98/100" },
        { label: "Audit Events", value: "247" },
      ],
      actions: [
        { label: "Generate Audit Report", format: "Audit", icon: Shield },
      ],
    },
  ];

  return (
    <div className="min-h-screen pt-28 pb-20 px-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-16 animate-fade-in">
          <h1 className="text-5xl font-bold mb-4">
            Evaluation <span className="gradient-text">Reports</span>
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Export comprehensive reports and analytics for compliance and analysis
          </p>
        </div>

        {/* Report Cards */}
        <div className="grid md:grid-cols-3 gap-8 mb-12">
          {reportCards.map((card, index) => {
            const Icon = card.icon;
            return (
              <div
                key={card.title}
                className="glass-card rounded-2xl p-8 hover-lift animate-scale-in"
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                {/* Header */}
                <div className="flex items-center gap-3 mb-6">
                  <div className="p-3 bg-gradient-to-br from-primary/20 to-accent/20 rounded-xl">
                    <Icon className="w-6 h-6 text-primary" />
                  </div>
                  <h3 className="text-xl font-bold">{card.title}</h3>
                </div>

                <p className="text-muted-foreground mb-6">{card.description}</p>

                {/* Metrics */}
                <div className="space-y-3 mb-6 p-4 bg-muted/30 rounded-xl">
                  {card.metrics.map((metric) => (
                    <div
                      key={metric.label}
                      className="flex justify-between items-center"
                    >
                      <span className="text-sm text-muted-foreground">
                        {metric.label}
                      </span>
                      <span className="font-semibold">{metric.value}</span>
                    </div>
                  ))}
                </div>

                {/* Actions */}
                <div className="space-y-2">
                  {card.actions.map((action) => {
                    const ActionIcon = action.icon;
                    return (
                      <Button
                        key={action.format}
                        onClick={() => handleDownload(action.format)}
                        className="w-full bg-gradient-to-r from-primary to-secondary hover:scale-105 transition-transform duration-200"
                      >
                        <ActionIcon className="w-4 h-4 mr-2" />
                        {action.label}
                      </Button>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>

        {/* Export All Section */}
        <div className="glass-card rounded-2xl p-12 text-center animate-fade-in">
          <h2 className="text-3xl font-bold mb-4">Export Complete Package</h2>
          <p className="text-muted-foreground mb-8 max-w-2xl mx-auto">
            Download all reports, metrics, and audit trails in a single
            comprehensive package
          </p>
          <Button
            onClick={() => handleDownload("Complete Package")}
            size="lg"
            className="px-8 py-6 text-lg bg-gradient-to-r from-primary to-secondary hover:scale-105 transition-transform duration-200 glow rounded-full"
          >
            <Download className="w-5 h-5 mr-2" />
            Download Complete Package
          </Button>
        </div>
      </div>
    </div>
  );
};

export default Reports;
