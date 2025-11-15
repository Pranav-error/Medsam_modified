import { Link } from "react-router-dom";
import { Building2, Database, AlertCircle, CheckCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

const hospitals = [
  {
    id: "A",
    name: "Hospital A",
    datasetSize: "15,000 records",
    nonIIDSeverity: "low",
    status: "active",
    location: "Boston, MA",
    specialization: "Radiology",
  },
  {
    id: "B",
    name: "Hospital B",
    datasetSize: "12,500 records",
    nonIIDSeverity: "medium",
    status: "active",
    location: "San Francisco, CA",
    specialization: "Pulmonology",
  },
  {
    id: "C",
    name: "Hospital C",
    datasetSize: "8,000 records",
    nonIIDSeverity: "high",
    status: "active",
    location: "New York, NY",
    specialization: "Emergency Care",
  },
];

const getSeverityBadge = (severity: string) => {
  const variants = {
    low: "bg-success/10 text-success border-success/20",
    medium: "bg-warning/10 text-warning border-warning/20",
    high: "bg-destructive/10 text-destructive border-destructive/20",
  };

  return variants[severity as keyof typeof variants] || variants.low;
};

const Hospitals = () => {
  return (
    <div className="min-h-screen pt-28 pb-20 px-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-16 animate-fade-in">
          <h1 className="text-5xl font-bold mb-4">
            Hospital <span className="gradient-text">Nodes</span>
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Decentralized network of healthcare institutions participating in
            federated learning
          </p>
        </div>

        {/* Hospital Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
          {hospitals.map((hospital, index) => (
            <div
              key={hospital.id}
              className="glass-card rounded-2xl p-6 hover-lift animate-scale-in border-l-4 border-primary"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              {/* Header */}
              <div className="flex items-start justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="p-3 bg-gradient-to-br from-primary/20 to-accent/20 rounded-xl">
                    <Building2 className="w-6 h-6 text-primary" />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold">{hospital.name}</h3>
                    <p className="text-sm text-muted-foreground">
                      {hospital.location}
                    </p>
                  </div>
                </div>
                <Badge className="bg-success/10 text-success border-success/20 gap-1">
                  <CheckCircle className="w-3 h-3" />
                  Active
                </Badge>
              </div>

              {/* Stats */}
              <div className="space-y-3 mb-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <Database className="w-4 h-4" />
                    <span className="text-sm">Dataset Size</span>
                  </div>
                  <span className="font-semibold">{hospital.datasetSize}</span>
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <AlertCircle className="w-4 h-4" />
                    <span className="text-sm">Non-IID Severity</span>
                  </div>
                  <Badge
                    className={getSeverityBadge(hospital.nonIIDSeverity)}
                  >
                    {hospital.nonIIDSeverity.charAt(0).toUpperCase() +
                      hospital.nonIIDSeverity.slice(1)}
                  </Badge>
                </div>

                <div className="pt-2 border-t border-border/50">
                  <p className="text-sm text-muted-foreground">
                    Specialization
                  </p>
                  <p className="font-medium">{hospital.specialization}</p>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* CTA Section */}
        <div className="text-center glass-card rounded-2xl p-12 animate-fade-in">
          <h2 className="text-3xl font-bold mb-4">
            Ready to Start Training?
          </h2>
          <p className="text-muted-foreground mb-8 max-w-2xl mx-auto">
            Launch federated training simulation across all hospital nodes
            without sharing sensitive patient data
          </p>
          <Link to="/training">
            <Button
              size="lg"
              className="px-8 py-6 text-lg bg-gradient-to-r from-primary to-secondary hover:scale-105 transition-transform duration-200 glow rounded-full"
            >
              Run Federated Training Simulation
            </Button>
          </Link>
        </div>
      </div>
    </div>
  );
};

export default Hospitals;
