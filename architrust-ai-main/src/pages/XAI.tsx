import { useState } from "react";
import { Upload, CheckCircle, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";

const XAI = () => {
  const [hasImage, setHasImage] = useState(false);
  const { toast } = useToast();
  const [imageInfo, setImageInfo] = useState(null);

  const features = [
    {
      name: "Lung opacity (left lower lobe)",
      contribution: 42,
      color: "bg-primary",
    },
    {
      name: "Consolidation pattern",
      contribution: 28,
      color: "bg-secondary",
    },
    {
      name: "Air bronchogram presence",
      contribution: 18,
      color: "bg-accent",
    },
    { name: "Cardiac silhouette", contribution: 12, color: "bg-muted" },
  ];

  return (
    <div className="min-h-screen pt-28 pb-20 px-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-16 animate-fade-in">
          <h1 className="text-5xl font-bold mb-4">
            Explainability <span className="gradient-text">(XAI)</span>
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Understand AI decisions with SHAP and GradCAM visualizations
          </p>
        </div>

        {/* Main Content */}
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Image Viewer */}
          <div className="glass-card rounded-2xl p-8 animate-scale-in">
            <h3 className="text-2xl font-bold mb-6">Clinical Image Analysis</h3>

            <div
              className={`border-3 border-dashed rounded-xl p-12 text-center transition-all duration-300 ${
                hasImage
                  ? "border-primary bg-primary/5"
                  : "border-border hover:border-primary hover:bg-muted/30 cursor-pointer"
              }`}
            >
              <input
                type="file"
                accept="image/*"
                className="hidden"
                id="imageUploadInput"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (!file) return;
                
                  // --- CONDITION 1: File must be ABOVE 5MB ---
                  const minSizeMB = 5;
                  if (file.size < minSizeMB * 1024 * 1024) {
                    toast({
                      title: "File Too Small",
                      description: `Image must be bigger than ${minSizeMB} MB.`,
                      variant: "destructive",
                    });
                    return;
                  }
                
                  // --- CONDITION 2: Resolution must be ABOVE 512x512 ---
                  const img = new Image();
                  img.src = URL.createObjectURL(file);
                
                  img.onload = () => {
                    const width = img.width;
                    const height = img.height;
                  
                    const minWidth = 512;
                    const minHeight = 512;
                  
                    if (width <= minWidth || height <= minHeight) {
                      toast({
                        title: "Resolution Too Small",
                        description: `Image must be larger than ${minWidth}×${minHeight} pixels.`,
                        variant: "destructive",
                      });
                      return;
                    }
                  
                    // VALID FILE → Accept
                    setHasImage(true);
                    setImageInfo({
                      size: (file.size / (1024 * 1024)).toFixed(1),
                      width,
                      height,
                    });
                  
                    toast({
                      title: "Image Accepted",
                      description: "Your upload meets all requirements.",
                    });
                  };
                }}
              />

              {!hasImage ? (
                <label htmlFor="imageUploadInput" className="block cursor-pointer">
                  <Upload className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
                  <p className="text-lg font-medium mb-2">Upload medical image for analysis</p>
                  <p className="text-sm text-muted-foreground">Supports: PNG, JPG, DICOM</p>
                </label>
              ) : (
                <div className="animate-scale-in">
                  <div className="w-full h-64 bg-gradient-to-br from-muted to-muted/50 rounded-lg mb-4 flex items-center justify-center">
                    <span className="text-4xl">🫁</span>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    {`Resolution: ${imageInfo.width}×${imageInfo.height}px • ${imageInfo.size}MB`}
                  </p>
                </div>
              )}
            </div>
            {!hasImage && (
              <Button
                onClick={() => document.getElementById("imageUploadInput")?.click()}
                className="w-full mt-6 bg-gradient-to-r from-primary to-secondary hover:scale-105 transition-transform duration-200"
              >
                Select Image
              </Button>
            )}
          </div>

          {/* Explanation Panel */}
          <div className="glass-card rounded-2xl p-8 animate-slide-in">
            <h3 className="text-2xl font-bold mb-6">AI Explanation</h3>

            {hasImage ? (
              <div className="space-y-6">
                {/* Prediction Summary */}
                <div className="bg-gradient-to-r from-info/10 to-primary/10 rounded-xl p-6 border border-info/20">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm text-muted-foreground mb-1">
                        Prediction Confidence
                      </p>
                      <p className="text-2xl font-bold">87%</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground mb-1">
                        Classification
                      </p>
                      <p className="text-xl font-bold">Pneumonia Detected</p>
                    </div>
                    <div className="col-span-2 pt-4 border-t border-border/50">
                      <p className="text-sm text-muted-foreground mb-1">
                        Fairness Check
                      </p>
                      <div className="flex items-center gap-2">
                        <CheckCircle className="w-5 h-5 text-success" />
                        <span className="font-semibold text-success">Pass</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Feature Contributions */}
                <div>
                  <h4 className="font-semibold text-lg mb-4">
                    Top Contributing Features
                  </h4>
                  <div className="space-y-3">
                    {features.map((feature, index) => (
                      <div
                        key={feature.name}
                        className="bg-muted/30 rounded-lg p-4 border border-border/50 hover:border-primary/50 transition-colors animate-slide-in"
                        style={{ animationDelay: `${index * 0.1}s` }}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-medium">{feature.name}</span>
                          <span className="text-sm font-bold text-primary">
                            {feature.contribution}%
                          </span>
                        </div>
                        <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
                          <div
                            className={`h-full ${feature.color} transition-all duration-1000`}
                            style={{ width: `${feature.contribution}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Model Confidence */}
                <div className="bg-muted/30 rounded-xl p-6 border border-border/50">
                  <h4 className="font-semibold mb-3">Model Reliability</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Training Accuracy</span>
                      <span className="font-semibold">91.2%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">FLED Score</span>
                      <span className="font-semibold">87/100</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Bias Detection</span>
                      <span className="font-semibold text-success">No Issues</span>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-12">
                <AlertCircle className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
                <p className="text-muted-foreground">
                  Upload an image to view AI explanation and feature analysis
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default XAI;
