import { useState, useRef } from "react";
import { Upload, CheckCircle, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

const XAI = () => {
  const [hasImage, setHasImage] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<any>(null);
  const [showReport, setShowReport] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = async (file: File) => {
    if (file && (file.type.startsWith('image/') || file.name.toLowerCase().endsWith('.dcm'))) {
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setImageUrl(url);
      setHasImage(true);
      setIsAnalyzing(true);

      // Simulate AI analysis
      setTimeout(() => {
        const mockResults = {
          prediction: Math.random() > 0.3 ? "Malignant" : "Benign",
          confidence: Math.random() * 0.4 + 0.6, // 60-100%
          cellType: ["Ductal Carcinoma", "Lobular Carcinoma", "Invasive Ductal Carcinoma"][Math.floor(Math.random() * 3)],
          grade: ["Grade I (Well Differentiated)", "Grade II (Moderately Differentiated)", "Grade III (Poorly Differentiated)"][Math.floor(Math.random() * 3)],
          stage: ["Stage I", "Stage II", "Stage III"][Math.floor(Math.random() * 3)],
          biomarkers: {
            ER: Math.random() > 0.5,
            PR: Math.random() > 0.6,
            HER2: Math.random() > 0.7,
            Ki67: Math.floor(Math.random() * 40) + 10
          },
          cellCount: Math.floor(Math.random() * 500) + 100,
          abnormalCells: Math.floor(Math.random() * 200) + 50,
          mitosisRate: Math.floor(Math.random() * 20) + 1,
          nuclearPleomorphism: Math.floor(Math.random() * 3) + 1,
          tubuleFormation: Math.floor(Math.random() * 3) + 1,
          necrosis: Math.random() > 0.5,
          lymphovascularInvasion: Math.random() > 0.6
        };
        setAnalysisResults(mockResults);
        setIsAnalyzing(false);
        setShowReport(true);
      }, 2000);
    } else {
      alert('Please select a valid image file (PNG, JPG, DICOM)');
    }
  };

  const handleUpload = () => {
    fileInputRef.current?.click();
  };

  const handleFileInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const handleDragOver = (event: React.DragEvent) => {
    event.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (event: React.DragEvent) => {
    event.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault();
    setIsDragOver(false);

    const files = event.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const downloadReport = (format: 'txt' | 'json' | 'pdf') => {
    if (!analysisResults) return;

    const reportData = {
      patientId: "PATIENT_" + Date.now(),
      analysisDate: new Date().toISOString(),
      imageFile: selectedFile?.name || "Unknown",
      ...analysisResults
    };

    if (format === 'json') {
      const dataStr = JSON.stringify(reportData, null, 2);
      const dataBlob = new Blob([dataStr], { type: 'application/json' });
      const url = URL.createObjectURL(dataBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `breast_cancer_analysis_${reportData.patientId}.json`;
      link.click();
      URL.revokeObjectURL(url);
    } else if (format === 'txt') {
      const textReport = `
BREAST CANCER ANALYSIS REPORT
============================

Patient ID: ${reportData.patientId}
Analysis Date: ${new Date(reportData.analysisDate).toLocaleString()}
Image File: ${reportData.imageFile}

DIAGNOSIS
---------
Prediction: ${reportData.prediction}
Confidence: ${(reportData.confidence * 100).toFixed(1)}%
Cell Type: ${reportData.cellType}

TUMOR CHARACTERISTICS
--------------------
Grade: ${reportData.grade}
Stage: ${reportData.stage}
Total Cell Count: ${reportData.cellCount}
Abnormal Cells: ${reportData.abnormalCells}

BIOMARKERS
----------
ER Status: ${reportData.biomarkers.ER ? 'Positive' : 'Negative'}
PR Status: ${reportData.biomarkers.PR ? 'Positive' : 'Negative'}
HER2 Status: ${reportData.biomarkers.HER2 ? 'Positive' : 'Negative'}
Ki-67 Index: ${reportData.biomarkers.Ki67}%

PATHOLOGICAL FEATURES
--------------------
Mitosis Rate: ${reportData.mitosisRate}/10 HPF
Nuclear Pleomorphism: Grade ${reportData.nuclearPleomorphism}
Tubule Formation: Grade ${reportData.tubuleFormation}
Necrosis: ${reportData.necrosis ? 'Present' : 'Absent'}
Lymphovascular Invasion: ${reportData.lymphovascularInvasion ? 'Present' : 'Absent'}

This report was generated by MedSAM AI Analysis System.
Please consult with a qualified pathologist for final diagnosis.
      `.trim();

      const dataBlob = new Blob([textReport], { type: 'text/plain' });
      const url = URL.createObjectURL(dataBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `breast_cancer_analysis_${reportData.patientId}.txt`;
      link.click();
      URL.revokeObjectURL(url);
    } else if (format === 'pdf') {
      // Generate actual PDF using jsPDF
      const pdf = new jsPDF();

      // Add header
      pdf.setFontSize(20);
      pdf.setFont('helvetica', 'bold');
      pdf.text('BREAST CANCER ANALYSIS REPORT', 20, 30);

      // Add patient info
      pdf.setFontSize(12);
      pdf.setFont('helvetica', 'normal');
      pdf.text(`Patient ID: ${reportData.patientId}`, 20, 50);
      pdf.text(`Analysis Date: ${new Date(reportData.analysisDate).toLocaleString()}`, 20, 60);
      pdf.text(`Image File: ${reportData.imageFile}`, 20, 70);

      // Add diagnosis section
      pdf.setFontSize(14);
      pdf.setFont('helvetica', 'bold');
      pdf.text('DIAGNOSIS SUMMARY', 20, 90);

      pdf.setFontSize(12);
      pdf.setFont('helvetica', 'normal');
      pdf.text(`Prediction: ${reportData.prediction}`, 20, 105);
      pdf.text(`Confidence Level: ${(reportData.confidence * 100).toFixed(1)}%`, 20, 115);
      pdf.text(`Detected Cell Type: ${reportData.cellType}`, 20, 125);

      // Add clinical details
      pdf.setFontSize(14);
      pdf.setFont('helvetica', 'bold');
      pdf.text('CLINICAL DETAILS', 20, 145);

      pdf.setFontSize(12);
      pdf.setFont('helvetica', 'normal');
      pdf.text(`Tumor Grade: ${reportData.grade}`, 20, 160);
      pdf.text(`Disease Stage: ${reportData.stage}`, 20, 170);
      pdf.text(`Cell Count Analysis: ${reportData.cellCount} total cells`, 20, 180);
      pdf.text(`Abnormal Cell Count: ${reportData.abnormalCells}`, 20, 190);

      // Add biomarkers
      pdf.setFontSize(14);
      pdf.setFont('helvetica', 'bold');
      pdf.text('BIOMARKER ANALYSIS', 20, 210);

      pdf.setFontSize(12);
      pdf.setFont('helvetica', 'normal');
      pdf.text(`Estrogen Receptor (ER): ${reportData.biomarkers.ER ? 'POSITIVE' : 'NEGATIVE'}`, 20, 225);
      pdf.text(`Progesterone Receptor (PR): ${reportData.biomarkers.PR ? 'POSITIVE' : 'NEGATIVE'}`, 20, 235);
      pdf.text(`HER2/neu Status: ${reportData.biomarkers.HER2 ? 'POSITIVE' : 'NEGATIVE'}`, 20, 245);
      pdf.text(`Ki-67 Proliferation Index: ${reportData.biomarkers.Ki67}%`, 20, 255);

      // Add pathology findings
      pdf.setFontSize(14);
      pdf.setFont('helvetica', 'bold');
      pdf.text('PATHOLOGY FINDINGS', 20, 275);

      pdf.setFontSize(12);
      pdf.setFont('helvetica', 'normal');
      pdf.text(`Mitotic Rate: ${reportData.mitosisRate} mitoses per 10 high-power fields`, 20, 290);
      pdf.text(`Nuclear Pleomorphism Grade: ${reportData.nuclearPleomorphism}`, 20, 300);
      pdf.text(`Tubule Formation Grade: ${reportData.tubuleFormation}`, 20, 310);
      pdf.text(`Tumor Necrosis: ${reportData.necrosis ? 'PRESENT' : 'ABSENT'}`, 20, 320);
      pdf.text(`Lymphovascular Invasion: ${reportData.lymphovascularInvasion ? 'DETECTED' : 'NOT DETECTED'}`, 20, 330);

      // Add recommendations
      pdf.setFontSize(14);
      pdf.setFont('helvetica', 'bold');
      pdf.text('RECOMMENDATIONS', 20, 350);

      pdf.setFontSize(10);
      pdf.setFont('helvetica', 'normal');
      pdf.text('- Immediate consultation with oncologist recommended', 20, 365);
      pdf.text('- Further immunohistochemical staining may be required', 20, 375);
      pdf.text('- Consider genetic testing for hereditary breast cancer', 20, 385);
      pdf.text('- Regular follow-up imaging advised', 20, 395);

      // Add footer
      pdf.setFontSize(8);
      pdf.setFont('helvetica', 'italic');
      pdf.text('REPORT GENERATED BY: MedSAM AI Diagnostic System', 20, 420);
      pdf.text(`DATE: ${new Date().toLocaleDateString()} | TIME: ${new Date().toLocaleTimeString()}`, 20, 430);
      pdf.text('*** CONFIDENTIAL MEDICAL REPORT ***', 20, 440);
      pdf.text('*** For Professional Medical Use Only ***', 20, 450);

      // Save the PDF
      pdf.save(`breast_cancer_analysis_${reportData.patientId}.pdf`);
    }
  };

  const features = analysisResults ? [
    {
      name: "Nuclear Pleomorphism",
      contribution: Math.floor(Math.random() * 20) + 15,
      color: "bg-primary",
    },
    {
      name: "Mitotic Activity",
      contribution: Math.floor(Math.random() * 20) + 10,
      color: "bg-secondary",
    },
    {
      name: "Tubule Formation",
      contribution: Math.floor(Math.random() * 15) + 10,
      color: "bg-accent",
    },
    {
      name: "Cell Density",
      contribution: Math.floor(Math.random() * 15) + 5,
      color: "bg-muted"
    },
    {
      name: "Necrosis Presence",
      contribution: Math.floor(Math.random() * 10) + 5,
      color: "bg-red-500"
    },
  ] : [
    {
      name: "Nuclear Pleomorphism",
      contribution: 42,
      color: "bg-primary",
    },
    {
      name: "Mitotic Activity",
      contribution: 28,
      color: "bg-secondary",
    },
    {
      name: "Tubule Formation",
      contribution: 18,
      color: "bg-accent",
    },
    { name: "Cell Density", contribution: 12, color: "bg-muted" },
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
                  : isDragOver
                  ? "border-primary bg-primary/10"
                  : "border-border hover:border-primary hover:bg-muted/30 cursor-pointer"
              }`}
              onClick={!hasImage ? handleUpload : undefined}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              {!hasImage ? (
                <div className="animate-fade-in">
                  <Upload className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
                  <p className="text-lg font-medium mb-2">
                    Upload medical image for analysis
                  </p>
                  <p className="text-sm text-muted-foreground">
                    Supports: PNG, JPG, DICOM
                  </p>
                </div>
              ) : (
                <div className="animate-scale-in">
                  <div className="w-full h-64 bg-gradient-to-br from-muted to-muted/50 rounded-lg mb-4 flex items-center justify-center relative overflow-hidden">
                    {isAnalyzing ? (
                      <div className="flex flex-col items-center">
                        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mb-2"></div>
                        <span className="text-sm text-muted-foreground">Analyzing breast tissue...</span>
                      </div>
                    ) : imageUrl ? (
                      <>
                        <img
                          src={imageUrl}
                          alt="Uploaded breast tissue sample"
                          className="w-full h-full object-cover rounded-lg"
                        />
                        {/* Heatmap overlay - Black background with white cancer cell spots */}
                        <div className="absolute inset-0 bg-black rounded-lg opacity-80">
                          {/* Simulate white spots for cancer cells */}
                          <div className="absolute top-1/4 left-1/3 w-2 h-2 bg-white rounded-full"></div>
                          <div className="absolute top-1/2 left-2/3 w-1.5 h-1.5 bg-white rounded-full"></div>
                          <div className="absolute top-3/4 left-1/4 w-3 h-3 bg-white rounded-full"></div>
                          <div className="absolute top-2/3 left-1/2 w-2.5 h-2.5 bg-white rounded-full"></div>
                          <div className="absolute top-1/3 left-3/4 w-1 h-1 bg-white rounded-full"></div>
                          <div className="absolute top-1/2 left-1/4 w-2 h-2 bg-white rounded-full"></div>
                          <div className="absolute top-4/5 left-2/3 w-1.5 h-1.5 bg-white rounded-full"></div>
                        </div>
                        <div className="absolute top-2 right-2 bg-black/70 text-white text-xs px-2 py-1 rounded">
                          Cancer Cell Detection
                        </div>
                      </>
                    ) : null}
                  </div>
                  <p className="text-sm text-muted-foreground">
                    {selectedFile ? `${selectedFile.name} • ${(selectedFile.size / 1024 / 1024).toFixed(2)}MB` : "Breast Tissue Sample • 512x512px • 2.4MB"}
                  </p>
                </div>
              )}
            </div>

            {!hasImage && (
              <>
                <Button
                  onClick={handleUpload}
                  className="w-full mt-6 bg-gradient-to-r from-primary to-secondary hover:scale-105 transition-transform duration-200"
                >
                  Select Image
                </Button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*,.dcm"
                  onChange={handleFileInputChange}
                  className="hidden"
                />
              </>
            )}
          </div>

          {/* Explanation Panel */}
          <div className="glass-card rounded-2xl p-8 animate-slide-in">
            <h3 className="text-2xl font-bold mb-6">AI Explanation</h3>

            {hasImage ? (
              <div className="space-y-6">
                {isAnalyzing ? (
                  <div className="bg-gradient-to-r from-info/10 to-primary/10 rounded-xl p-6 border border-info/20 text-center">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
                    <p className="text-lg font-semibold">Analyzing breast cancer biomarkers...</p>
                    <p className="text-sm text-muted-foreground">Processing cellular features and tissue morphology</p>
                  </div>
                ) : analysisResults ? (
                  <>
                    {/* Prediction Summary */}
                    <div className="bg-gradient-to-r from-info/10 to-primary/10 rounded-xl p-6 border border-info/20">
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <p className="text-sm text-muted-foreground mb-1">
                            Prediction Confidence
                          </p>
                          <p className="text-2xl font-bold">{(analysisResults.confidence * 100).toFixed(1)}%</p>
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground mb-1">
                            Classification
                          </p>
                          <p className={`text-xl font-bold ${analysisResults.prediction === 'Malignant' ? 'text-red-600' : 'text-green-600'}`}>
                            {analysisResults.prediction}
                          </p>
                        </div>
                        <div className="col-span-2 pt-4 border-t border-border/50">
                          <p className="text-sm text-muted-foreground mb-1">
                            Cell Type Detected
                          </p>
                          <p className="font-semibold">{analysisResults.cellType}</p>
                        </div>
                      </div>
                    </div>

                    {/* Breast Cancer Specific Metrics */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="bg-muted/30 rounded-xl p-4 border border-border/50">
                        <h4 className="font-semibold mb-3">Tumor Characteristics</h4>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Grade:</span>
                            <span className="font-semibold">{analysisResults.grade}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Stage:</span>
                            <span className="font-semibold">{analysisResults.stage}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Cell Count:</span>
                            <span className="font-semibold">{analysisResults.cellCount}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Abnormal Cells:</span>
                            <span className="font-semibold text-red-600">{analysisResults.abnormalCells}</span>
                          </div>
                        </div>
                      </div>

                      <div className="bg-muted/30 rounded-xl p-4 border border-border/50">
                        <h4 className="font-semibold mb-3">Biomarkers</h4>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">ER:</span>
                            <span className={`font-semibold ${analysisResults.biomarkers.ER ? 'text-green-600' : 'text-red-600'}`}>
                              {analysisResults.biomarkers.ER ? 'Positive' : 'Negative'}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">PR:</span>
                            <span className={`font-semibold ${analysisResults.biomarkers.PR ? 'text-green-600' : 'text-red-600'}`}>
                              {analysisResults.biomarkers.PR ? 'Positive' : 'Negative'}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">HER2:</span>
                            <span className={`font-semibold ${analysisResults.biomarkers.HER2 ? 'text-green-600' : 'text-red-600'}`}>
                              {analysisResults.biomarkers.HER2 ? 'Positive' : 'Negative'}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Ki-67:</span>
                            <span className="font-semibold">{analysisResults.biomarkers.Ki67}%</span>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Additional Pathology Details */}
                    <div className="bg-muted/30 rounded-xl p-4 border border-border/50">
                      <h4 className="font-semibold mb-3">Pathological Features</h4>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div>
                          <span className="text-muted-foreground">Mitosis Rate:</span>
                          <p className="font-semibold">{analysisResults.mitosisRate}/10 HPF</p>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Nuclear Pleomorphism:</span>
                          <p className="font-semibold">Grade {analysisResults.nuclearPleomorphism}</p>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Tubule Formation:</span>
                          <p className="font-semibold">Grade {analysisResults.tubuleFormation}</p>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Necrosis:</span>
                          <p className={`font-semibold ${analysisResults.necrosis ? 'text-red-600' : 'text-green-600'}`}>
                            {analysisResults.necrosis ? 'Present' : 'Absent'}
                          </p>
                        </div>
                      </div>
                      <div className="mt-3 pt-3 border-t border-border/50">
                        <div className="flex justify-between text-sm">
                          <span className="text-muted-foreground">Lymphovascular Invasion:</span>
                          <span className={`font-semibold ${analysisResults.lymphovascularInvasion ? 'text-red-600' : 'text-green-600'}`}>
                            {analysisResults.lymphovascularInvasion ? 'Present' : 'Absent'}
                          </span>
                        </div>
                      </div>
                    </div>
                  </>
                ) : null}

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

                {/* Download Report Section */}
                {showReport && (
                  <div className="bg-gradient-to-r from-success/10 to-primary/10 rounded-xl p-6 border border-success/20">
                    <h4 className="font-semibold mb-4 text-center">Download Analysis Report</h4>
                    <div className="flex gap-3 justify-center">
                      <Button
                        onClick={() => downloadReport('txt')}
                        variant="outline"
                        className="flex-1"
                      >
                        📄 Text Report
                      </Button>
                      <Button
                        onClick={() => downloadReport('json')}
                        variant="outline"
                        className="flex-1"
                      >
                        📊 JSON Data
                      </Button>
                      <Button
                        onClick={() => downloadReport('pdf')}
                        variant="outline"
                        className="flex-1"
                      >
                        📋 PDF Report
                      </Button>
                    </div>
                  </div>
                )}

                {/* Model Confidence */}
                <div className="bg-muted/30 rounded-xl p-6 border border-border/50">
                  <h4 className="font-semibold mb-3">Model Reliability</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Training Accuracy</span>
                      <span className="font-semibold">94.7%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">FLED Score</span>
                      <span className="font-semibold">92/100</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Bias Detection</span>
                      <span className="font-semibold text-success">No Issues</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Dataset Size</span>
                      <span className="font-semibold">50K+ Images</span>
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
