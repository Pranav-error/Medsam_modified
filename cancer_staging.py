"""
Cancer Staging and Medical Report Generation
Analyzes segmentation masks to estimate breast cancer stage and generate reports
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple
from datetime import datetime
import json


class BreastCancerStaging:
    """
    Breast cancer staging based on histopathology analysis.
    """
    
    # Thresholds for classification (can be tuned based on real data)
    STAGE_THRESHOLDS = {
        'normal': 0.05,      # < 5% abnormal tissue
        'stage_0': 0.15,     # 5-15% (in situ)
        'stage_1': 0.30,     # 15-30% (small tumor)
        'stage_2': 0.50,     # 30-50% (larger tumor)
        'stage_3': 0.70,     # 50-70% (regional spread)
        'stage_4': 1.0,      # > 70% (advanced)
    }
    
    def __init__(self):
        """Initialize the staging module."""
        pass
    
    def analyze_mask(self, mask: np.ndarray) -> Dict:
        """
        Analyze segmentation mask to extract features.
        
        Args:
            mask: Binary segmentation mask (H, W)
        
        Returns:
            Dictionary with mask features
        """
        # Basic statistics
        total_pixels = mask.shape[0] * mask.shape[1]
        positive_pixels = np.sum(mask > 0)
        coverage = positive_pixels / total_pixels
        
        # Find connected components (individual lesions)
        mask_uint8 = (mask * 255).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
        
        # Ignore background (label 0)
        num_lesions = num_labels - 1
        
        # Analyze each lesion
        lesion_sizes = []
        lesion_areas = []
        
        if num_lesions > 0:
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                lesion_areas.append(area)
                
                # Calculate equivalent diameter
                diameter = 2 * np.sqrt(area / np.pi)
                lesion_sizes.append(diameter)
        
        # Calculate density metrics
        if num_lesions > 0:
            avg_lesion_size = np.mean(lesion_sizes)
            max_lesion_size = np.max(lesion_sizes)
            total_lesion_area = np.sum(lesion_areas)
        else:
            avg_lesion_size = 0
            max_lesion_size = 0
            total_lesion_area = 0
        
        # Calculate spatial distribution (spread across image)
        if num_lesions > 0:
            centroid_coords = np.array([centroids[i] for i in range(1, num_labels)])
            if len(centroid_coords) > 1:
                spread = np.std(centroid_coords, axis=0).mean()
            else:
                spread = 0.0
        else:
            spread = 0.0
        
        return {
            'coverage': float(coverage),
            'num_lesions': int(num_lesions),
            'avg_lesion_size': float(avg_lesion_size),
            'max_lesion_size': float(max_lesion_size),
            'total_lesion_area': int(total_lesion_area),
            'spatial_spread': float(spread),
        }
    
    def estimate_stage(self, mask_features: Dict) -> Tuple[str, float]:
        """
        Estimate cancer stage based on mask features.
        
        Args:
            mask_features: Features extracted from mask analysis
        
        Returns:
            Tuple of (stage_name, confidence)
        """
        coverage = mask_features['coverage']
        num_lesions = mask_features['num_lesions']
        spread = mask_features['spatial_spread']
        
        # Determine stage based on coverage and other features
        if coverage < self.STAGE_THRESHOLDS['normal']:
            stage = 'Normal / Benign'
            confidence = 0.85
        elif coverage < self.STAGE_THRESHOLDS['stage_0']:
            stage = 'Stage 0 (Carcinoma in situ)'
            confidence = 0.75
        elif coverage < self.STAGE_THRESHOLDS['stage_1']:
            stage = 'Stage I (Early invasive)'
            confidence = 0.70
        elif coverage < self.STAGE_THRESHOLDS['stage_2']:
            if num_lesions > 3 or spread > 100:
                stage = 'Stage IIA-IIB'
            else:
                stage = 'Stage IB-IIA'
            confidence = 0.65
        elif coverage < self.STAGE_THRESHOLDS['stage_3']:
            stage = 'Stage IIIA-IIIB (Locally advanced)'
            confidence = 0.60
        else:
            stage = 'Stage IIIC-IV (Advanced/Metastatic)'
            confidence = 0.55
        
        # Adjust confidence based on lesion characteristics
        if num_lesions == 0:
            confidence = 0.95
        elif num_lesions > 5:
            confidence *= 0.9  # Lower confidence with many lesions
        
        return stage, confidence
    
    def generate_clinical_report(
        self,
        mask: np.ndarray,
        mask_features: Dict,
        stage: str,
        confidence: float,
        xai_data: Dict = None,
        patient_info: Dict = None
    ) -> Dict:
        """
        Generate comprehensive clinical report.
        
        Args:
            mask: Segmentation mask
            mask_features: Features from mask analysis
            stage: Estimated cancer stage
            confidence: Confidence score
            xai_data: XAI explanation data (optional)
            patient_info: Patient information (optional)
        
        Returns:
            Dictionary with complete report
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate findings description
        findings = self._generate_findings(mask_features, stage)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(stage)
        
        # Risk assessment
        risk_level = self._assess_risk(stage, mask_features)
        
        report = {
            'report_id': f"BR_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'timestamp': timestamp,
            'patient_info': patient_info or {},
            'analysis': {
                'stage': stage,
                'confidence': f"{confidence:.1%}",
                'risk_level': risk_level,
            },
            'quantitative_metrics': {
                'tissue_coverage': f"{mask_features['coverage']:.2%}",
                'num_lesions': mask_features['num_lesions'],
                'avg_lesion_size': f"{mask_features['avg_lesion_size']:.1f} pixels",
                'max_lesion_size': f"{mask_features['max_lesion_size']:.1f} pixels",
                'spatial_distribution': f"{mask_features['spatial_spread']:.1f}",
            },
            'findings': findings,
            'recommendations': recommendations,
            'ai_explanation': xai_data.get('description', '') if xai_data else '',
            'disclaimer': (
                "This is an AI-assisted analysis and should be reviewed by a qualified pathologist. "
                "Final diagnosis must be made by a medical professional based on complete clinical context."
            )
        }
        
        return report
    
    def _generate_findings(self, mask_features: Dict, stage: str) -> List[str]:
        """Generate clinical findings based on analysis."""
        findings = []
        
        coverage = mask_features['coverage']
        num_lesions = mask_features['num_lesions']
        
        # Coverage findings
        if coverage < 0.05:
            findings.append("Minimal to no abnormal tissue detected")
        elif coverage < 0.20:
            findings.append(f"Limited abnormal tissue identified ({coverage:.1%} of sample)")
        elif coverage < 0.50:
            findings.append(f"Moderate abnormal tissue presence ({coverage:.1%} of sample)")
        else:
            findings.append(f"Extensive abnormal tissue identified ({coverage:.1%} of sample)")
        
        # Lesion count findings
        if num_lesions == 0:
            findings.append("No distinct lesions identified")
        elif num_lesions == 1:
            findings.append("Single focal lesion detected")
        elif num_lesions <= 3:
            findings.append(f"Multiple focal lesions ({num_lesions}) detected")
        else:
            findings.append(f"Multifocal disease with {num_lesions} distinct lesions")
        
        # Spread findings
        spread = mask_features['spatial_spread']
        if spread > 150:
            findings.append("Wide spatial distribution suggesting potential diffuse involvement")
        elif spread > 75:
            findings.append("Moderate spatial distribution of abnormal tissue")
        
        return findings
    
    def _generate_recommendations(self, stage: str) -> List[str]:
        """Generate clinical recommendations based on stage."""
        recommendations = []
        
        if 'Normal' in stage or 'Benign' in stage:
            recommendations.append("Routine follow-up as per clinical guidelines")
            recommendations.append("Consider repeat screening based on risk factors")
        elif 'Stage 0' in stage:
            recommendations.append("Confirmatory biopsy recommended")
            recommendations.append("Consider sentinel lymph node evaluation")
            recommendations.append("Discuss treatment options including surgery")
        elif 'Stage I' in stage:
            recommendations.append("Multidisciplinary tumor board review recommended")
            recommendations.append("Staging workup with imaging studies")
            recommendations.append("Evaluate for systemic therapy options")
        elif 'Stage II' in stage:
            recommendations.append("Comprehensive staging with CT/MRI imaging")
            recommendations.append("Oncology referral for systemic therapy planning")
            recommendations.append("Consider neoadjuvant therapy options")
        else:  # Stage III-IV
            recommendations.append("Urgent oncology consultation required")
            recommendations.append("Full metastatic workup indicated")
            recommendations.append("Discuss palliative vs. curative treatment approach")
            recommendations.append("Consider clinical trial eligibility")
        
        return recommendations
    
    def _assess_risk(self, stage: str, mask_features: Dict) -> str:
        """Assess overall risk level."""
        if 'Normal' in stage or 'Benign' in stage:
            return "Low"
        elif 'Stage 0' in stage or 'Stage I' in stage:
            return "Moderate"
        elif 'Stage II' in stage:
            return "Moderate-High"
        else:
            return "High"
    
    def export_report_json(self, report: Dict, filepath: str):
        """Export report as JSON file."""
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
    
    def export_report_text(self, report: Dict, filepath: str):
        """Export report as formatted text file."""
        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"BREAST CANCER HISTOPATHOLOGY ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Report ID: {report['report_id']}\n")
            f.write(f"Generated: {report['timestamp']}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("ANALYSIS RESULTS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Stage: {report['analysis']['stage']}\n")
            f.write(f"Confidence: {report['analysis']['confidence']}\n")
            f.write(f"Risk Level: {report['analysis']['risk_level']}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("QUANTITATIVE METRICS\n")
            f.write("-" * 80 + "\n")
            for key, value in report['quantitative_metrics'].items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            f.write("\n")
            
            f.write("-" * 80 + "\n")
            f.write("FINDINGS\n")
            f.write("-" * 80 + "\n")
            for i, finding in enumerate(report['findings'], 1):
                f.write(f"{i}. {finding}\n")
            f.write("\n")
            
            f.write("-" * 80 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 80 + "\n")
            for i, rec in enumerate(report['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
            f.write("\n")
            
            if report['ai_explanation']:
                f.write("-" * 80 + "\n")
                f.write("AI EXPLANATION\n")
                f.write("-" * 80 + "\n")
                f.write(f"{report['ai_explanation']}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("DISCLAIMER\n")
            f.write("-" * 80 + "\n")
            f.write(f"{report['disclaimer']}\n")
            f.write("=" * 80 + "\n")
