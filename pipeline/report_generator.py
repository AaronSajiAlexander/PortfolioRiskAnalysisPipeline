import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.widgets.markers import makeMarker
import io
import base64

class ReportGenerator:
    """
    Stage 4: Report Generation Engine
    Generates comprehensive PDF reports with all pipeline findings
    """
    
    def __init__(self):
        self.report_dir = "reports"
        self.charts_dir = "charts"
        self.ensure_directories()
        
        # Report styling
        self.styles = getSampleStyleSheet()
        self.custom_styles = self.create_custom_styles()
    
    def ensure_directories(self):
        """Create necessary directories for reports and charts"""
        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)
    
    def create_custom_styles(self):
        """Create custom paragraph styles for the report"""
        custom_styles = {}
        
        # Title style
        custom_styles['CustomTitle'] = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.darkblue,
            alignment=1  # Center alignment
        )
        
        # Section header style
        custom_styles['SectionHeader'] = ParagraphStyle(
            'SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.darkred,
            borderWidth=1,
            borderColor=colors.gray,
            borderPadding=5
        )
        
        # Subsection header style
        custom_styles['SubsectionHeader'] = ParagraphStyle(
            'SubsectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=8,
            spaceBefore=15,
            textColor=colors.darkblue
        )
        
        return custom_styles
    
    def generate_report(self, portfolio_data: List[Dict], analysis_results: List[Dict], 
                       sentiment_results: List[Dict], ml_results: Dict = None) -> str:
        """
        Generate comprehensive PDF report
        
        Args:
            portfolio_data: Original portfolio data
            analysis_results: Risk analysis results
            sentiment_results: Sentiment analysis results
            ml_results: Machine learning analysis results (optional)
            
        Returns:
            Path to generated PDF report
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pdf_filename = f"portfolio_risk_report_{timestamp}.pdf"
        pdf_path = os.path.join(self.report_dir, pdf_filename)
        
        # Create PDF document
        doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)
        
        # Build report content
        story = []
        
        # Title page
        story.extend(self.create_title_page(portfolio_data, analysis_results))
        
        # Executive summary
        story.extend(self.create_executive_summary(portfolio_data, analysis_results, sentiment_results))
        
        # Portfolio overview
        story.extend(self.create_portfolio_overview(portfolio_data, analysis_results))
        
        # Risk analysis section
        story.extend(self.create_risk_analysis_section(analysis_results))
        
        # ML analysis section
        if ml_results:
            story.extend(self.create_ml_analysis_section(ml_results))
        
        # Sentiment analysis section
        if sentiment_results:
            story.extend(self.create_sentiment_analysis_section(sentiment_results))
        
        # Detailed asset analysis
        story.extend(self.create_detailed_analysis_section(analysis_results, sentiment_results))
        
        # Recommendations
        story.extend(self.create_recommendations_section(analysis_results, sentiment_results))
        
        # Appendix
        story.extend(self.create_appendix(portfolio_data, analysis_results))
        
        # Export test data to CSV files
        portfolio_csv, analysis_csv = self.export_test_data_to_csv(portfolio_data, analysis_results, timestamp)
        
        # Test Data Section (for further analysis)
        story.extend(self.create_test_data_section(portfolio_data, analysis_results, portfolio_csv, analysis_csv))
        
        # Build PDF
        doc.build(story)
        
        return {
            'pdf_path': pdf_path,
            'portfolio_csv': portfolio_csv,
            'analysis_csv': analysis_csv
        }
    
    def create_title_page(self, portfolio_data: List[Dict], analysis_results: List[Dict]) -> List:
        """Create report title page"""
        story = []
        
        # Main title
        title = Paragraph("Portfolio Risk Analysis Report", self.custom_styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 30))
        
        # Report metadata
        report_date = datetime.now().strftime('%B %d, %Y')
        metadata_text = f"""
        <para align=center>
        <b>Report Date:</b> {report_date}<br/>
        <b>Portfolio Size:</b> {len(portfolio_data)} Assets<br/>
        <b>Analysis Period:</b> 52 Weeks (Weekly Data)<br/>
        <b>Data Frequency:</b> Weekly Adjusted Prices<br/>
        <b>Generated by:</b> Automated Risk Analysis Pipeline
        </para>
        """
        story.append(Paragraph(metadata_text, self.styles['Normal']))
        story.append(Spacer(1, 50))
        
        # Key statistics
        red_count = len([a for a in analysis_results if a['risk_rating'] == 'RED'])
        yellow_count = len([a for a in analysis_results if a['risk_rating'] == 'YELLOW'])
        green_count = len([a for a in analysis_results if a['risk_rating'] == 'GREEN'])
        
        stats_text = f"""
        <para align=center fontSize=14>
        <b>Risk Assessment Summary</b><br/><br/>
        <font color=red><b>HIGH RISK (RED):</b> {red_count} Assets</font><br/>
        <font color=orange><b>MEDIUM RISK (YELLOW):</b> {yellow_count} Assets</font><br/>
        <font color=green><b>LOW RISK (GREEN):</b> {green_count} Assets</font>
        </para>
        """
        story.append(Paragraph(stats_text, self.styles['Normal']))
        
        story.append(PageBreak())
        return story
    
    def create_executive_summary(self, portfolio_data: List[Dict], 
                               analysis_results: List[Dict], sentiment_results: List[Dict]) -> List:
        """Create executive summary section"""
        story = []
        
        story.append(Paragraph("Executive Summary", self.custom_styles['SectionHeader']))
        
        # Calculate key metrics
        total_assets = len(portfolio_data)
        total_market_cap = sum(asset['market_cap'] for asset in portfolio_data)
        
        red_assets = [a for a in analysis_results if a['risk_rating'] == 'RED']
        yellow_assets = [a for a in analysis_results if a['risk_rating'] == 'YELLOW']
        
        # Calculate metrics with safety checks for empty arrays
        avg_volatility = np.mean([a['volatility'] for a in analysis_results]) if analysis_results else 0
        max_drawdown_portfolio = min([a['max_drawdown'] for a in analysis_results]) if analysis_results else 0
        
        # Executive summary text
        summary_text = f"""
        This report presents a comprehensive risk analysis of a portfolio containing {total_assets} assets 
        with a combined market capitalization of ${total_market_cap/1e9:.1f} billion. Analysis is based on 
        weekly adjusted price data covering approximately 52 weeks of trading history.
        
        <b>Key Findings:</b>
        • {len(red_assets)} assets ({len(red_assets)/total_assets*100:.1f}%) are rated as HIGH RISK (RED)
        • {len(yellow_assets)} assets ({len(yellow_assets)/total_assets*100:.1f}%) are rated as MEDIUM RISK (YELLOW)
        • Average portfolio volatility (annualized): {avg_volatility*100:.1f}%
        • Maximum drawdown observed: {max_drawdown_portfolio*100:.1f}%
        
        <b>Risk Concentration:</b>
        The analysis reveals significant risk concentration in {len(red_assets + yellow_assets)} assets 
        requiring immediate attention and potential portfolio rebalancing.
        """
        
        if sentiment_results:
            avg_sentiment = np.mean([s['sentiment_score'] for s in sentiment_results]) if sentiment_results else 0
            negative_sentiment_count = len([s for s in sentiment_results if s['sentiment_label'] == 'NEGATIVE'])
            
            summary_text += f"""
            
            <b>Sentiment Analysis:</b>
            • {negative_sentiment_count} RED-flagged assets show negative market sentiment
            • Average sentiment score for flagged assets: {avg_sentiment:.3f}
            • News coverage indicates heightened market concern for these positions
            """
        
        story.append(Paragraph(summary_text, self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        return story
    
    def create_portfolio_overview(self, portfolio_data: List[Dict], analysis_results: List[Dict]) -> List:
        """Create portfolio overview section"""
        story = []
        
        story.append(Paragraph("Portfolio Overview", self.custom_styles['SectionHeader']))
        
        # Create sector allocation table
        sector_data = {}
        for asset in portfolio_data:
            sector = asset['sector']
            if sector not in sector_data:
                sector_data[sector] = {'count': 0, 'market_cap': 0}
            sector_data[sector]['count'] += 1
            sector_data[sector]['market_cap'] += asset['market_cap']
        
        # Sector allocation table
        story.append(Paragraph("Sector Allocation", self.custom_styles['SubsectionHeader']))
        
        sector_table_data = [['Sector', 'Assets', 'Market Cap ($B)', 'Percentage']]
        total_market_cap = sum(data['market_cap'] for data in sector_data.values())
        
        for sector, data in sorted(sector_data.items(), key=lambda x: x[1]['market_cap'], reverse=True):
            percentage = data['market_cap'] / total_market_cap * 100
            sector_table_data.append([
                sector,
                str(data['count']),
                f"{data['market_cap']/1e9:.2f}",
                f"{percentage:.1f}%"
            ])
        
        sector_table = Table(sector_table_data)
        sector_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(sector_table)
        story.append(Spacer(1, 20))
        
        return story
    
    def create_risk_analysis_section(self, analysis_results: List[Dict]) -> List:
        """Create risk analysis section"""
        story = []
        
        story.append(Paragraph("Risk Analysis Results", self.custom_styles['SectionHeader']))
        
        # Risk distribution summary
        risk_counts = {'RED': 0, 'YELLOW': 0, 'GREEN': 0}
        for result in analysis_results:
            risk_counts[result['risk_rating']] += 1
        
        risk_text = f"""
        <b>Risk Distribution:</b><br/>
        • HIGH RISK (RED): {risk_counts['RED']} assets<br/>
        • MEDIUM RISK (YELLOW): {risk_counts['YELLOW']} assets<br/>
        • LOW RISK (GREEN): {risk_counts['GREEN']} assets<br/>
        """
        story.append(Paragraph(risk_text, self.styles['Normal']))
        story.append(Spacer(1, 15))
        
        # High risk assets table
        high_risk_assets = [a for a in analysis_results if a['risk_rating'] in ['RED', 'YELLOW']]
        
        if high_risk_assets:
            story.append(Paragraph("High Risk Assets", self.custom_styles['SubsectionHeader']))
            
            risk_table_data = [['Symbol', 'Sector', 'Risk Rating', 'Volatility\n(Annualized)', 'Max Drawdown', 'Risk Score']]
            
            for asset in sorted(high_risk_assets, key=lambda x: x['risk_score'], reverse=True):
                risk_table_data.append([
                    asset['symbol'],
                    asset['sector'],
                    asset['risk_rating'],
                    f"{asset['volatility']*100:.1f}%",
                    f"{asset['max_drawdown']*100:.1f}%",
                    str(asset['risk_score'])
                ])
            
            risk_table = Table(risk_table_data)
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            # Color code risk ratings
            for i, asset in enumerate(high_risk_assets, 1):
                if asset['risk_rating'] == 'RED':
                    risk_table.setStyle(TableStyle([('BACKGROUND', (2, i), (2, i), colors.lightcoral)]))
                elif asset['risk_rating'] == 'YELLOW':
                    risk_table.setStyle(TableStyle([('BACKGROUND', (2, i), (2, i), colors.lightyellow)]))
            
            story.append(risk_table)
        
        story.append(Spacer(1, 20))
        return story
    
    def create_sentiment_analysis_section(self, sentiment_results: List[Dict]) -> List:
        """Create sentiment analysis section"""
        story = []
        
        story.append(Paragraph("Sentiment Analysis", self.custom_styles['SectionHeader']))
        
        if not sentiment_results:
            story.append(Paragraph("No RED-flagged assets required sentiment analysis.", self.styles['Normal']))
            return story
        
        # Sentiment summary
        avg_sentiment = np.mean([s['sentiment_score'] for s in sentiment_results])
        negative_count = len([s for s in sentiment_results if s['sentiment_label'] == 'NEGATIVE'])
        
        sentiment_text = f"""
        Sentiment analysis was conducted on {len(sentiment_results)} RED-flagged assets using 
        financial news from the past 12 months.
        
        <b>Key Findings:</b><br/>
        • Average sentiment score: {avg_sentiment:.3f}<br/>
        • Assets with negative sentiment: {negative_count}/{len(sentiment_results)}<br/>
        • Total news articles analyzed: {sum(s['news_count'] for s in sentiment_results)}<br/>
        """
        story.append(Paragraph(sentiment_text, self.styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Sentiment results table
        story.append(Paragraph("Sentiment Analysis Results", self.custom_styles['SubsectionHeader']))
        
        sentiment_table_data = [['Symbol', 'Sentiment Score', 'Label', 'News Count', 'Trend', 'Key Themes']]
        
        for result in sorted(sentiment_results, key=lambda x: x['sentiment_score']):
            key_themes = ', '.join(result['key_themes'][:3]) if result['key_themes'] else 'None'
            sentiment_table_data.append([
                result['symbol'],
                f"{result['sentiment_score']:.3f}",
                result['sentiment_label'],
                str(result['news_count']),
                result['sentiment_trend'],
                key_themes
            ])
        
        sentiment_table = Table(sentiment_table_data)
        sentiment_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(sentiment_table)
        story.append(Spacer(1, 20))
        
        return story
    
    def create_ml_analysis_section(self, ml_results: Dict) -> List:
        """Create machine learning analysis section"""
        story = []
        
        story.append(Paragraph("Machine Learning Analysis", self.custom_styles['SectionHeader']))
        
        ml_summary = ml_results['ml_summary']
        
        # ML Overview
        ml_text = f"""
        Advanced machine learning techniques were applied to identify anomalies and predict future risk ratings.
        
        <b>Anomaly Detection Summary:</b><br/>
        • Total Anomalies Detected: {ml_summary['anomaly_summary']['total_anomalies']}<br/>
        • Critical Anomalies: {ml_summary['anomaly_summary']['critical_anomalies']}<br/>
        • High Risk Anomalies: {ml_summary['anomaly_summary']['high_anomalies']}<br/>
        • Anomaly Rate: {ml_summary['anomaly_summary']['anomaly_rate']}%<br/>
        """
        
        if ml_summary['prediction_summary']['model_trained']:
            ml_text += f"""
            
            <b>Risk Prediction Model:</b><br/>
            • Model Accuracy: {ml_summary['prediction_summary']['model_accuracy']}%<br/>
            • Rating Changes Predicted: {ml_summary['prediction_summary']['rating_changes_predicted']}<br/>
            • Assets Predicted to Deteriorate: {ml_summary['prediction_summary']['deteriorating_assets']}<br/>
            • Assets Predicted to Improve: {ml_summary['prediction_summary']['improving_assets']}<br/>
            """
        
        story.append(Paragraph(ml_text, self.styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Key ML Insights
        if ml_summary['key_insights']:
            story.append(Paragraph("Key Machine Learning Insights", self.custom_styles['SubsectionHeader']))
            insights_text = "<br/>".join([f"• {insight}" for insight in ml_summary['key_insights']])
            story.append(Paragraph(insights_text, self.styles['Normal']))
            story.append(Spacer(1, 15))
        
        # Anomaly Detection Results Table
        anomaly_results = ml_results['anomaly_detection']
        critical_anomalies = [a for a in anomaly_results if a['severity'] in ['CRITICAL', 'HIGH']]
        
        if critical_anomalies:
            story.append(Paragraph("Critical Anomalies Detected", self.custom_styles['SubsectionHeader']))
            
            anomaly_table_data = [['Symbol', 'Sector', 'Anomaly Score', 'Severity', 'Recommendation']]
            
            for anomaly in sorted(critical_anomalies, key=lambda x: x['anomaly_score'], reverse=True)[:10]:
                anomaly_table_data.append([
                    anomaly['symbol'],
                    anomaly['sector'],
                    f"{anomaly['anomaly_score']:.1f}",
                    anomaly['severity'],
                    anomaly['recommendation'][:40] + '...' if len(anomaly['recommendation']) > 40 else anomaly['recommendation']
                ])
            
            anomaly_table = Table(anomaly_table_data, colWidths=[60, 80, 70, 60, 180])
            anomaly_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 8)
            ]))
            
            story.append(anomaly_table)
            story.append(Spacer(1, 15))
        
        # Risk Predictions Table
        if ml_results['risk_prediction'].get('model_trained'):
            predictions = ml_results['risk_prediction']['predictions']
            rating_changes = [p for p in predictions if p['rating_change']]
            
            if rating_changes:
                story.append(Paragraph("Predicted Risk Rating Changes", self.custom_styles['SubsectionHeader']))
                
                pred_table_data = [['Symbol', 'Current Rating', 'Predicted Rating', 'Confidence', 'Trend']]
                
                for pred in sorted(rating_changes, key=lambda x: x['confidence'], reverse=True)[:10]:
                    pred_table_data.append([
                        pred['symbol'],
                        pred['current_rating'],
                        pred['predicted_rating'],
                        f"{pred['confidence']:.1f}%",
                        pred['trend']
                    ])
                
                pred_table = Table(pred_table_data)
                pred_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(pred_table)
                story.append(Spacer(1, 15))
        
        # Feature Importance
        if ml_results['feature_importance']:
            story.append(Paragraph("Top Risk Factors (Feature Importance)", self.custom_styles['SubsectionHeader']))
            
            top_features = ml_results['feature_importance'][:5]
            features_text = "<br/>".join([
                f"{i+1}. {f['feature']}: {f['importance']:.1f}% importance" 
                for i, f in enumerate(top_features)
            ])
            story.append(Paragraph(features_text, self.styles['Normal']))
            story.append(Spacer(1, 15))
        
        # Validation Results
        if 'validation' in ml_results:
            story.append(Paragraph("ML Validation Results", self.custom_styles['SubsectionHeader']))
            validation = ml_results['validation']
            
            # Overall validation status
            status_color = 'green' if validation['overall_status'] == 'PASS' else 'orange' if validation['overall_status'] == 'WARNING' else 'red'
            validation_text = f"""
            <b>Overall Validation Status:</b> <font color="{status_color}">{validation['overall_status']}</font><br/>
            <b>Checks Passed:</b> {validation['passed_checks']} / {validation['total_checks']}<br/>
            """
            
            # Add validation check summaries
            for check in validation['validation_checks']:
                check_status = '✓' if check['status'] == 'PASS' else '⚠' if check['status'] == 'WARNING' else '✗'
                validation_text += f"<br/><b>{check_status} {check['check_name']}:</b> {check['status']}"
                
                if check['metrics']:
                    metrics_str = ', '.join([f"{k}: {v}" for k, v in list(check['metrics'].items())[:3]])
                    validation_text += f"<br/>  {metrics_str}"
            
            # Add warnings if any
            if validation['warnings']:
                validation_text += f"<br/><br/><b>Validation Warnings ({len(validation['warnings'])}):</b><br/>"
                for i, warning in enumerate(validation['warnings'][:5], 1):
                    validation_text += f"{i}. {warning}<br/>"
                if len(validation['warnings']) > 5:
                    validation_text += f"... and {len(validation['warnings']) - 5} more warnings<br/>"
            
            story.append(Paragraph(validation_text, self.styles['Normal']))
            story.append(Spacer(1, 15))
        
        story.append(Spacer(1, 20))
        return story
    
    def create_detailed_analysis_section(self, analysis_results: List[Dict], 
                                       sentiment_results: List[Dict]) -> List:
        """Create detailed analysis section for top risk assets"""
        story = []
        
        story.append(Paragraph("Detailed Asset Analysis", self.custom_styles['SectionHeader']))
        
        # Focus on top 5 highest risk assets
        high_risk_assets = sorted(
            [a for a in analysis_results if a['risk_rating'] in ['RED', 'YELLOW']],
            key=lambda x: x['risk_score'],
            reverse=True
        )[:5]
        
        sentiment_dict = {s['symbol']: s for s in sentiment_results}
        
        for asset in high_risk_assets:
            story.append(Paragraph(f"Asset: {asset['symbol']}", self.custom_styles['SubsectionHeader']))
            
            # Basic information
            basic_info = f"""
            <b>Sector:</b> {asset['sector']}<br/>
            <b>Current Price:</b> ${asset['current_price']:.2f}<br/>
            <b>Market Cap:</b> ${asset['market_cap']/1e9:.2f}B<br/>
            <b>Risk Rating:</b> {asset['risk_rating']}<br/>
            """
            story.append(Paragraph(basic_info, self.styles['Normal']))
            
            # Risk metrics
            risk_metrics = f"""
            <b>Risk Metrics:</b><br/>
            • Volatility: {asset['volatility']*100:.1f}%<br/>
            • Maximum Drawdown: {asset['max_drawdown']*100:.1f}%<br/>
            • Beta: {asset['beta']:.2f}<br/>
            • Sharpe Ratio: {asset['sharpe_ratio']:.2f}<br/>
            • RSI: {asset['rsi']:.1f}<br/>
            """
            story.append(Paragraph(risk_metrics, self.styles['Normal']))
            
            # Performance metrics
            performance = f"""
            <b>Performance:</b><br/>
            • 1-Month Return: {asset['price_change_1m']*100:.1f}%<br/>
            • 3-Month Return: {asset['price_change_3m']*100:.1f}%<br/>
            • 6-Month Return: {asset['price_change_6m']*100:.1f}%<br/>
            """
            story.append(Paragraph(performance, self.styles['Normal']))
            
            # Risk flags
            risk_flags = [flag for flag, value in asset['risk_flags'].items() if value]
            if risk_flags:
                flags_text = f"""
                <b>Risk Flags:</b><br/>
                • {', '.join(flag.replace('_', ' ').title() for flag in risk_flags)}<br/>
                """
                story.append(Paragraph(flags_text, self.styles['Normal']))
            
            # Sentiment information if available
            if asset['symbol'] in sentiment_dict:
                sentiment_info = sentiment_dict[asset['symbol']]
                sentiment_text = f"""
                <b>Market Sentiment:</b><br/>
                • Sentiment Score: {sentiment_info['sentiment_score']:.3f} ({sentiment_info['sentiment_label']})<br/>
                • News Articles: {sentiment_info['news_count']}<br/>
                • Trend: {sentiment_info['sentiment_trend']}<br/>
                """
                story.append(Paragraph(sentiment_text, self.styles['Normal']))
            
            story.append(Spacer(1, 15))
        
        return story
    
    def create_recommendations_section(self, analysis_results: List[Dict], 
                                     sentiment_results: List[Dict]) -> List:
        """Create recommendations section"""
        story = []
        
        story.append(Paragraph("Recommendations", self.custom_styles['SectionHeader']))
        
        red_assets = [a for a in analysis_results if a['risk_rating'] == 'RED']
        yellow_assets = [a for a in analysis_results if a['risk_rating'] == 'YELLOW']
        
        recommendations_text = f"""
        Based on the comprehensive risk analysis, the following recommendations are provided:
        
        <b>Immediate Actions (RED-flagged assets):</b><br/>
        1. Consider reducing position sizes for {len(red_assets)} high-risk assets<br/>
        2. Implement stop-loss orders to limit further downside exposure<br/>
        3. Review fundamental analysis for potential divestiture candidates<br/>
        4. Monitor daily price movements and news flow closely<br/>
        
        <b>Medium-term Actions (YELLOW-flagged assets):</b><br/>
        1. Conduct deeper due diligence on {len(yellow_assets)} medium-risk assets<br/>
        2. Consider hedging strategies for positions with high volatility<br/>
        3. Review correlation with overall portfolio risk<br/>
        4. Set up enhanced monitoring and alerts<br/>
        
        <b>Portfolio-level Recommendations:</b><br/>
        1. Diversify across sectors to reduce concentration risk<br/>
        2. Consider alternative investments to reduce correlation<br/>
        3. Implement systematic risk management framework<br/>
        4. Schedule monthly portfolio risk reviews<br/>
        """
        
        if sentiment_results:
            negative_sentiment_assets = [s for s in sentiment_results if s['sentiment_label'] == 'NEGATIVE']
            recommendations_text += f"""
            
            <b>Sentiment-based Actions:</b><br/>
            1. Monitor news flow for {len(negative_sentiment_assets)} assets with negative sentiment<br/>
            2. Consider contrarian opportunities if fundamentals remain strong<br/>
            3. Assess impact of market sentiment on price movements<br/>
            4. Review analyst coverage and institutional positioning<br/>
            """
        
        story.append(Paragraph(recommendations_text, self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        return story
    
    def create_appendix(self, portfolio_data: List[Dict], analysis_results: List[Dict]) -> List:
        """Create appendix with methodology and data details"""
        story = []
        
        story.append(PageBreak())
        story.append(Paragraph("Appendix", self.custom_styles['SectionHeader']))
        
        # Methodology
        story.append(Paragraph("Methodology", self.custom_styles['SubsectionHeader']))
        
        methodology_text = """
        This risk analysis employs a five-stage pipeline methodology using weekly data:
        
        <b>Stage 1 - Data Ingestion:</b>
        • Weekly adjusted price data (~52 weeks of history)
        • Weekly trading volume information
        • Market capitalization data
        • Sector classifications
        • Data source: Alpha Vantage API (weekly adjusted series)
        
        <b>Stage 2 - Core Analysis:</b>
        • Volatility calculation (annualized using √52 factor for weekly data)
        • Maximum drawdown analysis on weekly cumulative returns
        • Beta coefficient estimation
        • Risk-adjusted return metrics (Sharpe ratio, annualized)
        • Technical indicators (RSI with 14-week period)
        • Price momentum analysis (4/13/26-week lookback periods)
        • Volume trend analysis (4/12-week windows)
        • Rule-based risk flagging system calibrated for weekly data
        
        <b>Stage 3 - ML Analysis:</b>
        • Anomaly detection using Isolation Forest algorithm
        • Risk prediction using Random Forest Classifier
        • Feature importance analysis
        • ML validation and quality checks
        
        <b>Stage 4 - Sentiment Analysis:</b>
        • News article collection for RED-flagged assets only
        • Natural language processing for sentiment scoring (TextBlob)
        • Trend analysis and confidence metrics
        • Key theme extraction from financial news
        
        <b>Stage 5 - Report Generation:</b>
        • Comprehensive risk assessment compilation
        • Visual data presentation with weekly time axes
        • Actionable recommendations
        • Professional PDF report output with CSV exports
        """
        
        story.append(Paragraph(methodology_text, self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Risk thresholds
        story.append(Paragraph("Risk Assessment Thresholds", self.custom_styles['SubsectionHeader']))
        
        thresholds_text = """
        <b>RED Flag Thresholds (calibrated for weekly data):</b>
        • Volatility > 35% (annualized from weekly returns using √52)
        • Maximum drawdown < -20%
        • Volume decline > 50% (comparing 4-week vs 12-week averages)
        • 1-month price decline > 15%
        
        <b>YELLOW Flag Thresholds (calibrated for weekly data):</b>
        • Volatility > 22% (annualized from weekly returns using √52)
        • Maximum drawdown < -10%
        • Volume decline > 30%
        • Multiple warning indicators present
        
        <b>Sentiment Thresholds:</b>
        • Negative sentiment: Score < -0.3
        • Positive sentiment: Score > 0.3
        • Neutral sentiment: -0.3 ≤ Score ≤ 0.3
        """
        
        story.append(Paragraph(thresholds_text, self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Report generation details
        report_details = f"""
        <b>Report Generation Details:</b><br/>
        • Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        • Analysis Period: 12 months<br/>
        • Total Assets Analyzed: {len(portfolio_data)}<br/>
        • Pipeline Version: 1.0<br/>
        """
        
        story.append(Paragraph(report_details, self.styles['Normal']))
        
        return story
    
    def export_test_data_to_csv(self, portfolio_data: List[Dict], analysis_results: List[Dict], timestamp: str):
        """Export complete portfolio and risk analysis data to CSV files"""
        
        # Export Portfolio Data to CSV
        portfolio_df = pd.DataFrame(portfolio_data)
        portfolio_csv_path = os.path.join(self.report_dir, f"portfolio_data_{timestamp}.csv")
        
        # Select and order columns for portfolio CSV
        portfolio_columns = ['symbol', 'company_name', 'sector', 'current_price', 'market_cap', 
                           'pe_ratio', 'dividend_yield', 'exchange', 'currency']
        portfolio_export = portfolio_df[portfolio_columns].copy()
        portfolio_export = portfolio_export.sort_values('market_cap', ascending=False)
        portfolio_export.to_csv(portfolio_csv_path, index=False)
        
        # Export Risk Analysis Data to CSV
        analysis_df = pd.DataFrame(analysis_results)
        analysis_csv_path = os.path.join(self.report_dir, f"risk_analysis_{timestamp}.csv")
        
        # Select and order columns for risk analysis CSV
        analysis_columns = ['symbol', 'sector', 'risk_rating', 'risk_score', 'volatility', 
                          'max_drawdown', 'volume_decline', 'beta', 'sharpe_ratio', 'rsi',
                          'price_change_1m', 'price_change_3m', 'price_change_6m']
        analysis_export = analysis_df[analysis_columns].copy()
        analysis_export = analysis_export.sort_values('risk_score', ascending=False)
        analysis_export.to_csv(analysis_csv_path, index=False)
        
        return portfolio_csv_path, analysis_csv_path
    
    def create_test_data_section(self, portfolio_data: List[Dict], analysis_results: List[Dict], 
                                 portfolio_csv: str, analysis_csv: str) -> List:
        """Create test data section with CSV references and summary tables for further analysis"""
        story = []
        
        story.append(PageBreak())
        story.append(Paragraph("Test Data - Portfolio Details", self.custom_styles['SectionHeader']))
        
        # Introduction text with CSV file references
        intro_text = f"""
        Complete portfolio and risk analysis data has been exported to CSV files for further analysis:
        <br/><br/>
        <b>Portfolio Data CSV:</b> {os.path.basename(portfolio_csv)}<br/>
        <b>Risk Analysis CSV:</b> {os.path.basename(analysis_csv)}<br/>
        <br/>
        The tables below provide summary information for quick reference.
        """
        story.append(Paragraph(intro_text, self.styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Performance Metrics Table
        story.append(Paragraph("Performance Metrics Data", self.custom_styles['SubsectionHeader']))
        
        performance_table_data = [['Symbol', '1M Return', '3M Return', '6M Return', 'Vol Decline', 'Sharpe Ratio']]
        
        for result in sorted(analysis_results, key=lambda x: x['symbol']):
            performance_table_data.append([
                result['symbol'],
                f"{result['price_change_1m']*100:.1f}%",
                f"{result['price_change_3m']*100:.1f}%",
                f"{result['price_change_6m']*100:.1f}%",
                f"{result['volume_decline']*100:.1f}%",
                f"{result['sharpe_ratio']:.2f}"
            ])
        
        performance_table = Table(performance_table_data)
        performance_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8)
        ]))
        
        story.append(performance_table)
        story.append(Spacer(1, 20))
        
        # Risk Flags Details Table
        story.append(Paragraph("Risk Flags Details", self.custom_styles['SubsectionHeader']))
        
        risk_flags_table_data = [['Symbol', 'High Vol', 'Ext. DD', 'Vol Collapse', 'Severe Dec', 'Ext. Dec', 'Poor Sharpe', 'Mom. Break']]
        
        for result in sorted(analysis_results, key=lambda x: x['risk_score'], reverse=True):
            flags = result['risk_flags']
            risk_flags_table_data.append([
                result['symbol'],
                '✓' if flags.get('high_volatility') else '✗',
                '✓' if flags.get('extreme_drawdown') else '✗',
                '✓' if flags.get('volume_collapse') else '✗',
                '✓' if flags.get('severe_decline') else '✗',
                '✓' if flags.get('extended_decline') else '✗',
                '✓' if flags.get('poor_risk_return') else '✗',
                '✓' if flags.get('momentum_breakdown') else '✗'
            ])
        
        risk_flags_table = Table(risk_flags_table_data)
        risk_flags_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 7)
        ]))
        
        story.append(risk_flags_table)
        story.append(Spacer(1, 20))
        
        # Data summary note
        summary_note = f"""
        <b>Data Summary:</b><br/>
        • Total Assets: {len(portfolio_data)}<br/>
        • Data Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        • Note: This test data is provided for further analysis, validation, and detailed review purposes.<br/>
        • All metrics are calculated from 252 trading days of historical data.
        """
        story.append(Paragraph(summary_note, self.styles['Normal']))
        
        return story
