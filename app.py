import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import time

# Import pipeline modules
from pipeline.data_ingestion import DataIngestionEngine
from pipeline.core_analysis import CoreAnalysisEngine
from pipeline.ml_analysis import MLAnalysisEngine
from pipeline.sentiment_analysis import SentimentAnalysisEngine
from pipeline.report_generator import ReportGenerator

# Configure Streamlit page
st.set_page_config(page_title="Risk Analysis Pipeline",
                   page_icon="Intelligent Asset Risk Analysis Pipeline📊",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Initialize session state
if 'pipeline_results' not in st.session_state:
    st.session_state.pipeline_results = None
if 'execution_time' not in st.session_state:
    st.session_state.execution_time = None


def main():
    st.title("Intelligent Asset Risk Analysis Pipeline")
    st.markdown("---")

    # Sidebar for pipeline controls
    st.sidebar.header("Pipeline Controls")

    # Fixed portfolio size (all 20 stocks)
    portfolio_size = 20
    st.sidebar.info(f"📊 Portfolio Size: {portfolio_size} stocks")

    # Pipeline execution button
    execute_pipeline = st.sidebar.button("🚀 Execute Full Pipeline",
                                         type="primary")

    # Stage-by-stage execution
    st.sidebar.subheader("Stage-by-Stage Execution")
    stage1_btn = st.sidebar.button("Stage 1: Data Ingestion")
    stage2_btn = st.sidebar.button("Stage 2: Core Analysis")
    stage3_btn = st.sidebar.button("Stage 3: ML Analysis")
    stage4_btn = st.sidebar.button("Stage 4: Sentiment Analysis")
    stage5_btn = st.sidebar.button("Stage 5: Report Generation")

    # Main content area
    if execute_pipeline:
        execute_full_pipeline(portfolio_size)

    # Individual stage execution
    if stage1_btn:
        execute_stage_1(portfolio_size)
    elif stage2_btn:
        execute_stage_2()
    elif stage3_btn:
        execute_stage_3()
    elif stage4_btn:
        execute_stage_4()
    elif stage5_btn:
        execute_stage_5()

    # Display results if available
    if st.session_state.pipeline_results:
        display_pipeline_results()


def execute_full_pipeline(portfolio_size):
    """Execute the complete five-stage pipeline"""
    st.header("🔄 Executing Full Pipeline")

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    start_time = time.time()

    try:
        # Stage 1: Data Ingestion
        status_text.text("Stage 1: Ingesting Bloomberg data...")
        progress_bar.progress(10)

        data_engine = DataIngestionEngine()
        portfolio_data = data_engine.ingest_portfolio_data(portfolio_size)

        progress_bar.progress(25)
        st.success(
            f"✅ Stage 1 Complete: Ingested data for {len(portfolio_data)} assets"
        )

        # Stage 2: Core Analysis
        status_text.text("Stage 2: Running core risk analysis...")
        progress_bar.progress(40)

        analysis_engine = CoreAnalysisEngine()
        analysis_results = analysis_engine.analyze_portfolio(portfolio_data)

        progress_bar.progress(50)
        red_flags = len(
            [a for a in analysis_results if a['risk_rating'] == 'RED'])
        yellow_flags = len(
            [a for a in analysis_results if a['risk_rating'] == 'YELLOW'])
        st.success(
            f"✅ Stage 2 Complete: {red_flags} RED flags, {yellow_flags} YELLOW flags"
        )

        # Stage 3: ML Analysis
        status_text.text(
            "Stage 3: Running ML analysis (Anomaly Detection & Risk Prediction)..."
        )
        progress_bar.progress(60)

        ml_engine = MLAnalysisEngine()
        ml_results = ml_engine.analyze_portfolio_ml(analysis_results)

        progress_bar.progress(70)
        anomaly_count = ml_results['ml_summary']['anomaly_summary'][
            'total_anomalies']
        st.success(
            f"✅ Stage 3 Complete: {anomaly_count} anomalies detected, ML model trained"
        )

        # Stage 4: Sentiment Analysis
        status_text.text("Stage 4: Analyzing sentiment for flagged assets...")
        progress_bar.progress(80)

        sentiment_engine = SentimentAnalysisEngine()
        red_flagged_assets = [
            a for a in analysis_results if a['risk_rating'] == 'RED'
        ]
        sentiment_results = sentiment_engine.analyze_sentiment(
            red_flagged_assets)

        progress_bar.progress(90)
        st.success(
            f"✅ Stage 4 Complete: Sentiment analysis for {len(sentiment_results)} RED-flagged assets"
        )

        # Stage 5: Report Generation
        status_text.text("Stage 5: Generating PDF report with ML insights...")
        progress_bar.progress(95)

        report_generator = ReportGenerator()
        report_files = report_generator.generate_report(
            portfolio_data, analysis_results, sentiment_results, ml_results)

        progress_bar.progress(100)

        end_time = time.time()
        execution_time = end_time - start_time

        st.session_state.pipeline_results = {
            'portfolio_data': portfolio_data,
            'analysis_results': analysis_results,
            'ml_results': ml_results,
            'sentiment_results': sentiment_results,
            'pdf_path': report_files['pdf_path'],
            'portfolio_csv': report_files['portfolio_csv'],
            'analysis_csv': report_files['analysis_csv'],
            'red_flags': red_flags,
            'yellow_flags': yellow_flags
        }
        st.session_state.execution_time = execution_time

        status_text.text("Pipeline execution completed!")
        st.success(f"✅ Pipeline Complete in {execution_time:.2f} seconds!")

        # Offer file downloads
        col1, col2, col3 = st.columns(3)

        with col1:
            if os.path.exists(report_files['pdf_path']):
                with open(report_files['pdf_path'], 'rb') as pdf_file:
                    st.download_button(label="📄 Download PDF Report",
                                       data=pdf_file.read(),
                                       file_name=os.path.basename(
                                           report_files['pdf_path']),
                                       mime="application/pdf")

        with col2:
            if os.path.exists(report_files['portfolio_csv']):
                with open(report_files['portfolio_csv'], 'rb') as csv_file:
                    st.download_button(label="📊 Download Portfolio CSV",
                                       data=csv_file.read(),
                                       file_name=os.path.basename(
                                           report_files['portfolio_csv']),
                                       mime="text/csv")

        with col3:
            if os.path.exists(report_files['analysis_csv']):
                with open(report_files['analysis_csv'], 'rb') as csv_file:
                    st.download_button(label="📈 Download Risk Analysis CSV",
                                       data=csv_file.read(),
                                       file_name=os.path.basename(
                                           report_files['analysis_csv']),
                                       mime="text/csv")

    except Exception as e:
        st.error(f"❌ Pipeline execution failed: {str(e)}")
        status_text.text("Pipeline execution failed!")
        progress_bar.progress(0)


def execute_stage_1(portfolio_size):
    """Execute Stage 1: Data Ingestion"""
    st.header("📥 Stage 1: Data Ingestion")

    with st.spinner("Connecting to Bloomberg API and fetching data..."):
        data_engine = DataIngestionEngine()
        portfolio_data = data_engine.ingest_portfolio_data(portfolio_size)

    st.success(
        f"✅ Successfully ingested data for {len(portfolio_data)} assets")

    # Display sample of ingested data
    df = pd.DataFrame(portfolio_data)
    st.subheader("📊 Portfolio Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # Store in session state
    if 'stage_results' not in st.session_state:
        st.session_state.stage_results = {}
    st.session_state.stage_results['stage1'] = portfolio_data


def execute_stage_2():
    """Execute Stage 2: Core Analysis"""
    st.header("🔍 Stage 2: Core Analysis")

    if 'stage_results' not in st.session_state or 'stage1' not in st.session_state.stage_results:
        st.error("❌ Please run Stage 1 first to ingest portfolio data")
        return

    portfolio_data = st.session_state.stage_results['stage1']

    with st.spinner("Running time-series and rule-based analysis..."):
        analysis_engine = CoreAnalysisEngine()
        analysis_results = analysis_engine.analyze_portfolio(portfolio_data)

    red_flags = len([a for a in analysis_results if a['risk_rating'] == 'RED'])
    yellow_flags = len(
        [a for a in analysis_results if a['risk_rating'] == 'YELLOW'])
    green_flags = len(
        [a for a in analysis_results if a['risk_rating'] == 'GREEN'])

    st.success(
        f"✅ Analysis complete: {red_flags} RED, {yellow_flags} YELLOW, {green_flags} GREEN"
    )

    # Display results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔴 RED Flags", red_flags)
    with col2:
        st.metric("🟡 YELLOW Flags", yellow_flags)
    with col3:
        st.metric("🟢 GREEN Assets", green_flags)

    # Display flagged assets
    flagged_assets = [
        a for a in analysis_results if a['risk_rating'] in ['RED', 'YELLOW']
    ]
    if flagged_assets:
        st.subheader("⚠️ Flagged Assets")
        flagged_df = pd.DataFrame(flagged_assets)
        st.dataframe(flagged_df, use_container_width=True)

    st.session_state.stage_results['stage2'] = analysis_results


def execute_stage_3():
    """Execute Stage 3: ML Analysis"""
    st.header("🤖 Stage 3: ML Analysis")

    if 'stage_results' not in st.session_state or 'stage2' not in st.session_state.stage_results:
        st.error("❌ Please run Stage 2 first to generate analysis results")
        return

    analysis_results = st.session_state.stage_results['stage2']

    with st.spinner(
            "Running ML analysis (Anomaly Detection & Risk Prediction)..."):
        ml_engine = MLAnalysisEngine()
        ml_results = ml_engine.analyze_portfolio_ml(analysis_results)

    anomaly_count = ml_results['ml_summary']['anomaly_summary'][
        'total_anomalies']
    critical_count = ml_results['ml_summary']['anomaly_summary'][
        'critical_anomalies']

    st.success(
        f"✅ ML analysis complete: {anomaly_count} anomalies detected ({critical_count} critical)"
    )

    # Display ML summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Anomalies", anomaly_count)
    with col2:
        st.metric("Critical Anomalies", critical_count)
    with col3:
        if ml_results['risk_prediction'].get('model_trained'):
            st.metric("Model Accuracy",
                      f"{ml_results['risk_prediction']['test_accuracy']}%")
        else:
            st.metric("Model Status", "Not Trained")

    # Display anomaly results
    if ml_results['anomaly_detection']:
        st.subheader("🔍 Anomaly Detection Results")
        anomaly_df = pd.DataFrame(ml_results['anomaly_detection'])
        st.dataframe(anomaly_df[[
            'symbol', 'sector', 'anomaly_score', 'severity', 'is_anomaly'
        ]],
                     use_container_width=True)

    st.session_state.stage_results['stage3'] = ml_results


def execute_stage_4():
    """Execute Stage 4: Sentiment Analysis"""
    st.header("📰 Stage 4: Sentiment Analysis")

    if 'stage_results' not in st.session_state or 'stage2' not in st.session_state.stage_results:
        st.error("❌ Please run Stage 2 first to identify RED-flagged assets")
        return

    analysis_results = st.session_state.stage_results['stage2']
    red_flagged_assets = [
        a for a in analysis_results if a['risk_rating'] == 'RED'
    ]

    if not red_flagged_assets:
        st.info(
            "ℹ️ No RED-flagged assets found. Sentiment analysis not needed.")
        return

    with st.spinner(
            f"Analyzing sentiment for {len(red_flagged_assets)} RED-flagged assets..."
    ):
        sentiment_engine = SentimentAnalysisEngine()
        sentiment_results = sentiment_engine.analyze_sentiment(
            red_flagged_assets)

    st.success(
        f"✅ Sentiment analysis complete for {len(sentiment_results)} assets")

    # Display sentiment results
    if sentiment_results:
        st.subheader("📊 Sentiment Analysis Results")
        sentiment_df = pd.DataFrame(sentiment_results)
        st.dataframe(sentiment_df, use_container_width=True)

        # Sentiment distribution
        avg_sentiment = np.mean(
            [s['sentiment_score'] for s in sentiment_results])
        st.metric("📈 Average Sentiment Score", f"{avg_sentiment:.3f}")

    st.session_state.stage_results['stage4'] = sentiment_results


def execute_stage_5():
    """Execute Stage 5: Report Generation"""
    st.header("📄 Stage 5: Report Generation")

    required_stages = ['stage1', 'stage2', 'stage3']
    if 'stage_results' not in st.session_state or not all(
            stage in st.session_state.stage_results
            for stage in required_stages):
        st.error("❌ Please run Stages 1, 2, and 3 first")
        return

    portfolio_data = st.session_state.stage_results['stage1']
    analysis_results = st.session_state.stage_results['stage2']
    ml_results = st.session_state.stage_results['stage3']
    sentiment_results = st.session_state.stage_results.get('stage4', [])

    with st.spinner("Generating comprehensive PDF report and CSV files..."):
        report_generator = ReportGenerator()
        report_files = report_generator.generate_report(
            portfolio_data, analysis_results, sentiment_results, ml_results)

    st.success("✅ PDF report and CSV files generated successfully!")

    # Display download buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if os.path.exists(report_files['pdf_path']):
            with open(report_files['pdf_path'], 'rb') as pdf_file:
                st.download_button(label="📄 Download PDF Report",
                                   data=pdf_file.read(),
                                   file_name=os.path.basename(
                                       report_files['pdf_path']),
                                   mime="application/pdf",
                                   key="stage5_pdf")

    with col2:
        if os.path.exists(report_files['portfolio_csv']):
            with open(report_files['portfolio_csv'], 'rb') as csv_file:
                st.download_button(label="📊 Download Portfolio CSV",
                                   data=csv_file.read(),
                                   file_name=os.path.basename(
                                       report_files['portfolio_csv']),
                                   mime="text/csv",
                                   key="stage5_portfolio_csv")

    with col3:
        if os.path.exists(report_files['analysis_csv']):
            with open(report_files['analysis_csv'], 'rb') as csv_file:
                st.download_button(label="📈 Download Risk Analysis CSV",
                                   data=csv_file.read(),
                                   file_name=os.path.basename(
                                       report_files['analysis_csv']),
                                   mime="text/csv",
                                   key="stage5_analysis_csv")


def display_pipeline_results():
    """Display comprehensive pipeline results"""
    st.header("📈 Pipeline Results Summary")

    results = st.session_state.pipeline_results

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Assets", len(results['portfolio_data']))
    with col2:
        st.metric("🔴 RED Flags", results['red_flags'])
    with col3:
        st.metric("🟡 YELLOW Flags", results['yellow_flags'])
    with col4:
        st.metric("⏱️ Execution Time",
                  f"{st.session_state.execution_time:.2f}s")

    # Tabs for detailed results
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Portfolio Overview", "⚠️ Risk Analysis", "🤖 ML Analysis",
        "📰 Sentiment", "📄 Report"
    ])

    with tab1:
        st.subheader("Portfolio Data")
        df = pd.DataFrame(results['portfolio_data'])
        st.dataframe(df, use_container_width=True)

    with tab2:
        st.subheader("Risk Analysis Results")
        analysis_df = pd.DataFrame(results['analysis_results'])
        st.dataframe(analysis_df, use_container_width=True)

        # Risk distribution chart
        risk_counts = analysis_df['risk_rating'].value_counts()
        st.bar_chart(risk_counts)

    with tab3:
        st.subheader("Machine Learning Analysis")
        if 'ml_results' in results and results['ml_results']:
            ml_data = results['ml_results']

            # ML Summary
            st.markdown("#### ML Analysis Summary")
            summary = ml_data['ml_summary']

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Anomalies",
                          summary['anomaly_summary']['total_anomalies'])
            with col2:
                st.metric("Critical Anomalies",
                          summary['anomaly_summary']['critical_anomalies'])
            with col3:
                if summary['prediction_summary']['model_trained']:
                    st.metric(
                        "Model Accuracy",
                        f"{summary['prediction_summary']['model_accuracy']}%")
                else:
                    st.metric("Model Status", "Not Trained")

            # Key Insights
            st.markdown("#### Key ML Insights")
            for insight in summary['key_insights']:
                st.info(f"• {insight}")

            # Validation Results
            if 'validation' in ml_data:
                st.markdown("#### ML Validation Results")
                validation = ml_data['validation']

                # Overall status
                if validation['overall_status'] == 'PASS':
                    st.success(
                        f"✅ All validation checks passed ({validation['passed_checks']}/{validation['total_checks']})"
                    )
                elif validation['overall_status'] == 'WARNING':
                    st.warning(
                        f"⚠️ Validation completed with warnings ({validation['passed_checks']}/{validation['total_checks']} passed)"
                    )
                else:
                    st.error(
                        f"❌ Validation failed ({validation['passed_checks']}/{validation['total_checks']} passed)"
                    )

                # Validation checks details
                with st.expander("View Detailed Validation Checks"):
                    for check in validation['validation_checks']:
                        status_icon = "✅" if check[
                            'status'] == 'PASS' else "⚠️" if check[
                                'status'] == 'WARNING' else "❌"
                        st.markdown(
                            f"**{status_icon} {check['check_name']}** - {check['status']}"
                        )

                        if check['metrics']:
                            st.json(check['metrics'])

                        if check['issues']:
                            for issue in check['issues']:
                                st.warning(f"  • {issue}")

                # Display warnings if any
                if validation['warnings']:
                    with st.expander(
                            f"⚠️ {len(validation['warnings'])} Validation Warnings"
                    ):
                        for warning in validation['warnings']:
                            st.text(f"• {warning}")

            # Anomaly Detection Results
            st.markdown("#### Anomaly Detection Results")
            anomaly_df = pd.DataFrame(ml_data['anomaly_detection'])
            st.dataframe(anomaly_df[[
                'symbol', 'sector', 'anomaly_score', 'severity', 'is_anomaly',
                'recommendation'
            ]],
                         use_container_width=True)

            # Risk Predictions
            if ml_data['risk_prediction'].get('model_trained'):
                st.markdown("#### Risk Rating Predictions")
                pred_df = pd.DataFrame(
                    ml_data['risk_prediction']['predictions'])
                st.dataframe(pred_df[[
                    'symbol', 'current_rating', 'predicted_rating',
                    'confidence', 'trend'
                ]],
                             use_container_width=True)

            # Feature Importance
            if ml_data['feature_importance']:
                st.markdown("#### Feature Importance (Risk Drivers)")
                importance_df = pd.DataFrame(ml_data['feature_importance'])
                st.bar_chart(importance_df.set_index('feature')['importance'])
        else:
            st.info("ML analysis not available")

    with tab4:
        st.subheader("Sentiment Analysis")
        if results['sentiment_results']:
            sentiment_df = pd.DataFrame(results['sentiment_results'])
            st.dataframe(sentiment_df, use_container_width=True)
        else:
            st.info("No RED-flagged assets required sentiment analysis")

    with tab5:
        st.subheader("Generated Report & Data Files")

        # Download buttons in columns
        col1, col2, col3 = st.columns(3)

        with col1:
            if 'pdf_path' in results and os.path.exists(results['pdf_path']):
                st.success("📄 PDF Report Ready")
                with open(results['pdf_path'], 'rb') as pdf_file:
                    st.download_button(label="📄 Download PDF Report",
                                       data=pdf_file.read(),
                                       file_name=os.path.basename(
                                           results['pdf_path']),
                                       mime="application/pdf",
                                       key="tab_pdf")
            else:
                st.error("PDF not found")

        with col2:
            if 'portfolio_csv' in results and os.path.exists(
                    results['portfolio_csv']):
                st.success("📊 Portfolio CSV Ready")
                with open(results['portfolio_csv'], 'rb') as csv_file:
                    st.download_button(label="📊 Download Portfolio CSV",
                                       data=csv_file.read(),
                                       file_name=os.path.basename(
                                           results['portfolio_csv']),
                                       mime="text/csv",
                                       key="tab_portfolio_csv")
            else:
                st.info("Portfolio CSV not available")

        with col3:
            if 'analysis_csv' in results and os.path.exists(
                    results['analysis_csv']):
                st.success("📈 Analysis CSV Ready")
                with open(results['analysis_csv'], 'rb') as csv_file:
                    st.download_button(label="📈 Download Risk Analysis CSV",
                                       data=csv_file.read(),
                                       file_name=os.path.basename(
                                           results['analysis_csv']),
                                       mime="text/csv",
                                       key="tab_analysis_csv")
            else:
                st.info("Analysis CSV not available")


if __name__ == "__main__":
    main()
