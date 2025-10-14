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
from pipeline.sentiment_analysis import SentimentAnalysisEngine
from pipeline.report_generator import ReportGenerator

# Configure Streamlit page
st.set_page_config(
    page_title="Portfolio Risk Analysis Pipeline",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'pipeline_results' not in st.session_state:
    st.session_state.pipeline_results = None
if 'execution_time' not in st.session_state:
    st.session_state.execution_time = None

def main():
    st.title("📊 Portfolio Risk Analysis Pipeline")
    st.markdown("---")
    
    # Sidebar for pipeline controls
    st.sidebar.header("Pipeline Controls")
    
    # Portfolio selection
    st.sidebar.subheader("Portfolio Configuration")
    portfolio_size = st.sidebar.slider("Portfolio Size", min_value=10, max_value=100, value=25)
    
    # Pipeline execution button
    execute_pipeline = st.sidebar.button("🚀 Execute Full Pipeline", type="primary")
    
    # Stage-by-stage execution
    st.sidebar.subheader("Stage-by-Stage Execution")
    stage1_btn = st.sidebar.button("Stage 1: Data Ingestion")
    stage2_btn = st.sidebar.button("Stage 2: Core Analysis")
    stage3_btn = st.sidebar.button("Stage 3: Sentiment Analysis")
    stage4_btn = st.sidebar.button("Stage 4: Report Generation")
    
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
    
    # Display results if available
    if st.session_state.pipeline_results:
        display_pipeline_results()

def execute_full_pipeline(portfolio_size):
    """Execute the complete four-stage pipeline"""
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
        st.success(f"✅ Stage 1 Complete: Ingested data for {len(portfolio_data)} assets")
        
        # Stage 2: Core Analysis
        status_text.text("Stage 2: Running core risk analysis...")
        progress_bar.progress(40)
        
        analysis_engine = CoreAnalysisEngine()
        analysis_results = analysis_engine.analyze_portfolio(portfolio_data)
        
        progress_bar.progress(60)
        red_flags = len([a for a in analysis_results if a['risk_rating'] == 'RED'])
        yellow_flags = len([a for a in analysis_results if a['risk_rating'] == 'YELLOW'])
        st.success(f"✅ Stage 2 Complete: {red_flags} RED flags, {yellow_flags} YELLOW flags")
        
        # Stage 3: Sentiment Analysis
        status_text.text("Stage 3: Analyzing sentiment for flagged assets...")
        progress_bar.progress(75)
        
        sentiment_engine = SentimentAnalysisEngine()
        red_flagged_assets = [a for a in analysis_results if a['risk_rating'] == 'RED']
        sentiment_results = sentiment_engine.analyze_sentiment(red_flagged_assets)
        
        progress_bar.progress(85)
        st.success(f"✅ Stage 3 Complete: Sentiment analysis for {len(sentiment_results)} RED-flagged assets")
        
        # Stage 4: Report Generation
        status_text.text("Stage 4: Generating PDF report...")
        progress_bar.progress(95)
        
        report_generator = ReportGenerator()
        pdf_path = report_generator.generate_report(
            portfolio_data, analysis_results, sentiment_results
        )
        
        progress_bar.progress(100)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        st.session_state.pipeline_results = {
            'portfolio_data': portfolio_data,
            'analysis_results': analysis_results,
            'sentiment_results': sentiment_results,
            'pdf_path': pdf_path,
            'red_flags': red_flags,
            'yellow_flags': yellow_flags
        }
        st.session_state.execution_time = execution_time
        
        status_text.text("Pipeline execution completed!")
        st.success(f"✅ Pipeline Complete in {execution_time:.2f} seconds!")
        
        # Offer PDF download
        if os.path.exists(pdf_path):
            with open(pdf_path, 'rb') as pdf_file:
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_file.read(),
                    file_name=f"portfolio_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
        
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
    
    st.success(f"✅ Successfully ingested data for {len(portfolio_data)} assets")
    
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
    yellow_flags = len([a for a in analysis_results if a['risk_rating'] == 'YELLOW'])
    green_flags = len([a for a in analysis_results if a['risk_rating'] == 'GREEN'])
    
    st.success(f"✅ Analysis complete: {red_flags} RED, {yellow_flags} YELLOW, {green_flags} GREEN")
    
    # Display results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔴 RED Flags", red_flags)
    with col2:
        st.metric("🟡 YELLOW Flags", yellow_flags)
    with col3:
        st.metric("🟢 GREEN Assets", green_flags)
    
    # Display flagged assets
    flagged_assets = [a for a in analysis_results if a['risk_rating'] in ['RED', 'YELLOW']]
    if flagged_assets:
        st.subheader("⚠️ Flagged Assets")
        flagged_df = pd.DataFrame(flagged_assets)
        st.dataframe(flagged_df, use_container_width=True)
    
    st.session_state.stage_results['stage2'] = analysis_results

def execute_stage_3():
    """Execute Stage 3: Sentiment Analysis"""
    st.header("📰 Stage 3: Sentiment Analysis")
    
    if 'stage_results' not in st.session_state or 'stage2' not in st.session_state.stage_results:
        st.error("❌ Please run Stage 2 first to identify RED-flagged assets")
        return
    
    analysis_results = st.session_state.stage_results['stage2']
    red_flagged_assets = [a for a in analysis_results if a['risk_rating'] == 'RED']
    
    if not red_flagged_assets:
        st.info("ℹ️ No RED-flagged assets found. Sentiment analysis not needed.")
        return
    
    with st.spinner(f"Analyzing sentiment for {len(red_flagged_assets)} RED-flagged assets..."):
        sentiment_engine = SentimentAnalysisEngine()
        sentiment_results = sentiment_engine.analyze_sentiment(red_flagged_assets)
    
    st.success(f"✅ Sentiment analysis complete for {len(sentiment_results)} assets")
    
    # Display sentiment results
    if sentiment_results:
        st.subheader("📊 Sentiment Analysis Results")
        sentiment_df = pd.DataFrame(sentiment_results)
        st.dataframe(sentiment_df, use_container_width=True)
        
        # Sentiment distribution
        avg_sentiment = np.mean([s['sentiment_score'] for s in sentiment_results])
        st.metric("📈 Average Sentiment Score", f"{avg_sentiment:.3f}")
    
    st.session_state.stage_results['stage3'] = sentiment_results

def execute_stage_4():
    """Execute Stage 4: Report Generation"""
    st.header("📄 Stage 4: Report Generation")
    
    required_stages = ['stage1', 'stage2']
    if 'stage_results' not in st.session_state or not all(stage in st.session_state.stage_results for stage in required_stages):
        st.error("❌ Please run Stages 1 and 2 first")
        return
    
    portfolio_data = st.session_state.stage_results['stage1']
    analysis_results = st.session_state.stage_results['stage2']
    sentiment_results = st.session_state.stage_results.get('stage3', [])
    
    with st.spinner("Generating comprehensive PDF report..."):
        report_generator = ReportGenerator()
        pdf_path = report_generator.generate_report(
            portfolio_data, analysis_results, sentiment_results
        )
    
    st.success("✅ PDF report generated successfully!")
    
    if os.path.exists(pdf_path):
        with open(pdf_path, 'rb') as pdf_file:
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_file.read(),
                file_name=f"portfolio_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )

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
        st.metric("⏱️ Execution Time", f"{st.session_state.execution_time:.2f}s")
    
    # Tabs for detailed results
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Portfolio Overview", "⚠️ Risk Analysis", "📰 Sentiment", "📄 Report"])
    
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
        st.subheader("Sentiment Analysis")
        if results['sentiment_results']:
            sentiment_df = pd.DataFrame(results['sentiment_results'])
            st.dataframe(sentiment_df, use_container_width=True)
        else:
            st.info("No RED-flagged assets required sentiment analysis")
    
    with tab4:
        st.subheader("Generated Report")
        if os.path.exists(results['pdf_path']):
            st.success("PDF report ready for download")
            with open(results['pdf_path'], 'rb') as pdf_file:
                st.download_button(
                    label="📄 Download Complete Report",
                    data=pdf_file.read(),
                    file_name=f"portfolio_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
        else:
            st.error("Report file not found")

if __name__ == "__main__":
    main()
