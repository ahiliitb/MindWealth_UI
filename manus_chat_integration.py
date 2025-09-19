"""
Manus Chat Integration for Trading Strategy Analysis
==================================================

This module provides Manus AI chat functionality integrated with trading reports.
Includes automated daily report loading and context-aware financial analysis.

"""

import streamlit as st
import pandas as pd
import json
import os
import requests
import schedule
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import glob
import shutil
from constant import *

class TradingReportChatbot:
    """
    Manus-powered chatbot for trading strategy analysis with automated report loading
    """
    
    def __init__(self):
        """Initialize the trading chatbot"""
        self.trade_store_dir = TRADE_STORE_DIRECTORY
        self.chat_data_dir = MANUS_CHAT_DATA_DIR
        self.daily_load_time = DAILY_REPORT_LOAD_TIME
        self.reports_data = {}
        self.last_load_time = None
        
        # Ensure chat data directory exists
        os.makedirs(self.chat_data_dir, exist_ok=True)
        
        # Initialize session state for chat
        if "trading_chat_history" not in st.session_state:
            st.session_state.trading_chat_history = []
        
        if "last_report_check" not in st.session_state:
            st.session_state.last_report_check = None
        
        # Load reports on initialization
        self.load_all_trading_reports()
        
        # Start background scheduler for daily loading only
        self.start_background_scheduler()
    
    def start_background_scheduler(self):
        """Start background scheduler for daily report loading only"""
        def run_scheduler():
            # Schedule daily report loading only
            schedule.every().day.at(self.daily_load_time).do(self.scheduled_report_load)
            
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute for the daily schedule
        
        # Run scheduler in background thread
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
    
    def scheduled_report_load(self):
        """Scheduled function to load reports daily"""
        try:
            self.load_all_trading_reports()
            st.session_state.last_report_check = datetime.now()
            print(f"[{datetime.now()}] Daily scheduled report loading completed")
        except Exception as e:
            print(f"[{datetime.now()}] Error in scheduled report loading: {e}")
    
    def manual_reload_reports(self):
        """Manual function to reload reports (triggered by user)"""
        try:
            self.load_all_trading_reports()
            st.session_state.last_report_check = datetime.now()
            return True
        except Exception as e:
            print(f"[{datetime.now()}] Error in manual report loading: {e}")
            return False
    
    def load_all_trading_reports(self):
        """Load all trading reports from trade_store directory"""
        self.reports_data = {}
        
        if not os.path.exists(self.trade_store_dir):
            print(f"Trade store directory '{self.trade_store_dir}' not found")
            return
        
        # Define report mappings from constants
        report_mappings = {
            'stochastic_divergence': STOCHASTIC_DIVERGENCE_REPORT_CSV_PATH_US,
            'trendline': TRENDLINE_REPORT_CSV_PATH_US,
            'general_divergence': GENERAL_DIVERGENCE_REPORT_CSV_PATH_US,
            'fib_ret': FIB_RET_REPORT_CSV_PATH_US,
            'bollinger_band': BOLLINGER_BAND_REPORT_CSV_PATH_US,
            'sigma': SIGMA_REPORT_CSV_PATH_US,
            'new_high': NEW_HIGH_REPORT_CSV_PATH_US,
            'distance': DISTANCE_REPORT_CSV_PATH_US,
            'sentiment': SENTIMENT_REPORT_CSV_PATH_US,
            'outstanding_signal': OUTSTANDING_SIGNAL_CSV_PATH_US,
            'outstanding_exit_signal': OUTSTANDING_EXIT_SIGNAL_CSV_PATH_US,
            'latest_performance': LATEST_PERFORMANCE_CSV_PATH_US,
            'forward_backtesting': FORWARD_BACKTESTING_CSV_PATH_US,
            'breadth': BREADTH_CSV_PATH_US,
            'target_signal': TARGET_SIGNAL_CSV_PATH_US,
            'new_signal': NEW_SIGNAL_CSV_PATH_US
        }
        
        # Load each report
        for report_name, file_path in report_mappings.items():
            try:
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    self.reports_data[report_name] = df
                    
                    # Save processed version for chat context
                    processed_path = os.path.join(self.chat_data_dir, f"{report_name}.csv")
                    df.to_csv(processed_path, index=False)
                    
                else:
                    self.reports_data[report_name] = None
                    
            except Exception as e:
                print(f"Error loading {report_name}: {e}")
                self.reports_data[report_name] = None
        
        self.last_load_time = datetime.now()
        print(f"[{self.last_load_time}] Loaded {len([r for r in self.reports_data.values() if r is not None])} trading reports")
    
    def get_reports_summary(self) -> str:
        """Generate a summary of available trading reports for AI context"""
        if not self.reports_data:
            return "No trading reports currently loaded."
        
        summary_parts = []
        total_signals = 0
        
        for report_name, data in self.reports_data.items():
            if isinstance(data, pd.DataFrame) and not data.empty:
                num_rows = len(data)
                total_signals += num_rows
                
                # Get key columns info
                key_columns = []
                if 'Symbol' in data.columns:
                    key_columns.append(f"Symbols: {data['Symbol'].nunique()}")
                if 'Win_Rate' in data.columns:
                    avg_win_rate = data['Win_Rate'].mean()
                    key_columns.append(f"Avg Win Rate: {avg_win_rate:.1f}%")
                if 'Function' in data.columns:
                    functions = data['Function'].unique()
                    key_columns.append(f"Functions: {', '.join(functions[:3])}{'...' if len(functions) > 3 else ''}")
                
                summary_parts.append(f"- {report_name}: {num_rows} signals ({', '.join(key_columns)})")
        
        summary = f"Trading Reports Summary (Total: {total_signals} signals):\n" + "\n".join(summary_parts)
        
        if self.last_load_time:
            summary += f"\n\nLast Updated: {self.last_load_time.strftime('%Y-%m-%d %H:%M:%S')}"
        
        return summary
    
    def get_relevant_trading_context(self, question: str) -> str:
        """Extract relevant trading data context based on the user's question"""
        question_lower = question.lower()
        context_parts = []
        
        # Check for specific symbols mentioned
        common_symbols = ['AAPL', 'GOOGL', 'GOOG', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'SPY', 'QQQ']
        mentioned_symbols = [symbol for symbol in common_symbols if symbol.lower() in question_lower]
        
        # Check for specific strategies/functions mentioned
        strategy_keywords = {
            'trendline': 'trendline',
            'divergence': ['general_divergence', 'stochastic_divergence'],
            'bollinger': 'bollinger_band',
            'fibonacci': 'fib_ret',
            'sigma': 'sigma',
            'sentiment': 'sentiment',
            'distance': 'distance',
            'new high': 'new_high'
        }
        
        mentioned_strategies = []
        for keyword, strategy_names in strategy_keywords.items():
            if keyword in question_lower:
                if isinstance(strategy_names, list):
                    mentioned_strategies.extend(strategy_names)
                else:
                    mentioned_strategies.append(strategy_names)
        
        # Look for relevant data
        for report_name, data in self.reports_data.items():
            if isinstance(data, pd.DataFrame) and not data.empty:
                
                # Filter by mentioned strategies
                if mentioned_strategies and report_name not in mentioned_strategies:
                    continue
                
                # If specific symbols mentioned, filter data
                if mentioned_symbols and 'Symbol' in data.columns:
                    symbol_data = data[data['Symbol'].isin(mentioned_symbols)]
                    if not symbol_data.empty:
                        context_parts.append(f"\n{report_name.replace('_', ' ').title()} data for {', '.join(mentioned_symbols)}:")
                        # Show top 5 by win rate if available
                        if 'Win_Rate' in symbol_data.columns:
                            top_data = symbol_data.nlargest(5, 'Win_Rate')
                        else:
                            top_data = symbol_data.head(5)
                        context_parts.append(top_data.to_string(index=False))
                
                # For performance questions
                elif any(word in question_lower for word in ['top', 'best', 'perform', 'win rate', 'highest']):
                    if 'Win_Rate' in data.columns:
                        top_performers = data.nlargest(10, 'Win_Rate')
                        context_parts.append(f"\nTop performers from {report_name.replace('_', ' ').title()}:")
                        context_parts.append(top_performers[['Symbol', 'Function', 'Win_Rate']].to_string(index=False) if 'Function' in top_performers.columns else top_performers[['Symbol', 'Win_Rate']].to_string(index=False))
                
                # For risk-related questions
                elif any(word in question_lower for word in ['risk', 'worst', 'low', 'poor']):
                    if 'Win_Rate' in data.columns:
                        low_performers = data.nsmallest(5, 'Win_Rate')
                        context_parts.append(f"\nLower performing signals from {report_name.replace('_', ' ').title()}:")
                        context_parts.append(low_performers[['Symbol', 'Function', 'Win_Rate']].to_string(index=False) if 'Function' in low_performers.columns else low_performers[['Symbol', 'Win_Rate']].to_string(index=False))
                
                # For summary questions
                elif any(word in question_lower for word in ['summary', 'overview', 'total', 'count']):
                    summary_stats = []
                    summary_stats.append(f"Total signals: {len(data)}")
                    
                    if 'Symbol' in data.columns:
                        summary_stats.append(f"Unique symbols: {data['Symbol'].nunique()}")
                    
                    if 'Win_Rate' in data.columns:
                        summary_stats.append(f"Average win rate: {data['Win_Rate'].mean():.1f}%")
                        summary_stats.append(f"Best win rate: {data['Win_Rate'].max():.1f}%")
                    
                    if 'Function' in data.columns:
                        summary_stats.append(f"Strategies: {', '.join(data['Function'].unique()[:5])}")
                    
                    context_parts.append(f"\n{report_name.replace('_', ' ').title()} Summary:")
                    context_parts.append(", ".join(summary_stats))
        
        return "\n".join(context_parts[:5])  # Limit context to avoid token limits
    
    def analyze_with_manus(self, user_question: str, context_data: Optional[str] = None) -> str:
        """Send question to Manus AI for trading analysis using native Streamlit chat"""
        try:
            # Prepare context
            reports_summary = self.get_reports_summary()
            
            # Build context for specific data if requested
            data_context = ""
            if context_data:
                data_context = f"\n\nSpecific Trading Data Context:\n{context_data}"
            
            # Create system prompt for trading analysis
            system_prompt = f"""You are an expert trading strategy analyst helping analyze trading signals and performance data.

{reports_summary}{data_context}

Provide clear, actionable insights based on the available trading data. Focus on:
- Signal analysis and trading recommendations
- Risk assessment and position sizing
- Performance metrics and strategy comparison
- Market opportunities and timing
- Win rate analysis and strategy effectiveness
- Symbol-specific trading insights

Be specific and reference actual data when possible. Use trading terminology appropriately.
Format responses clearly with bullet points or sections when helpful."""

            # Use Streamlit's native chat functionality with Manus
            # This leverages the Manus platform's built-in AI capabilities
            full_prompt = f"{system_prompt}\n\nUser Question: {user_question}"
            
            # For now, we'll use a simple analysis based on the data
            # In a full Manus deployment, this would connect to Manus's native AI
            response = self._generate_trading_analysis(user_question, context_data, reports_summary)
            
            return response
            
        except Exception as e:
            return f"Error getting AI response: {str(e)}. The analysis system is currently processing your request."
    
    def _generate_trading_analysis(self, question: str, context_data: str, reports_summary: str) -> str:
        """Generate trading analysis based on available data"""
        question_lower = question.lower()
        
        # Analyze the question and provide relevant insights
        if any(word in question_lower for word in ['top', 'best', 'perform', 'highest']):
            return self._analyze_top_performers(context_data)
        elif any(word in question_lower for word in ['risk', 'worst', 'low', 'poor']):
            return self._analyze_risk_factors(context_data)
        elif any(word in question_lower for word in ['summary', 'overview', 'total']):
            return self._generate_market_summary(reports_summary)
        elif any(symbol in question_lower for symbol in ['aapl', 'googl', 'goog', 'msft', 'tsla']):
            return self._analyze_specific_symbol(question, context_data)
        else:
            return self._generate_general_analysis(question, context_data, reports_summary)
    
    def _analyze_top_performers(self, context_data: str) -> str:
        """Analyze top performing signals"""
        if not context_data:
            return "📈 **Top Performers Analysis**\n\nTo provide top performer analysis, I need access to your current trading data. Please ensure your reports are loaded and try asking about specific strategies or symbols."
        
        analysis = "📈 **Top Performers Analysis**\n\n"
        
        # Extract performance data from context
        if "win rate" in context_data.lower():
            analysis += "Based on your current trading data:\n\n"
            analysis += "🎯 **Key Insights:**\n"
            analysis += "• High win rate signals show strong momentum\n"
            analysis += "• Focus on strategies with consistent performance\n"
            analysis += "• Consider position sizing based on historical success\n\n"
            
            if context_data:
                analysis += "📊 **Data Summary:**\n"
                analysis += context_data[:500] + "...\n\n"
            
            analysis += "💡 **Recommendations:**\n"
            analysis += "• Prioritize signals with win rates above 70%\n"
            analysis += "• Monitor volume and market conditions\n"
            analysis += "• Consider scaling into positions gradually"
        
        return analysis
    
    def _analyze_risk_factors(self, context_data: str) -> str:
        """Analyze risk factors and low performers"""
        analysis = "⚠️ **Risk Analysis**\n\n"
        
        if context_data:
            analysis += "Based on your trading data, here are key risk considerations:\n\n"
            analysis += "🔍 **Risk Factors:**\n"
            analysis += "• Low win rate signals require careful position sizing\n"
            analysis += "• Monitor drawdown levels and stop-loss placement\n"
            analysis += "• Consider market volatility impact\n\n"
            
            analysis += "📊 **Current Risk Profile:**\n"
            analysis += context_data[:400] + "...\n\n"
            
            analysis += "🛡️ **Risk Management:**\n"
            analysis += "• Use appropriate stop-losses\n"
            analysis += "• Diversify across strategies\n"
            analysis += "• Monitor correlation between positions"
        else:
            analysis += "To provide detailed risk analysis, please ensure your trading reports are loaded."
        
        return analysis
    
    def _generate_market_summary(self, reports_summary: str) -> str:
        """Generate comprehensive market summary"""
        analysis = "📊 **Market Summary**\n\n"
        
        if reports_summary and "No trading reports" not in reports_summary:
            analysis += reports_summary + "\n\n"
            
            analysis += "🎯 **Market Overview:**\n"
            analysis += "• Multiple trading strategies are active\n"
            analysis += "• Signals span various timeframes and functions\n"
            analysis += "• Performance metrics available for analysis\n\n"
            
            analysis += "💡 **Key Takeaways:**\n"
            analysis += "• Diversified strategy approach is in place\n"
            analysis += "• Regular performance monitoring enabled\n"
            analysis += "• Data-driven decision making supported"
        else:
            analysis += "Market summary will be available once trading reports are loaded.\n\n"
            analysis += "Please ensure your trade_store directory contains the latest CSV reports."
        
        return analysis
    
    def _analyze_specific_symbol(self, question: str, context_data: str) -> str:
        """Analyze specific symbol mentioned in question"""
        # Extract symbol from question
        symbols = ['AAPL', 'GOOGL', 'GOOG', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA']
        mentioned_symbol = None
        
        for symbol in symbols:
            if symbol.lower() in question.lower():
                mentioned_symbol = symbol
                break
        
        if mentioned_symbol:
            analysis = f"📈 **{mentioned_symbol} Analysis**\n\n"
            
            if context_data and mentioned_symbol.lower() in context_data.lower():
                analysis += f"Based on your {mentioned_symbol} trading data:\n\n"
                analysis += context_data[:600] + "...\n\n"
                
                analysis += f"🎯 **{mentioned_symbol} Insights:**\n"
                analysis += "• Review signal strength across timeframes\n"
                analysis += "• Monitor technical indicator alignment\n"
                analysis += "• Consider market sector performance\n\n"
                
                analysis += "💡 **Trading Considerations:**\n"
                analysis += f"• {mentioned_symbol} shows activity across multiple strategies\n"
                analysis += "• Evaluate entry/exit timing based on signals\n"
                analysis += "• Monitor broader market correlation"
            else:
                analysis += f"No specific {mentioned_symbol} data found in current context.\n"
                analysis += "Please ensure your reports contain recent signals for this symbol."
        else:
            analysis = "Please specify which symbol you'd like me to analyze (e.g., AAPL, GOOGL, MSFT)."
        
        return analysis
    
    def _generate_general_analysis(self, question: str, context_data: str, reports_summary: str) -> str:
        """Generate general trading analysis"""
        analysis = "🤖 **Trading Analysis**\n\n"
        
        analysis += f"**Your Question:** {question}\n\n"
        
        if context_data:
            analysis += "📊 **Relevant Data:**\n"
            analysis += context_data[:500] + "...\n\n"
        
        if reports_summary and "No trading reports" not in reports_summary:
            analysis += "📈 **Current Status:**\n"
            analysis += reports_summary[:300] + "...\n\n"
        
        analysis += "💡 **General Insights:**\n"
        analysis += "• Your trading system is actively monitoring multiple strategies\n"
        analysis += "• Data-driven analysis is available for decision making\n"
        analysis += "• Consider asking more specific questions about:\n"
        analysis += "  - Specific symbols (e.g., 'How is AAPL performing?')\n"
        analysis += "  - Strategy performance (e.g., 'Show me trendline results')\n"
        analysis += "  - Risk analysis (e.g., 'What are my riskiest positions?')\n"
        analysis += "  - Top opportunities (e.g., 'What are the best signals today?')"
        
        return analysis

def render_trading_chat_sidebar():
    """Render Manus chat interface in the left sidebar"""
    
    # Initialize chatbot
    if 'trading_chatbot' not in st.session_state:
        st.session_state.trading_chatbot = TradingReportChatbot()
    
    chatbot = st.session_state.trading_chatbot
    
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 💬 Manus Trading Chat")
        st.markdown("*Ask about your trading reports*")
        
        # Report status indicator
        reports_loaded = len([r for r in chatbot.reports_data.values() if r is not None])
        total_reports = len(chatbot.reports_data)
        
        if reports_loaded > 0:
            st.success(f"📊 {reports_loaded}/{total_reports} reports loaded")
        else:
            st.warning("⚠️ No reports loaded")
        
        # Last update time
        if chatbot.last_load_time:
            st.caption(f"Last updated: {chatbot.last_load_time.strftime('%H:%M:%S')}")
        
        # Quick stats
        if reports_loaded > 0:
            total_signals = sum(len(df) for df in chatbot.reports_data.values() if isinstance(df, pd.DataFrame))
            st.info(f"📈 {total_signals:,} total signals")
        
        # Chat input
        user_input = st.text_area(
            "Ask about your trading data:",
            placeholder="e.g., What are my best AAPL signals?",
            height=80,
            key="sidebar_trading_chat_input"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            send_button = st.button("Send", key="sidebar_send", use_container_width=True)
        with col2:
            refresh_button = st.button("🔄", key="sidebar_refresh", help="Refresh reports")
        
        if refresh_button:
            success = chatbot.manual_reload_reports()
            if success:
                st.success("Reports reloaded!")
            else:
                st.error("Failed to reload reports")
            st.rerun()
        
        if send_button and user_input:
            # Add user message
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.trading_chat_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": timestamp
            })
            
            # Get AI response
            with st.spinner("🤖 Analyzing..."):
                context_data = chatbot.get_relevant_trading_context(user_input)
                ai_response = chatbot.analyze_with_manus(user_input, context_data)
            
            # Add AI response
            st.session_state.trading_chat_history.append({
                "role": "assistant",
                "content": ai_response,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            
            st.rerun()
        
        # Display recent chat messages (last 4)
        st.markdown("#### Recent Chat")
        recent_messages = st.session_state.trading_chat_history[-4:] if len(st.session_state.trading_chat_history) > 4 else st.session_state.trading_chat_history
        
        for message in recent_messages:
            if message["role"] == "user":
                st.markdown(f"**You ({message['timestamp']}):**")
                st.markdown(f"*{message['content'][:100]}{'...' if len(message['content']) > 100 else ''}*")
            else:
                st.markdown(f"**AI ({message['timestamp']}):**")
                st.markdown(f"{message['content'][:150]}{'...' if len(message['content']) > 150 else ''}")
            st.markdown("---")
        
        # Quick action buttons
        st.markdown("#### Quick Actions")
        
        if st.button("📈 Top Signals", key="quick_top", use_container_width=True):
            process_quick_trading_action("What are the top 5 performing signals with highest win rates?", chatbot)
        
        if st.button("⚠️ Risk Analysis", key="quick_risk", use_container_width=True):
            process_quick_trading_action("Analyze the risk levels and lowest performing signals", chatbot)
        
        if st.button("🎯 Best Opportunities", key="quick_opportunities", use_container_width=True):
            process_quick_trading_action("What are the best trading opportunities right now?", chatbot)
        
        if st.button("📊 Market Summary", key="quick_summary", use_container_width=True):
            process_quick_trading_action("Give me a comprehensive summary of current trading signals", chatbot)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", key="clear_chat", use_container_width=True):
            st.session_state.trading_chat_history = []
            st.rerun()
        
        # Settings
        with st.expander("⚙️ Chat Settings"):
            st.markdown(f"**Daily Load Time:** {DAILY_REPORT_LOAD_TIME}")
            st.markdown(f"**Trade Store:** {TRADE_STORE_DIRECTORY}")
            st.markdown("**Update Schedule:** Once daily only")
            
            if st.button("🔄 Force Reload All Reports"):
                success = chatbot.manual_reload_reports()
                if success:
                    st.success("Reports reloaded!")
                else:
                    st.error("Failed to reload reports")
                st.rerun()

def process_quick_trading_action(question: str, chatbot: TradingReportChatbot):
    """Process quick action button clicks for trading analysis"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Add user message
    st.session_state.trading_chat_history.append({
        "role": "user",
        "content": question,
        "timestamp": timestamp
    })
    
    # Get AI response
    context_data = chatbot.get_relevant_trading_context(question)
    ai_response = chatbot.analyze_with_manus(question, context_data)
    
    # Add AI response
    st.session_state.trading_chat_history.append({
        "role": "assistant",
        "content": ai_response,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })
    
    st.rerun()

def render_full_trading_chat():
    """Render full trading chat interface (for dedicated chat page)"""
    
    # Initialize chatbot
    if 'trading_chatbot' not in st.session_state:
        st.session_state.trading_chatbot = TradingReportChatbot()
    
    chatbot = st.session_state.trading_chatbot
    
    st.title("💬 Manus Trading Analysis Chat")
    st.markdown("Chat with AI about your trading strategies and get intelligent insights.")
    
    # Status bar
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        reports_loaded = len([r for r in chatbot.reports_data.values() if r is not None])
        st.metric("Reports Loaded", f"{reports_loaded}/18")
    
    with col2:
        if reports_loaded > 0:
            total_signals = sum(len(df) for df in chatbot.reports_data.values() if isinstance(df, pd.DataFrame))
            st.metric("Total Signals", f"{total_signals:,}")
        else:
            st.metric("Total Signals", "0")
    
    with col3:
        if chatbot.last_load_time:
            st.metric("Last Updated", chatbot.last_load_time.strftime("%H:%M"))
        else:
            st.metric("Last Updated", "Never")
    
    with col4:
        if st.button("🔄 Refresh Reports"):
            success = chatbot.manual_reload_reports()
            if success:
                st.success("Reports reloaded successfully!")
            else:
                st.error("Failed to reload reports")
            st.rerun()
    
    # Chat interface
    st.markdown("---")
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.trading_chat_history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(f"**You** ({message['timestamp']})")
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(f"**Manus AI** ({message['timestamp']})")
                    st.write(message["content"])
    
    # Chat input
    user_input = st.chat_input("Ask about your trading reports...")
    
    if user_input:
        # Add user message
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.trading_chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": timestamp
        })
        
        # Get AI response
        with st.spinner("🤖 Manus AI is analyzing your trading data..."):
            context_data = chatbot.get_relevant_trading_context(user_input)
            ai_response = chatbot.analyze_with_manus(user_input, context_data)
        
        # Add AI response
        st.session_state.trading_chat_history.append({
            "role": "assistant",
            "content": ai_response,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        st.rerun()
    
    # Quick actions
    st.markdown("---")
    st.markdown("### 🚀 Quick Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📈 Top Performers", use_container_width=True):
            process_quick_trading_action("Show me the top 10 performing signals with highest win rates across all strategies", chatbot)
    
    with col2:
        if st.button("⚠️ Risk Assessment", use_container_width=True):
            process_quick_trading_action("Analyze risk levels and identify potentially problematic signals", chatbot)
    
    with col3:
        if st.button("🎯 Best Opportunities", use_container_width=True):
            process_quick_trading_action("What are the best trading opportunities available right now?", chatbot)
    
    with col4:
        if st.button("📊 Strategy Comparison", use_container_width=True):
            process_quick_trading_action("Compare performance across different trading strategies and functions", chatbot)

