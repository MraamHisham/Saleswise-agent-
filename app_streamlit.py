# ==================== SalesWise AI Agent - Streamlit App ====================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="SalesWise AI Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        background-color: #f5f5f5;
    }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .insight-box {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
<div class="main-header">
    <h1>📊 SalesWise AI Agent</h1>
    <p>Intelligent Sales Analytics with AI-Powered Q&A</p>
</div>
""", unsafe_allow_html=True)

# ==================== LOAD DATA ====================
@st.cache_data
def load_data():
    """Load and preprocess data"""
    df = pd.read_csv(r"D:\Semester 5\Data visooo\project1\train.csv")
    
    # Convert dates
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d/%m/%Y')
    
    # Extract date features
    df['Order Year'] = df['Order Date'].dt.year
    df['Order Month'] = df['Order Date'].dt.month
    df['Order Quarter'] = df['Order Date'].dt.quarter
    df['Order Weekday'] = df['Order Date'].dt.day_name()
    
    # Calculate shipping time
    df['Shipping Time'] = (df['Ship Date'] - df['Order Date']).dt.days
    
    # Create year-month column
    df['Year-Month'] = df['Order Date'].dt.to_period('M').astype(str)
    
    return df

# Load data
with st.spinner("Loading data..."):
    df = load_data()

st.success(f"✅ Loaded {len(df):,} records | 📅 {df['Order Date'].min().date()} to {df['Order Date'].max().date()}")

# ==================== SIDEBAR ====================
st.sidebar.markdown("## 🎛️ Filters")
st.sidebar.markdown("---")

# Year filter
years = sorted(df['Order Year'].unique())
selected_years = st.sidebar.multiselect(
    "Select Year(s)",
    options=years,
    default=years
)

# Category filter
categories = df['Category'].unique().tolist()
selected_categories = st.sidebar.multiselect(
    "Select Category(s)",
    options=categories,
    default=categories
)

# Region filter
regions = df['Region'].unique().tolist()
selected_regions = st.sidebar.multiselect(
    "Select Region(s)",
    options=regions,
    default=regions
)

# Apply filters
filtered_df = df[
    (df['Order Year'].isin(selected_years)) &
    (df['Category'].isin(selected_categories)) &
    (df['Region'].isin(selected_regions))
]

if len(filtered_df) == 0:
    st.warning("No data available with current filters. Please adjust your selection.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**📊 Showing:** {len(filtered_df):,} records")

# ==================== KPI CARDS ====================
st.markdown("## 📈 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_sales = filtered_df['Sales'].sum()
    st.metric("💰 Total Sales", f"${total_sales:,.2f}")

with col2:
    total_orders = filtered_df['Order ID'].nunique()
    st.metric("📦 Total Orders", f"{total_orders:,}")

with col3:
    avg_order = total_sales / total_orders if total_orders > 0 else 0
    st.metric("💵 Average Order", f"${avg_order:,.2f}")

with col4:
    unique_customers = filtered_df['Customer ID'].nunique()
    st.metric("👥 Customers", f"{unique_customers:,}")

# ==================== CHARTS SECTION ====================
st.markdown("---")
st.markdown("## 📊 Visual Analytics")

# Row 1: Category and Region
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🏆 Sales by Category")
    category_sales = filtered_df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    category_sales.plot(kind='bar', color=colors[:len(category_sales)], ax=ax)
    ax.set_title('Total Sales by Category', fontsize=14, fontweight='bold')
    ax.set_ylabel('Sales ($)', fontsize=12)
    ax.set_xlabel('Category', fontsize=12)
    ax.tick_params(axis='x', rotation=0)
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)
    plt.close()

with col2:
    st.markdown("### 🗺️ Sales by Region")
    region_sales = filtered_df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    region_sales.plot(kind='pie', autopct='%1.1f%%', colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], ax=ax)
    ax.set_title('Sales Distribution by Region', fontsize=14, fontweight='bold')
    ax.set_ylabel('')
    st.pyplot(fig)
    plt.close()

# Row 2: Monthly Trend and Heatmap
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📈 Monthly Sales Trend")
    monthly_sales = filtered_df.groupby('Year-Month')['Sales'].sum().reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly_sales['Year-Month'], monthly_sales['Sales'], marker='o', linewidth=2, color='#2E86AB')
    ax.set_title('Monthly Sales Trend', fontsize=14, fontweight='bold')
    ax.set_ylabel('Sales ($)', fontsize=12)
    ax.set_xlabel('Month', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    st.markdown("### 🔥 Sales Heatmap")
    heatmap_data = filtered_df.pivot_table(values='Sales', index='Category', columns='Region', aggfunc='sum')
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd', linewidths=0.5, ax=ax)
    ax.set_title('Sales Heatmap: Category vs Region', fontsize=14, fontweight='bold')
    st.pyplot(fig)
    plt.close()

# Row 3: Seasonality and Forecast
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📅 Seasonality Analysis")
    monthly_avg = filtered_df.groupby('Order Month')['Sales'].mean()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig, ax = plt.subplots(figsize=(10, 5))
    monthly_avg.plot(kind='bar', color='#F18F01', ax=ax)
    ax.set_title('Average Sales by Month', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Sales ($)', fontsize=12)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_xticklabels(month_names, rotation=0)
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)
    plt.close()
    
    best_month = monthly_avg.idxmax()
    worst_month = monthly_avg.idxmin()
    st.info(f"🌟 **Best Month:** {month_names[best_month-1]} (${monthly_avg[best_month]:,.2f})\n\n⚠️ **Worst Month:** {month_names[worst_month-1]} (${monthly_avg[worst_month]:,.2f})")

with col2:
    st.markdown("### 🔮 Sales Forecast")
    
    # Calculate forecast
    monthly_series = filtered_df.groupby('Year-Month')['Sales'].sum()
    if len(monthly_series) >= 3:
        last_3_months = monthly_series.tail(3)
        avg_growth = last_3_months.pct_change().mean()
        last_month_sales = monthly_series.iloc[-1]
        forecast = last_month_sales * (1 + avg_growth)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        months = monthly_series.index.tolist()
        sales_values = monthly_series.values.tolist()
        
        # Add forecast
        months.append("Forecast")
        sales_values.append(forecast)
        
        colors = ['#2E86AB'] * (len(sales_values)-1) + ['#F18F01']
        ax.bar(months, sales_values, color=colors)
        ax.set_title('Sales with Forecast', fontsize=14, fontweight='bold')
        ax.set_ylabel('Sales ($)', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
        plt.close()
        
        st.metric(
            "📊 Next Month Forecast", 
            f"${forecast:,.2f}",
            delta=f"{avg_growth*100:+.1f}%"
        )
    else:
        st.warning("Not enough data for forecast (need at least 3 months)")

# ==================== TOP LISTS ====================
st.markdown("---")
st.markdown("## 🎯 Top Performers")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🏆 Top 10 Products")
    top_products = filtered_df.groupby('Product Name')['Sales'].sum().nlargest(10).reset_index()
    top_products.columns = ['Product', 'Sales']
    top_products['Sales'] = top_products['Sales'].apply(lambda x: f"${x:,.2f}")
    st.dataframe(top_products, use_container_width=True, hide_index=True)

with col2:
    st.markdown("### 👑 Top 10 Customers")
    top_customers = filtered_df.groupby('Customer Name')['Sales'].sum().nlargest(10).reset_index()
    top_customers.columns = ['Customer', 'Total Spent']
    top_customers['Total Spent'] = top_customers['Total Spent'].apply(lambda x: f"${x:,.2f}")
    st.dataframe(top_customers, use_container_width=True, hide_index=True)

# ==================== BUSINESS INSIGHTS ====================
st.markdown("---")
st.markdown("## 💡 Business Insights")

insights = []

# Calculate insights
best_category = category_sales.idxmax() if len(category_sales) > 0 else "N/A"
best_region = region_sales.idxmax() if len(region_sales) > 0 else "N/A"
avg_shipping = filtered_df['Shipping Time'].mean()

insights.append(f"📌 **Best Category:** {best_category} with ${category_sales.max():,.2f} in sales")
insights.append(f"📌 **Top Region:** {best_region} with ${region_sales.max():,.2f} in sales")
insights.append(f"📌 **Shipping Performance:** Average {avg_shipping:.1f} days delivery time")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown(insights[0])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown(insights[1])
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown(insights[2])
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== AI AGENT SECTION ====================
st.markdown("---")
st.markdown("## 🤖 Ask the AI Agent")

st.markdown("Ask me anything about the sales data!")

# Predefined questions
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("💰 Total Sales", use_container_width=True):
        st.session_state['ai_question'] = "What are total sales?"
with col2:
    if st.button("🏆 Best Category", use_container_width=True):
        st.session_state['ai_question'] = "Which is the best category?"
with col3:
    if st.button("🔮 Forecast Next Month", use_container_width=True):
        st.session_state['ai_question'] = "What is the forecast for next month?"

# Question input
question = st.text_input("Or type your own question:", key="ai_input", placeholder="e.g., What is the best month for sales?")

# Process question
if 'ai_question' in st.session_state and st.session_state['ai_question']:
    question = st.session_state['ai_question']
    st.session_state['ai_question'] = ""

if question:
    with st.spinner("Thinking..."):
        question_lower = question.lower()
        
        if 'total sales' in question_lower:
            answer = f"💰 Total sales: ${filtered_df['Sales'].sum():,.2f}"
        
        elif 'best category' in question_lower or 'top category' in question_lower:
            cat_sales = filtered_df.groupby('Category')['Sales'].sum()
            best = cat_sales.idxmax()
            amount = cat_sales.max()
            answer = f"🏆 Best category: **{best}** with ${amount:,.2f} in sales"
        
        elif 'best region' in question_lower:
            reg_sales = filtered_df.groupby('Region')['Sales'].sum()
            best = reg_sales.idxmax()
            amount = reg_sales.max()
            answer = f"🗺️ Best region: **{best}** with ${amount:,.2f} in sales"
        
        elif 'forecast' in question_lower or 'predict' in question_lower:
            monthly_series = filtered_df.groupby('Year-Month')['Sales'].sum()
            if len(monthly_series) >= 3:
                last_3_months = monthly_series.tail(3)
                avg_growth = last_3_months.pct_change().mean()
                last_month_sales = monthly_series.iloc[-1]
                forecast = last_month_sales * (1 + avg_growth)
                answer = f"🔮 Next month forecast: ${forecast:,.2f} (based on {avg_growth*100:.1f}% growth)"
            else:
                answer = "⚠️ Not enough data for forecast"
        
        elif 'best month' in question_lower or 'season' in question_lower:
            monthly_avg = filtered_df.groupby('Order Month')['Sales'].mean()
            best = monthly_avg.idxmax()
            month_names_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            answer = f"📅 Best month: **{month_names_list[best-1]}** with ${monthly_avg[best]:,.2f} average sales"
        
        elif 'shipping' in question_lower:
            avg_shipping = filtered_df['Shipping Time'].mean()
            answer = f"🚚 Average shipping time: **{avg_shipping:.1f} days**"
        
        elif 'customers' in question_lower or 'customer count' in question_lower:
            customers = filtered_df['Customer ID'].nunique()
            answer = f"👥 Total unique customers: **{customers}**"
        
        elif 'help' in question_lower:
            answer = """
            🤖 **I can answer questions about:**
            - Total sales
            - Best category
            - Best region
            - Sales forecast
            - Best month (seasonality)
            - Shipping time
            - Customer count
            - And more!
            
            Try: "What is the forecast?" or "Which is the best month?"
            """
        
        else:
            answer = "🤔 I'm not sure. Try asking about total sales, best category, forecast, or best month. Type 'help' for options."
        
        # Display answer
        st.markdown("---")
        st.markdown(f"### 🤖 AI Agent Response")
        st.info(answer)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>📊 SalesWise AI Agent | Powered by Streamlit | Real-time Sales Analytics</p>
    <p>Filter data using the sidebar to explore different segments!</p>
</div>
""", unsafe_allow_html=True)
