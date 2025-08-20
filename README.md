# 🛒 Olist Customer Satisfaction Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://prsuman1-olist-customer-satisfaction-predi-streamlit-app-1pq9vv.streamlit.app)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

An interactive **Streamlit dashboard** for exploring customer satisfaction predictions using the Olist Brazilian e-commerce dataset. Built with machine learning insights and business intelligence.

## 🌟 Dashboard Features

### 📊 **9 Interactive Pages**
- **🏠 Home**: Project overview and key metrics
- **📊 Data Overview**: Dataset statistics and relationships  
- **🔍 Data Quality**: Missing value analysis and quality metrics
- **📈 Exploratory Analysis**: Interactive visualizations and patterns
- **⚙️ Feature Engineering**: 38+ engineered features documentation
- **🤖 Model Performance**: ML model comparison (80.4% accuracy)
- **💼 Business Insights**: ROI analysis (R$ 2.3M potential impact)
- **🎯 Make Predictions**: Interactive prediction interface
- **📋 Technical Details**: Architecture and deployment guides

### 🎯 **Key Results**
- **Best Model**: XGBoost with **80.4% accuracy**
- **Business Impact**: **R$ 2.3M** potential annual revenue impact
- **Data Coverage**: **99,441 orders** from Brazilian e-commerce
- **Features**: **38+ engineered features** with anti-leakage design

## 🚀 Quick Start

### Local Development
```bash
# Clone the repository
git clone https://github.com/prsuman1/olist-customer-satisfaction-predictor.git
cd olist-customer-satisfaction-predictor

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py
```

### 🐳 Docker Deployment
```bash
# Build and run with Docker
docker build -t olist-dashboard .
docker run -p 8501:8501 olist-dashboard
```

## 📁 Dashboard Structure

```
streamlit_app.py              # Main Streamlit application
streamlit_pages/              # Dashboard pages
├── data_overview.py          # Dataset overview
├── data_quality.py           # Quality analysis  
├── eda.py                    # Exploratory analysis
├── feature_engineering.py    # Feature insights
├── model_performance.py      # ML model comparison
├── business_insights.py      # ROI and business impact
├── prediction.py             # Interactive predictions
└── technical.py              # Technical documentation
```

## 📊 Data Source

Uses the [Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/olistbr/brazilian-ecommerce):
- **99,441 orders** from 2016-2018
- **9 interconnected tables** 
- **Customer reviews, payments, and logistics data**

## 🛠️ Technologies

- **Frontend**: Streamlit, Plotly, HTML/CSS
- **Backend**: Python, Pandas, NumPy
- **ML Models**: XGBoost, Random Forest, Logistic Regression, LightGBM
- **Deployment**: Docker, Streamlit Cloud

## 🚀 Live Demo

🌐 **[View Live Dashboard](https://prsuman1-olist-customer-satisfaction-predi-streamlit-app-1pq9vv.streamlit.app)**

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Olist** for the comprehensive dataset
- **Streamlit** for the amazing dashboard framework  
- **Plotly** for interactive visualizations

---

<div align="center">

### 🌟 **Star this repo if you found it helpful!** 🌟

**Built with ❤️ using Streamlit and modern data science practices**

</div>