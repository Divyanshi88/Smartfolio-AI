# SmartFolio AI - Advanced Stock Portfolio Management

SmartFolio AI is a comprehensive stock portfolio management application built with Streamlit. It provides advanced analytics, predictive modeling, and portfolio optimization tools to help users make informed investment decisions.

## Features

- **Stock Data Analysis**: Visualize and analyze historical stock data
- **Predictive Modeling**: Utilize machine learning models (LSTM, BiLSTM, GRU, Transformer, Informer) to predict future stock prices
- **Portfolio Optimization**: Optimize your investment portfolio based on risk and return metrics
- **Interactive Visualizations**: Explore data through interactive charts and graphs
- **Dark/Light Mode**: Toggle between dark and light themes for comfortable viewing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/smartfolio-ai.git
cd smartfolio-ai
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run main.py
```

## Usage

1. Navigate to the application in your web browser (typically at http://localhost:8501)
2. Use the sidebar to select different features and options
3. Upload your own data or use the built-in stock data
4. Analyze, predict, and optimize your portfolio

## Models

The application includes several time series forecasting models:
- LSTM (Long Short-Term Memory)
- BiLSTM (Bidirectional LSTM)
- GRU (Gated Recurrent Unit)
- Transformer
- Informer

## Deployment

The application can be deployed on Streamlit Cloud for easy access:
1. Push your code to GitHub
2. Connect your GitHub repository to Streamlit Cloud
3. Deploy the application with a single click

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [yfinance](https://github.com/ranaroussi/yfinance) for stock data
- [TensorFlow](https://www.tensorflow.org/) and [PyTorch](https://pytorch.org/) for machine learning capabilities