from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

class SimpleFraudDetectionApp:
    def __init__(self):
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load data"""
        try:
            self.df = pd.read_csv('data/creditcard.csv')
            print("✅ Data loaded successfully")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            self.df = None
    
    def get_dashboard_data(self):
        """Get data for dashboard"""
        if self.df is None:
            return None
        
        # Basic statistics
        total_transactions = len(self.df)
        fraud_count = self.df['Class'].sum()
        legitimate_count = total_transactions - fraud_count
        fraud_rate = (fraud_count / total_transactions) * 100
        
        # Amount statistics
        amount_stats = {
            'mean': self.df['Amount'].mean(),
            'median': self.df['Amount'].median(),
            'std': self.df['Amount'].std(),
            'min': self.df['Amount'].min(),
            'max': self.df['Amount'].max()
        }
        
        return {
            'total_transactions': total_transactions,
            'fraud_count': fraud_count,
            'legitimate_count': legitimate_count,
            'fraud_rate': fraud_rate,
            'amount_stats': amount_stats
        }
    
    def create_simple_charts(self):
        """Create simple charts data"""
        if self.df is None:
            return {}
        
        # Simple fraud distribution data
        fraud_counts = self.df['Class'].value_counts()
        
        # Amount distribution data
        legitimate_amounts = self.df[self.df['Class'] == 0]['Amount'].tolist()
        fraud_amounts = self.df[self.df['Class'] == 1]['Amount'].tolist()
        
        return {
            'fraud_counts': fraud_counts.to_dict(),
            'legitimate_amounts': legitimate_amounts,
            'fraud_amounts': fraud_amounts
        }
    
    def predict_simple(self, transaction_data):
        """Simple prediction based on amount thresholds"""
        try:
            amount = float(transaction_data.get('Amount', 0))
            
            # Simple rule-based prediction
            if amount > 1000:
                prediction = 1  # High risk
                confidence = 0.8
            elif amount > 500:
                prediction = 1  # Medium risk
                confidence = 0.6
            else:
                prediction = 0  # Low risk
                confidence = 0.9
            
            return {
                'prediction': prediction,
                'is_fraud': bool(prediction),
                'confidence': confidence,
                'amount': amount
            }
            
        except Exception as e:
            return {'error': str(e)}

# Initialize the app
fraud_app = SimpleFraudDetectionApp()

@app.route('/')
def dashboard():
    """Main dashboard page"""
    dashboard_data = fraud_app.get_dashboard_data()
    charts = fraud_app.create_simple_charts()
    
    return render_template('simple_dashboard.html', 
                         data=dashboard_data, 
                         charts=charts)

@app.route('/data')
def data_view():
    """Data table view"""
    if fraud_app.df is None:
        return "Data not loaded"
    
    # Get sample data for display
    sample_data = fraud_app.df.head(50).to_dict('records')
    columns = fraud_app.df.columns.tolist()
    
    return render_template('simple_data.html', 
                         data=sample_data, 
                         columns=columns,
                         total_rows=len(fraud_app.df))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page"""
    if request.method == 'POST':
        # Get form data
        transaction_data = {}
        for key in request.form:
            if key != 'csrf_token':
                transaction_data[key] = request.form[key]
        
        # Make prediction
        result = fraud_app.predict_simple(transaction_data)
        return render_template('simple_predict.html', result=result, form_data=transaction_data)
    
    # Show prediction form
    return render_template('simple_predict.html', 
                         result=None, 
                         form_data=None)

@app.route('/download_data')
def download_data():
    """Download the dataset"""
    if fraud_app.df is None:
        return "Data not available"
    
    # Save to temporary file
    temp_file = 'temp_creditcard_data.csv'
    fraud_app.df.to_csv(temp_file, index=False)
    
    return send_file(temp_file, 
                    as_attachment=True,
                    download_name=f'creditcard_fraud_data_{datetime.now().strftime("%Y%m%d")}.csv')

if __name__ == '__main__':
    print("🚀 Starting Simple Credit Card Fraud Detection Web App...")
    print("🌐 Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)
