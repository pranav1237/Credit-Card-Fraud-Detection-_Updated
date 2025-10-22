import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_analyze_data():
    """Load and analyze the credit card fraud dataset"""
    print("🔍 Loading Credit Card Fraud Dataset...")
    
    # Load the data
    df = pd.read_csv('data/creditcard.csv')
    
    print(f"📊 Dataset Shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    print("\n" + "="*60)
    
    return df

def show_data_preview(df):
    """Show data preview and basic statistics"""
    print("📋 DATA PREVIEW:")
    print(df.head())
    
    print("\n📊 BASIC STATISTICS:")
    print(df.describe())
    
    print("\n🔍 DATA TYPES:")
    print(df.dtypes)
    
    print("\n❓ MISSING VALUES:")
    print(df.isnull().sum())

def analyze_fraud_distribution(df):
    """Analyze fraud vs legitimate transaction distribution"""
    print("\n" + "="*60)
    print("🎯 FRAUD ANALYSIS:")
    
    # Count fraud vs legitimate
    fraud_counts = df['Class'].value_counts()
    fraud_percentage = (fraud_counts[1] / len(df)) * 100
    
    print(f"✅ Legitimate Transactions: {fraud_counts[0]:,} ({100-fraud_percentage:.2f}%)")
    print(f"🚨 Fraudulent Transactions: {fraud_counts[1]:,} ({fraud_percentage:.2f}%)")
    print(f"📈 Total Transactions: {len(df):,}")
    
    return fraud_counts, fraud_percentage

def create_visualizations(df, fraud_counts, fraud_percentage):
    """Create comprehensive visualizations"""
    print("\n🎨 Creating Visualizations...")
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Fraud Distribution Pie Chart
    plt.subplot(3, 3, 1)
    colors = ['#2E8B57', '#DC143C']
    plt.pie(fraud_counts.values, labels=['Legitimate', 'Fraud'], 
            autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('Transaction Distribution\n(Fraud vs Legitimate)', fontsize=14, fontweight='bold')
    
    # 2. Transaction Amount Distribution
    plt.subplot(3, 3, 2)
    plt.hist(df[df['Class'] == 0]['Amount'], bins=50, alpha=0.7, 
             label='Legitimate', color='#2E8B57', density=True)
    plt.hist(df[df['Class'] == 1]['Amount'], bins=50, alpha=0.7, 
             label='Fraud', color='#DC143C', density=True)
    plt.xlabel('Transaction Amount ($)')
    plt.ylabel('Density')
    plt.title('Transaction Amount Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.xlim(0, 500)  # Limit to show distribution better
    
    # 3. Feature V1 Distribution (most important feature)
    plt.subplot(3, 3, 3)
    plt.hist(df[df['Class'] == 0]['V1'], bins=50, alpha=0.7, 
             label='Legitimate', color='#2E8B57', density=True)
    plt.hist(df[df['Class'] == 1]['V1'], bins=50, alpha=0.7, 
             label='Fraud', color='#DC143C', density=True)
    plt.xlabel('V1 Feature Value')
    plt.ylabel('Density')
    plt.title('V1 Feature Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    
    # 4. Transaction Amount by Fraud Status (Box Plot)
    plt.subplot(3, 3, 4)
    df.boxplot(column='Amount', by='Class', ax=plt.gca())
    plt.title('Transaction Amount by Fraud Status', fontsize=14, fontweight='bold')
    plt.suptitle('')  # Remove default title
    plt.xlabel('Fraud Status (0=Legitimate, 1=Fraud)')
    plt.ylabel('Amount ($)')
    
    # 5. Time vs Fraud Pattern
    plt.subplot(3, 3, 5)
    time_fraud = df.groupby(['Time', 'Class']).size().unstack(fill_value=0)
    time_fraud.plot(kind='line', marker='', ax=plt.gca(), alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Number of Transactions')
    plt.title('Time vs Transaction Pattern', fontsize=14, fontweight='bold')
    plt.legend(['Legitimate', 'Fraud'])
    plt.grid(True, alpha=0.3)
    
    # 6. Feature Correlation Heatmap (top features)
    plt.subplot(3, 3, 6)
    # Select top features for correlation
    top_features = ['V1', 'V2', 'V3', 'V4', 'V5', 'Amount', 'Class']
    corr_matrix = df[top_features].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    
    # 7. Amount vs V1 Feature
    plt.subplot(3, 3, 7)
    plt.scatter(df[df['Class'] == 0]['V1'], 
                df[df['Class'] == 0]['Amount'], 
                alpha=0.6, s=20, c='#2E8B57', label='Legitimate')
    plt.scatter(df[df['Class'] == 1]['V1'], 
                df[df['Class'] == 1]['Amount'], 
                alpha=0.8, s=30, c='#DC143C', label='Fraud')
    plt.xlabel('V1 Feature')
    plt.ylabel('Transaction Amount ($)')
    plt.title('V1 Feature vs Transaction Amount', fontsize=14, fontweight='bold')
    plt.legend()
    
    # 8. Feature Importance Preview (V1-V10)
    plt.subplot(3, 3, 8)
    v_features = [f'V{i}' for i in range(1, 11)]
    feature_means = df[v_features].mean()
    plt.barh(range(len(v_features)), feature_means.values, color='#4682B4')
    plt.yticks(range(len(v_features)), v_features)
    plt.xlabel('Mean Value')
    plt.title('Mean Values of V1-V10 Features', fontsize=14, fontweight='bold')
    
    # 9. Amount Distribution by Class
    plt.subplot(3, 3, 9)
    df[df['Class'] == 0]['Amount'].hist(bins=30, alpha=0.7, color='#2E8B57', 
                                         label='Legitimate', density=True)
    df[df['Class'] == 1]['Amount'].hist(bins=30, alpha=0.7, color='#DC143C', 
                                         label='Fraud', density=True)
    plt.xlabel('Amount ($)')
    plt.ylabel('Density')
    plt.title('Amount Distribution by Class', fontsize=14, fontweight='bold')
    plt.legend()
    plt.xlim(0, 300)
    
    plt.tight_layout()
    plt.savefig('fraud_analysis_visualizations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def train_and_evaluate_model(df):
    """Train a model and show fraud detection metrics"""
    print("\n" + "="*60)
    print("🤖 TRAINING FRAUD DETECTION MODEL...")
    
    # Prepare features (V1-V28 + Amount, exclude Time and Class)
    feature_columns = [f'V{i}' for i in range(1, 29)] + ['Amount']
    
    # Prepare X and y
    X = df[feature_columns]
    y = df['Class']
    
    print(f"📊 Features used: {len(feature_columns)}")
    print(f"📊 Training samples: {len(X)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n📈 MODEL PERFORMANCE METRICS:")
    print(f"🎯 ROC-AUC Score: {auc_score:.4f}")
    
    # Classification report
    print(f"\n📊 CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n🔍 CONFUSION MATRIX:")
    print(f"True Negatives (Correctly identified legitimate): {cm[0,0]}")
    print(f"False Positives (Legitimate marked as fraud): {cm[0,1]}")
    print(f"False Negatives (Fraud missed): {cm[1,0]}")
    print(f"True Positives (Correctly identified fraud): {cm[1,1]}")
    
    # Calculate additional metrics
    precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n🎯 FRAUD DETECTION RATES:")
    print(f"Precision (Accuracy of fraud predictions): {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall (Fraud detection rate): {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score (Balanced measure): {f1_score:.4f} ({f1_score*100:.2f}%)")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔝 TOP 10 MOST IMPORTANT FEATURES:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    # Create ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, color='#DC143C', lw=2, 
             label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Fraud Detection Model', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create feature importance plot
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'], color='#4682B4')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Feature Importances', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, auc_score, precision, recall, f1_score

def main():
    """Main analysis function"""
    print("🚀 CREDIT CARD FRAUD DETECTION ANALYSIS")
    print("="*60)
    
    # Load data
    df = load_and_analyze_data()
    
    # Show data preview
    show_data_preview(df)
    
    # Analyze fraud distribution
    fraud_counts, fraud_percentage = analyze_fraud_distribution(df)
    
    # Create visualizations
    fig = create_visualizations(df, fraud_counts, fraud_percentage)
    
    # Train and evaluate model
    model, auc_score, precision, recall, f1_score = train_and_evaluate_model(df)
    
    # Summary
    print("\n" + "="*60)
    print("📋 ANALYSIS SUMMARY:")
    print(f"📊 Dataset: {df.shape[0]:,} transactions with {df.shape[1]} features")
    print(f"🎯 Fraud Rate: {fraud_percentage:.2f}%")
    print(f"🤖 Model Performance:")
    print(f"   - ROC-AUC: {auc_score:.4f}")
    print(f"   - Precision: {precision:.4f}")
    print(f"   - Recall: {recall:.4f}")
    print(f"   - F1-Score: {f1_score:.4f}")
    print("\n💾 Visualizations saved as:")
    print("   - fraud_analysis_visualizations.png")
    print("   - roc_curve.png")
    print("   - feature_importance.png")

if __name__ == "__main__":
    main()
