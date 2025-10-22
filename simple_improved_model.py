import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def main():
    """Main function to run improved fraud detection"""
    print("IMPROVED CREDIT CARD FRAUD DETECTION")
    print("="*60)
    
    try:
        # Load data
        print("Loading Credit Card Fraud Dataset...")
        df = pd.read_csv('data/creditcard.csv', sep='\t')
        print(f"Dataset Shape: {df.shape}")
        print(f"Fraud Rate: {(df['Class'].sum() / len(df)) * 100:.2f}%")
        
        # Prepare features
        print("Preparing Features...")
        feature_columns = [f'V{i}' for i in range(1, 29)] + ['Amount']
        X = df[feature_columns]
        y = df['Class']
        
        print(f"Features used: {len(feature_columns)}")
        print(f"Training samples: {len(X)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        print("Training Multiple Models...")
        
        # Model 1: Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42, 
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        
        # Model 2: Gradient Boosting
        print("Training Gradient Boosting...")
        gb_model = GradientBoostingClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        
        # Model 3: Logistic Regression
        print("Training Logistic Regression...")
        lr_model = LogisticRegression(
            random_state=42,
            max_iter=1000
        )
        lr_model.fit(X_train_scaled, y_train)

        # Model 4: Support Vector Machine (SVM)
        print("Training Support Vector Machine...")
        svm_model = SVC(
            probability=True,
            random_state=42,
            kernel='rbf'
        )
        svm_model.fit(X_train_scaled, y_train)

        # Model 5: Naive Bayes (Gaussian)
        print("Training Naive Bayes...")
        nb_model = GaussianNB()
        nb_model.fit(X_train_scaled, y_train)

        # Model 6: Decision Tree
        print("Training Decision Tree...")
        dt_model = DecisionTreeClassifier(
            max_depth=10,
            random_state=42
        )
        dt_model.fit(X_train_scaled, y_train)

        # Model 7: K-Nearest Neighbors
        print("Training K-Nearest Neighbors...")
        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_model.fit(X_train_scaled, y_train)

        # Model 8: Neural Network (MLP)
        print("Training Neural Network...")
        nn_model = MLPClassifier(
            hidden_layer_sizes=(50, 25),
            max_iter=1000,
            random_state=42
        )
        nn_model.fit(X_train_scaled, y_train)

        # Model 9: AdaBoost
        print("Training AdaBoost...")
        ada_model = AdaBoostClassifier(
            n_estimators=100,
            random_state=42
        )
        ada_model.fit(X_train_scaled, y_train)

        # Model 10: Gaussian Process Classifier
        print("Training Gaussian Process...")
        gp_model = GaussianProcessClassifier(random_state=42)
        gp_model.fit(X_train_scaled, y_train)

        # Create ensemble
        print("Training Ensemble...")
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('gb', gb_model),
                ('lr', lr_model),
                ('svm', svm_model),
                ('nb', nb_model),
                ('dt', dt_model),
                ('knn', knn_model),
                ('nn', nn_model),
                ('ada', ada_model),
                ('gp', gp_model)
            ],
            voting='soft'
        )
        ensemble.fit(X_train_scaled, y_train)
        
        # Evaluate models
        print("Evaluating Models...")
        models = {
            'Random Forest': rf_model,
            'Gradient Boosting': gb_model,
            'Logistic Regression': lr_model,
            'Support Vector Machine': svm_model,
            'Naive Bayes': nb_model,
            'Decision Tree': dt_model,
            'K-Nearest Neighbors': knn_model,
            'Neural Network': nn_model,
            'AdaBoost': ada_model,
            'Gaussian Process': gp_model,
            'Ensemble': ensemble
        }
        
        results = {}
        for name, model in models.items():
            print(f"\nEvaluating {name}...")
            
            # Make predictions
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            cm = confusion_matrix(y_test, y_pred)
            
            # Calculate additional metrics
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            
            results[name] = {
                'auc': auc_score,
                'precision': precision,
                'recall': recall,
                'f1': f1_score,
                'accuracy': accuracy,
                'confusion_matrix': cm,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"   ROC-AUC: {auc_score:.4f}")
            print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
            print(f"   Recall: {recall:.4f} ({recall*100:.2f}%)")
            print(f"   F1-Score: {f1_score:.4f} ({f1_score*100:.2f}%)")
            print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Save models
        print("Saving Models...")
        os.makedirs('models', exist_ok=True)

        # Save individual models (excluding ensemble for separate handling)
        individual_models = {k: v for k, v in models.items() if k != 'Ensemble'}
        for name, model in individual_models.items():
            filename = f'models/{name.lower().replace(" ", "_")}.pkl'
            joblib.dump(model, filename)
            print(f"   Saved {name}: {filename}")

        # Save ensemble separately
        joblib.dump(ensemble, 'models/ensemble_model.pkl')
        print("   Saved Ensemble: models/ensemble_model.pkl")
        
        # Save scaler
        joblib.dump(scaler, 'models/scaler.pkl')
        print("   Saved scaler: models/scaler.pkl")
        
        # Save feature columns
        joblib.dump(feature_columns, 'models/feature_columns.pkl')
        print("   Saved feature columns: models/feature_columns.pkl")
        
        # Save feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv('models/feature_importance.csv', index=False)
        print("   Saved feature importance: models/feature_importance.csv")
        
        # Print comprehensive results summary
        print("\nMODEL PERFORMANCE SUMMARY")
        print("="*80)
        print(f"{'Model':<25} {'ROC-AUC':<10} {'Precision':<12} {'Recall':<10} {'F1-Score':<10} {'Accuracy':<10}")
        print("-" * 80)

        for name, metrics in results.items():
            print(f"{name:<25} {metrics['auc']:<10.4f} {metrics['precision']:<12.4f} {metrics['recall']:<10.4f} {metrics['f1']:<10.4f} {metrics['accuracy']:<10.4f}")

        # Find best performing models
        best_auc = max(results.items(), key=lambda x: x[1]['auc'])
        best_f1 = max(results.items(), key=lambda x: x[1]['f1'])

        print("-" * 80)
        print(f"Best ROC-AUC: {best_auc[0]} ({best_auc[1]['auc']:.4f})")
        print(f"Best F1-Score: {best_f1[0]} ({best_f1[1]['f1']:.4f})")

        print("\n" + "="*60)
        print("IMPROVED FRAUD DETECTION COMPLETED!")
        print("Check the 'models/' directory for saved models")
        print(f"Total models trained: {len(models)}")
        print(f"Models saved in: models/")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
