# app.py
from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
import os, joblib, pandas as pd, traceback, warnings, csv
from werkzeug.security import generate_password_hash, check_password_hash

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)
app.secret_key = 'your_secret_key_here'  # Change this in production

# Initialize cache with fallback
try:
    from flask_caching import Cache
    cache = Cache(app, config={'CACHE_TYPE': 'simple'})
    CACHE_AVAILABLE = True
except ImportError:
    print("[WARNING] flask-caching not available. Caching features will be disabled.")
    CACHE_AVAILABLE = False
    # Create a dummy cache object for compatibility
    class DummyCache:
        def cached(self, timeout=None):
            def decorator(f):
                return f
            return decorator
    cache = DummyCache()

# Simple in-memory user storage (not for production)
users = {}

# ✅ Safe CSV reader function (inlined instead of importing from utils)
def safe_read_csv(path):
    """
    Safely reads a CSV file and handles different delimiters.
    """
    try:
        with open(path, 'r', errors='ignore') as f:
            sample = f.read(2048)
            dialect = csv.Sniffer().sniff(sample)
            delimiter = dialect.delimiter
            print(f"[safe_read_csv] Detected delimiter: {repr(delimiter)}")
            return pd.read_csv(path, delimiter=delimiter)
    except Exception as e:
        print(f"[safe_read_csv] Failed to detect delimiter: {e}")
        # Fallback: try common delimiters, choose the one with most columns
        candidates = []
        for delim in ['\t', ',', ';']:
            try:
                df_test = pd.read_csv(path, delimiter=delim, nrows=5)
                candidates.append((delim, len(df_test.columns), df_test.columns.tolist()))
                print(f"[safe_read_csv] Tried '{delim}': {len(df_test.columns)} columns")
            except Exception as ex:
                print(f"[safe_read_csv] Failed '{delim}': {ex}")
        if candidates:
            # Choose the one with >1 columns, preferably most
            valid = [c for c in candidates if c[1] > 1]
            if valid:
                chosen = max(valid, key=lambda x: x[1])
                delim = chosen[0]
                print(f"[safe_read_csv] Chosen delimiter '{delim}' with {chosen[1]} columns: {chosen[2]}")
                return pd.read_csv(path, delimiter=delim)
        raise ValueError("Could not parse CSV with any common delimiter")

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

class FraudDetectionApp:
    def __init__(self):
        self.df = None
        self.models = {}
        self.scaler = None
        self.feature_columns = None
        self.feature_importance = None

        # [+] PROJECT-LOCAL CSV PATH
        self.csv_path = os.path.join(BASE_DIR, "data", "creditcard.csv")

        self.load_data_and_models()

    def load_data_and_models(self):
        try:
            # LOAD CSV
            if os.path.exists(self.csv_path):
                print(f"[app] [+] Loading CSV from: {self.csv_path}")
                self.df = safe_read_csv(self.csv_path)
                print(f"[app] [+] Data loaded. shape={self.df.shape}")
            else:
                print(f"[app] [X] CSV not found at: {self.csv_path}")
                self.df = None

            # LOAD MODELS (optional)
            models_dir = os.path.join(BASE_DIR, "models")
            ensemble_path = os.path.join(models_dir, "ensemble.pkl")
            scaler_path = os.path.join(models_dir, "scaler.pkl")
            featcols_path = os.path.join(models_dir, "feature_columns.pkl")

            print(f"[app] Checking model files: ensemble={os.path.exists(ensemble_path)}, scaler={os.path.exists(scaler_path)}, featcols={os.path.exists(featcols_path)}")

            if os.path.exists(ensemble_path) and os.path.exists(scaler_path) and os.path.exists(featcols_path):
                print("[app] [+] Loading saved models...")
                try:
                    self.models['ensemble'] = joblib.load(ensemble_path)
                    print("[app] Ensemble model loaded.")
                except Exception as e:
                    print(f"[app] Error loading ensemble: {e}")
                    return
                try:
                    self.scaler = joblib.load(scaler_path)
                    print("[app] Scaler loaded.")
                except Exception as e:
                    print(f"[app] Error loading scaler: {e}")
                    return
                try:
                    self.feature_columns = joblib.load(featcols_path)
                    print(f"[app] Feature columns loaded: {self.feature_columns}")
                except Exception as e:
                    print(f"[app] Error loading feature columns: {e}")
                    return

                fi_path = os.path.join(models_dir, "feature_importance.csv")
                if os.path.exists(fi_path):
                    self.feature_importance = pd.read_csv(fi_path)
                    print("[app] Feature importance loaded.")

                print("[app] [+] Models loaded successfully.")
            else:
                print("[app] [!] Model files not found in 'models/'. If you want predictions, run improved_model.py first.")

        except Exception as e:
            print("[app] [X] Error in load_data_and_models:", e)
            traceback.print_exc()

fraud_app = FraudDetectionApp()

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
        if username in users and check_password_hash(users[username], password):
            session['user_id'] = username
            return redirect(url_for('index'))
        else:
            return render_template("login.html", error="Invalid username or password")
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        if username in users:
            return render_template("signup.html", error="Username already exists")
        if password != confirm_password:
            return render_template("signup.html", error="Passwords do not match")
        users[username] = generate_password_hash(password)
        return redirect(url_for('login'))
    return render_template("signup.html")

@app.route("/logout")
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

@app.route("/")
@login_required
def index():
    username = session['user_id']
    return render_template("index.html", username=username)

@app.route("/data")
@login_required
def data_page():
    if fraud_app.df is None:
        return "[X] Dataset not loaded. Check terminal logs."
    df_sample = fraud_app.df.head(100)  # show only first 100 rows
    data_records = df_sample.to_dict(orient="records")
    columns = df_sample.columns.tolist()
    username = session['user_id']
    return render_template("data.html", data=data_records, columns=columns, total_rows=len(fraud_app.df), username=username)

@app.route("/api/data")
def api_data():
    if fraud_app.df is None:
        return jsonify({"error": "Dataset not loaded"}), 500
    df_sample = fraud_app.df.head(100)
    return jsonify({
        "total_rows": len(fraud_app.df),
        "columns": df_sample.columns.tolist(),
        "data": df_sample.to_dict(orient="records")
    })

@app.route("/api/update_data", methods=["POST"])
@login_required
def api_update_data():
    try:
        edited_data = request.get_json()
        if not edited_data:
            return jsonify({"error": "No data received"}), 400
        # Store edited data in session as list of dicts
        session['edited_data'] = edited_data
        session.modified = True
        return jsonify({"status": "success", "message": "Data updated successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/download_data")
@login_required
def download_data():
    if not os.path.exists(fraud_app.csv_path):
        return "CSV not found.", 404
    return send_file(fraud_app.csv_path, as_attachment=True)

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "GET":
        username = session['user_id']
        return render_template("predict.html", username=username)
    
    if fraud_app.df is None:
        return jsonify({"error": "Dataset not loaded"}), 500
    if 'ensemble' not in fraud_app.models or fraud_app.scaler is None or fraud_app.feature_columns is None:
        return jsonify({"error": "Model not ready. Run improved_model.py first."}), 500

    payload = request.get_json()
    if not payload:
        return jsonify({"error": "No JSON payload received"}), 400

    try:
        row = pd.DataFrame([payload], columns=fraud_app.feature_columns).fillna(0)
        row_scaled = fraud_app.scaler.transform(row)
        pred = fraud_app.models['ensemble'].predict(row_scaled)[0]
        prob = fraud_app.models['ensemble'].predict_proba(row_scaled)[0, 1] if hasattr(fraud_app.models['ensemble'], "predict_proba") else None
        features = payload
        top_importances = fraud_app.feature_importance.head(10).to_dict('records') if fraud_app.feature_importance is not None else []
        session['last_prediction'] = {
            'features': payload,
            'prediction': int(pred),
            'probability': float(prob) if prob is not None else None
        }
        session.modified = True
        return jsonify({
            "prediction": int(pred),
            "probability": float(prob) if prob is not None else None,
            "features": features,
            "top_importances": top_importances
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/last_prediction")
def api_last_prediction():
    last_pred = session.get('last_prediction', None)
    if last_pred is None:
        return jsonify({"error": "No recent prediction made"}), 404
    return jsonify(last_pred)

@app.route("/api/prediction_comparison")
def api_prediction_comparison():
    last_pred = session.get('last_prediction', None)
    if last_pred is None:
        return jsonify({"error": "No recent prediction made"}), 404
    if fraud_app.df is None:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    pred_class = last_pred['prediction']
    features = last_pred['features']
    
    # Get means for the predicted class from original data
    class_means = fraud_app.df[fraud_app.df['Class'] == pred_class][fraud_app.feature_columns].mean().to_dict()
    
    # Prepare comparison data for selected features (to match existing viz style)
    selected_features = ['Time', 'Amount', 'V1', 'V2', 'V3', 'V4', 'V5']  # Consistent with feature_means
    comparison = {}
    for feat in selected_features:
        if feat in fraud_app.feature_columns:
            pred_val = features.get(feat, 0)
            mean_val = class_means.get(feat, 0)
            comparison[feat] = {
                'prediction': pred_val,
                'class_mean': mean_val
            }
    
    # Also include prediction summary
    comparison['summary'] = {
        'prediction': pred_class,
        'probability': last_pred.get('probability', None),
        'label': 'Fraud' if pred_class == 1 else 'Not Fraud'
    }
    
    # Clear the session key after use to avoid persistence
    if 'last_prediction' in session:
        del session['last_prediction']
        session.modified = True
    
    return jsonify(comparison)

@app.route("/api/comparison_stats")
def api_comparison_stats():
    if fraud_app.df is None:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    try:
        non_fraud_count = fraud_app.df['Class'].value_counts().get(0, 0)
        fraud_count = fraud_app.df['Class'].value_counts().get(1, 0)
        total = len(fraud_app.df)
        fraud_percentage = (fraud_count / total) * 100 if total > 0 else 0
        original_stats = {
            "non_fraud": int(non_fraud_count),
            "fraud": int(fraud_count),
            "total": int(total),
            "fraud_percentage": float(fraud_percentage)
        }
        edited_data = session.get('edited_data', [])
        if not isinstance(edited_data, list):
            edited_data = []
        if edited_data:
            try:
                edited_df = pd.DataFrame(edited_data)
                if len(edited_df) > 0 and 'Class' in edited_df.columns:
                    fraud_counts = edited_df['Class'].value_counts()
                    non_fraud = fraud_counts.get(0, 0)
                    fraud = fraud_counts.get(1, 0)
                    total = len(edited_df)
                    fraud_percentage = (fraud / total) * 100 if total > 0 else 0
                    edited_stats = {
                        "non_fraud": int(non_fraud),
                        "fraud": int(fraud),
                        "total": int(total),
                        "fraud_percentage": float(fraud_percentage)
                    }
                else:
                    raise ValueError("Invalid or empty edited data")
            except Exception as inner_e:
                print(f"[api_comparison_stats] Error processing edited data: {inner_e}")
                edited_stats = {
                    "non_fraud": 0,
                    "fraud": 0,
                    "total": 0,
                    "fraud_percentage": 0.0
                }
        else:
            edited_stats = {
                "non_fraud": 0,
                "fraud": 0,
                "total": 0,
                "fraud_percentage": 0.0
            }
        return jsonify({
            "original": original_stats,
            "edited": edited_stats
        })
    except Exception as e:
        print(f"[api_comparison_stats] Unexpected error: {e}")
        return jsonify({
            "original": original_stats if 'original_stats' in locals() else {},
            "edited": {"non_fraud": 0, "fraud": 0, "total": 0, "fraud_percentage": 0.0}
        }), 200


@app.route("/api/stats")
@cache.cached(timeout=300)
def api_stats():
    if fraud_app.df is None:
        return jsonify({"error": "Dataset not loaded"}), 500

    fraud_counts = fraud_app.df['Class'].value_counts().to_dict()
    non_fraud = fraud_counts.get(0, 0)
    fraud = fraud_counts.get(1, 0)
    total = len(fraud_app.df)
    fraud_percentage = (fraud / total) * 100 if total > 0 else 0

    return jsonify({
        "non_fraud": int(non_fraud),
        "fraud": int(fraud),
        "total": int(total),
        "fraud_percentage": float(fraud_percentage)
    })

@app.route("/api/edited_stats")
def api_edited_stats():
    try:
        edited_data = session.get('edited_data', [])
        if edited_data:
            edited_df = pd.DataFrame(edited_data)
            fraud_counts = edited_df['Class'].value_counts().to_dict()
        else:
            edited_df = fraud_app.df
            fraud_counts = fraud_app.df['Class'].value_counts().to_dict()
        non_fraud = fraud_counts.get(0, 0)
        fraud = fraud_counts.get(1, 0)
        total = len(edited_df)
        fraud_percentage = (fraud / total) * 100 if total > 0 else 0
        return jsonify({
            "non_fraud": int(non_fraud),
            "fraud": int(fraud),
            "total": int(total),
            "fraud_percentage": float(fraud_percentage)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/feature_means")
@cache.cached(timeout=300)
def api_feature_means():
    if fraud_app.df is None:
        return jsonify({"error": "Dataset not loaded"}), 500

    selected_features = ['Time', 'Amount', 'V1', 'V2', 'V3', 'V4', 'V5']
    means = fraud_app.df.groupby('Class')[selected_features].mean().to_dict()
    return jsonify(means)


@app.route("/api/edited_feature_means")
def api_edited_feature_means():
    try:
        edited_data = session.get('edited_data', [])
        if edited_data:
            edited_df = pd.DataFrame(edited_data)
        else:
            edited_df = fraud_app.df
        if edited_df is None:
            return jsonify({"error": "Dataset not loaded"}), 500
        
        selected_features = ['Time', 'Amount', 'V1', 'V2', 'V3', 'V4', 'V5']
        means = edited_df.groupby('Class')[selected_features].mean().to_dict()
        return jsonify(means)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/scatter_data")
@cache.cached(timeout=300)
def api_scatter_data():
    if fraud_app.df is None:
        return jsonify({"error": "Dataset not loaded"}), 500

    sample_size = min(1000, len(fraud_app.df))
    sample = fraud_app.df.sample(n=sample_size, random_state=42)
    return jsonify({
        "V1": sample["V1"].tolist(),
        "V2": sample["V2"].tolist(),
        "Class": sample["Class"].tolist()
    })


@app.route("/api/edited_scatter_data")
def api_edited_scatter_data():
    try:
        edited_data = session.get('edited_data', [])
        if edited_data:
            edited_df = pd.DataFrame(edited_data)
            if len(edited_df) > 0:
                sample_size = min(1000, len(edited_df))
                sample = edited_df.sample(n=sample_size, random_state=42)
                return jsonify({
                    "V1": sample["V1"].tolist(),
                    "V2": sample["V2"].tolist(),
                    "Class": sample["Class"].tolist()
                })
        # Fallback to original
        if fraud_app.df is None:
            return jsonify({"error": "Dataset not loaded"}), 500
        
        sample_size = min(1000, len(fraud_app.df))
        sample = fraud_app.df.sample(n=sample_size, random_state=42)
        return jsonify({
            "V1": sample["V1"].tolist(),
            "V2": sample["V2"].tolist(),
            "Class": sample["Class"].tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/visualize")
# @login_required  # Temporarily disabled for testing
def visualize():
    username = session.get('user_id', 'Guest')  # Use Guest if not logged in
    return render_template("visualize.html", username=username)

if __name__ == "__main__":
    print("[+] Starting web app; templates folder:", os.path.join(BASE_DIR, "templates"))
    app.run(debug=True, host="0.0.0.0", port=5000)
