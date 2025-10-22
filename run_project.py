import os
import sys
import subprocess
import time
import socket


def run_improved_model():
    """Run the improved model script to train and save models"""
    print("\n===== STEP 1: TRAINING MODELS =====\n")
    model_file = 'improved_model.py'
    print(f"Checking if '{model_file}' exists...")
    if not os.path.exists(model_file):
        print(f"❌ Model file '{model_file}' not found.")
        return False
    print(f"[OK] '{model_file}' found. Running model training...")
    try:
        subprocess.run([sys.executable, 'improved_model.py'], check=True)
        print("\n[+] Models trained and saved successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[X] Error running improved_model.py: {e}\n")
        return False

def run_web_app():
    """Run the Flask web application and open browser"""
    print("\n===== STEP 2: STARTING WEB APPLICATION =====\n")
    app_file = 'app.py'
    print(f"Checking if '{app_file}' exists...")
    if not os.path.exists(app_file):
        print(f"❌ Web app file '{app_file}' not found.")
        return
    print(f"[OK] '{app_file}' found. Starting the web application...")
    print("The web application will be available at: http://localhost:5000")
    print("\nAvailable pages:")
    print("  - Dashboard: http://localhost:5000")
    print("  - Data View: http://localhost:5000/data")
    print("  - Prediction: http://localhost:5000/predict")
    print("  - Interactive Visualizations: http://localhost:5000/visualize")
    print("\nPress Ctrl+C to stop the application when you're done.\n")
    
    try:
        # Start Flask app in background
        process = subprocess.Popen([sys.executable, 'app.py'])
        
        # Wait for Flask server to be ready
        print("Waiting for Flask server to start...")
        def wait_for_server(host='localhost', port=5000, timeout=30):
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    result = sock.connect_ex((host, port))
                    sock.close()
                    if result == 0:
                        print(f"Flask server ready on {host}:{port}")
                        return True
                    time.sleep(1)
                except Exception:
                    time.sleep(1)
            print(f"Flask server did not start within {timeout} seconds")
            return False
        
        if wait_for_server():
            # Open browser using Windows start command for reliability
            os.system('start http://localhost:5000/visualize')
        else:
            print("Server not ready, please open http://localhost:5000/visualize manually")
        
        # Keep showing Flask logs until user stops
        process.wait()
    except KeyboardInterrupt:
        print("\n[S] Stopping web application...")
        process.terminate()
        print("[+] Web application stopped.\n")
    except Exception as e:
        print(f"\n[X] Unexpected error: {e}\n")

def check_requirements():
    """Check if all required packages are installed"""
    print("\n===== CHECKING REQUIREMENTS =====\n")
    req_file = 'requirements_web.txt'
    print(f"Checking if '{req_file}' exists...")
    if not os.path.exists(req_file):
        print(f"[ERROR] Requirements file '{req_file}' not found.")
        return False
    print(f"[OK] '{req_file}' found. Installing requirements...")
    try:
        
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', req_file], check=True)
        print("\n[+] All requirements installed successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[X] Error installing requirements: {e}\n")
        return False

def main():
    """Main function to run the entire project"""
    print("\n=====================================================")
    print("       CREDIT CARD FRAUD DETECTION PROJECT        ")
    print("=====================================================\n")
    
    # Step 1: Check requirements
    if not check_requirements():
        print("\n[X] Failed to install requirements. Exiting...\n")
        return
    
    # Step 2: Run improved model
    if not run_improved_model():
        print("\n[!] Model training failed or was incomplete.")
        print("Proceeding to web application anyway...")
        # proceed = input("Do you want to proceed to the web application anyway? (y/n): ")
        # if proceed.lower() != 'y':
        #     print("\n[X] Exiting...\n")
        #     return
    
    # Step 3: Run web application
    run_web_app()

if __name__ == "__main__":
    main()