Credit Card Fraud Detection
Overview

This project is an interactive, frontend web application designed for visualizing, editing, and analyzing credit card transaction data to help detect potential fraud. Built with React, TypeScript, and Tailwind CSS, the application provides an intuitive interface for exploring transaction patterns and anomalies.

Features

. Interactive Data Table: View, sort, and edit transaction data with pagination. Any changes are reflected immediately in the visualizations.

. Dynamic Visualizations: A responsive bar chart displays the distribution of transaction categories, updating in real-time as the data is edited.

. Data Summary: Key statistics, such as total transactions, total amount, and known fraudulent transactions, are displayed and updated live.

. Modern UI/UX: Clean, responsive design with dark mode support using Tailwind CSS.

. Editable Transactions: Modify transaction details such as amount, category, or merchant directly from the table.

How to Use

1. View Data: Scroll through the "Transaction Data" table. Use pagination controls to navigate large datasets.

2. Edit Data: Click the pencil icon on any row to modify its details. Save changes with the checkmark or cancel with the 'x'.

3. Visualize Changes: Observe how the "Transaction Categories" chart and the "Data Summary" card update automatically after edits.

4. Analyze Trends: Use the visualizations to identify unusual patterns that might indicate fraudulent activity.

## Python Backend Setup

This project includes a Python backend with Flask web application and machine learning models for credit card fraud detection.

### Virtual Environment Setup

The project includes a pre-configured virtual environment for running the Python components. To ensure all dependencies are properly isolated:

**For Command Prompt:**
```cmd
run_project.bat
```

**For PowerShell:**
```powershell
.\run_project.ps1
```

**For Git Bash or other Unix-like terminals:**
```bash
source activate
python run_project.py
```

### What the setup includes:

- **Automated virtual environment activation**: The batch/PowerShell scripts automatically activate the virtual environment before running the Python code
- **Dependency management**: All required packages (Flask, scikit-learn, pandas, etc.) are installed in the virtual environment
- **Project launcher**: The `run_project.py` script handles the complete workflow:
  1. Installs/updates dependencies automatically
  2. Trains the fraud detection models
  3. Starts the Flask web application
  4. Opens the application in your browser

### Manual activation (if needed):

**Command Prompt:**
```cmd
activate.bat
```

**PowerShell:**
```powershell
.\Activate.ps1
```

**Git Bash/Unix:**
```bash
source activate
```

After activation, you can run Python scripts normally and they will use the virtual environment.

src/
├── components/         # React components
│   ├── charts/         # Chart components using Recharts
│   └── shared/         # Reusable components like Cards, Buttons, and Icons
├── data/               # Sample transaction data
├── services/           # Optional services (e.g., local analysis or utilities)
├── types.ts            # TypeScript type definitions
├── App.tsx             # Root component managing state
└── index.tsx           # Main entry point
public/
└── index.html          # HTML template    

## You can Access the project also through this link: https://drive.google.com/file/d/1OD8-lV5_0EBiFR-AUnVSh5929E1vaiWi/view?usp=drive_link
