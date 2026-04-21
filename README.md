# Credit Card Fraud Detection & Clustering

This project implements an end-to-end machine learning pipeline to detect fraudulent credit card transactions and perform customer segmentation using unsupervised learning. It features a real-time interactive dashboard built with **Gradio** for single and batch predictions.

##  Features
- **Exploratory Data Analysis (EDA):** Visualizing transaction distributions and fraud patterns using Seaborn and Matplotlib.
- **Unsupervised Learning:** Implements **K-Means Clustering** to segment transactions, using the **Elbow Method** to determine the optimal number of clusters.
- **Dimensionality Reduction:** Uses **t-SNE** and **PCA** to visualize high-dimensional transaction data in 2D space.
- **Interactive Dashboard:** - **Single Transaction Prediction:** Input manual features to classify a transaction instantly.
    - **Batch Prediction:** Upload a `.csv` file to process thousands of transactions at once and receive a downloadable results file.

##  Tech Stack
- **Languages:** Python
- **Libraries:** - Data Processing: `Pandas`, `NumPy`
    - Machine Learning: `Scikit-Learn` (KMeans, StandardScaler, PCA, t-SNE)
    - Visualization: `Matplotlib`, `Seaborn`
    - UI/Deployment: `Gradio`

##  Dataset
The project uses the **Credit Card Fraud Detection Dataset**, which contains transactions made by European cardholders. Features $V1$ through $V28$ are principal components obtained with PCA to protect user privacy.

##  Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn gradio
```

### 3. Run the Project
Open the `CCFD18 (1).ipynb` notebook in Jupyter or Google Colab and run all cells. The Gradio interface will launch at the end, providing a local or public URL to access the dashboard.

##  Methodology
1. **Preprocessing:** Data is cleaned by removing missing values and normalized using `StandardScaler` to ensure all features contribute equally to the distance-based clustering.
2. **Clustering:** K-Means identifies patterns in the transaction data. The performance is evaluated using the **Silhouette Score**.
3. **Classification:** The model distinguishes between legitimate (Class 0) and fraudulent (Class 1) transactions.

##  Dashboard Preview
The dashboard includes:
- **Feature Inputs:** 30 numerical fields for transaction data.
- **Prediction Output:** A clear classification of "Fraud" or "Legitimate".
- **CSV Batch Upload:** For enterprise-level data processing.

***

