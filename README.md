# Customer Churn Prediction

## Week 1: Exploratory Data Analysis

### Dataset
- Source: Telco Customer Churn Dataset (Kaggle)
- Size: 7,043 customers, 21 columns
- Target: Predict customer churn (Yes/No)

### Files
- `week1_eda.ipynb` - Exploratory analysis notebook
- `customer_data.csv` - Dataset (not included in GitHub by default; download separately)

### Key Findings (Week 1)
- Overall churn rate: **26.54%**
- Month-to-month contract churn: **42.71%** (highest among contracts)
- Fiber optic churn: **41.89%** (higher than DSL **18.96%**)
- Electronic check churn: **45.29%** (highest among payment methods)
- Tenure < 6 months churn: **54.27%**

### Next Steps
- Week 2: Build machine learning models
- Week 3: Model optimization
- Week 4: Deploy interactive dashboard

### Setup
```bash
pip install pandas numpy matplotlib seaborn jupyter
jupyter notebook week1_eda.ipynb
```
