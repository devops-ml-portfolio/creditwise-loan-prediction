#  **Intelligent Loan Approval System** 

**Built an end-to-end supervised ML pipeline using KNN, Logistic Regression and Naive Bayes to predict loan approval.**  
**Implemented Binary Classification along with EDA, feature engineering & model evaluation (Precision, Recall, F1).**


## **PROBLEM STATEMENT**

**HBC Trust Bank**, a mid-sized financial company in India, offers personal and home loans across urban and rural regions. Every day, hundreds of customers apply through online and branch applications.

### **Current Challenges**:
-Manual verification = slow + biased
-Good customers rejected → Lost revenue
-High-risk approved → Default losses

### **Solution**: **Intelligent ML Loan Approval System**
- **Automatically analyzes** applicant details (income, credit score, DTI ratio, etc.)
- **Predicts Approve/Reject** before final human verification
- **Learns patterns** from historical loan data
- **Fast, accurate, unbiased decisions**


##  **RESULTS** (Test Set) - **PRECISION PRIORITY** 

| Model                   | **Precision** | F1-Score | Recall | Accuracy |
|-------------------------|---------------|----------|--------|----------|
| **Naive Bayes**         | **0.80**      | 0.76     | 0.73   | 86.5%    |
|   Logistic Regression   | 0.77          | 0.79     | 0.80   | 87%      |
|   KNN                   | 0.62          | 0.52     | 0.44   | 75%      |

**PRODUCTION MODEL**: **Naive Bayes** - Highest precision minimizes risky approvals

**What we built** (Simple):
- Explored loan data  
- Fixed missing values  
- Feature engineering  
- Split data 80/20  
- Trained 3 ML models  
- Picked best model (80% accurate)  
- Exported for production


##  **1-CLICK PRODUCTION** 

```bash
# Clone & Deploy (45 seconds)
git clone https://github.com/YOURUSERNAME/creditwise-loan-prediction.git
cd creditwise-loan-prediction
pip install -r requirements.txt
jupyter notebook creditwise_pipeline.ipynb




