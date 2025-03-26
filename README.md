# 🚗 Salifort Motors HR Analytics Project – Predicting Employee Turnover with Data Science 📊

Welcome to the repository of the **Salifort Motors Capstone Project** part of Google Data Anlaysis Certification, where I used data to tackle one of the most costly challenges in business: **employee attrition**. Sadly 😢 , the databse cannot be published as I'm not the owner nor it is of open for public access. Nonetheles, feel free to check the code and the results to see how I decided to tackel the problem and proposed actionable solutions.

I built predictive models to identify employees at risk of leaving, both econometric and ML models. 
The goal? **Boost retention, cut costs, and support HR with data-driven insights.**


## 🔍 Key Project Outcomes

✅ Developed multiple predictive models (Logistic Regression, Decision Tree, Random Forest)  
✅ Final **Random Forest model** achieved:  
- **96% accuracy** on test data  
- **AUC Score**: 0.956 on unseen data  
✅ Identified **top attrition factors**:
- Low satisfaction levels
- Excessive workload (avg. >175 hours/month)
- High project counts (6–7 projects)
- Lack of promotions

🧠 **Insights for HR**: Overworked, undervalued employees are most likely to leave.  
💼 **Actionable Impact**: Tailored interventions can now be made to improve employee retention!

## 📣 Non-Technical Overview

This project is about understanding **why employees leave** and using data to **predict it before it happens**.

Imagine you're in HR. Wouldn't it be amazing to know who's most likely to quit—and why? That's what we did here.  

I found that employees who were:
- working **long hours**
- had **too many projects**
- and **hadn't been promoted** in years were far more likely to leave.

✨ With this model HR can now focus efforts **where it matters most**, reduce hiring costs, and create a happier workplace.

## 📐 Technical Insights

**Modeling Pipeline**:
- Data cleaning: 15,000+ rows → handled missing values, duplicates (20%), outliers (tenure-based)
- Feature engineering: created a binary `overworked` variable
- Categorical encoding: ordinal for `salary`, one-hot for `department`

**Models Built**:
1. **Logistic Regression**
   - Good interpretability
   - F1-Score (left class): ~33%
2. **Decision Tree (GridSearchCV)**
   - AUC: 0.969  
3. **Random Forest Classifier**
   - AUC: 0.980 (CV), 0.956 (Test)
   - Precision: 96%
   - Recall: 92%
   - F1: 94%

**Key Features (by importance)**:
- `last_evaluation`
- `number_project`
- `tenure`
- `overworked`

🎯 I used **cross-validated grid search** for hyperparameter tuning and addressed potential **data leakage** by excluding sensitive variables in model variations.

### 📌 Conclusions

This project reveals critical insights into **why employees leave** based on the analyzed information and how data science can help mitigate that risk.

- Employee **satisfaction level** is the strongest negative predictor of attrition.
- **Overworking** (especially >175 hours/month) is heavily correlated with higher quit rates.
- Employees with **6 or 7 projects** had the highest likelihood of leaving.
- Lack of **promotions** and **recognition** over time contributes significantly to dissatisfaction.
- Surprisingly, high performers often left due to **burnout**, not poor evaluation.

These findings point toward **systemic issues in workload balance and employee engagement** for the analyzed dataset.


### ✅ Recommendations

Based on the predictive model results and feature analysis, here are some strategic recommendations for fruther discussions:

1. **Implement Workload Caps**  
   📉 Limit employees to **3–5 projects** at any given time to reduce burnout and attrition risk.

2. **Track and Act on Satisfaction Metrics**  
   💬 Regular surveys should be implemented, and employees with satisfaction scores <0.5 should be flagged for immediate attention.

3. **Review Promotion Cycles**  
   📈 Employees without a promotion in **5+ years** should be reviewed to improve retention.
