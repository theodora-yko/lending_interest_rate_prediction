# lending_interest_rate_prediction

This project simulates how an underwriting engine like Slope's could offer more competitive rates to low-risk borrowers using Lending Club data. [Lending Club](https://www.lendingclub.com/personal-deposits) is a peer-to-peer Lending company based in the US. They match people looking to invest money with people looking to borrow money. 

data in use: [Lending Club Loan Data](https://www.openintro.org/data/index.php?data=loans_full_schema)
- loan information: interest rate, loan amount, current loan status (Current, Late, Fully Paid, etc.), etc. 
- Features (aka independent variables) include credit scores, number of finance inquiries, address including zip codes and state, and collections among others.


## Question:

Given pre-loan features for an accepted LendingClub loan application:
1. Can we predict how low or high the offered interest rate was?
2. Can we estimate the interest rate LendingClub would offer to new applicants?

## Business Motivation:**

As a potential competitor to LendingClub, our goal is to attract more borrowers by offering more competitive interest rates to safe customers.

If we can:
- Accurately identify "safe" loan applications, and
- Offer slightly lower interest rates (and advertise it),
then we can gain market share without increasing default risk.

## Project Goals**
1. Predict LendingClub’s offered interest rate given borrower information — to simulate competitive pricing.
2. Identify "safe" loans we could offer a lower rate to, while maintaining predicted low risk.
3. Create interpretable models (using SHAP or Scorecard techniques) to justify lending decisions.

## Project Summary
In this project, I developed a machine learning pipeline to predict interest rates for low-risk loans and optimize lending strategies. Starting from exploratory data analysis (EDA), I identified missing data patterns, feature distributions, and engineered transformations where appropriate. Missing values were handled systematically, and input features were standardized using a StandardScaler.

After establishing a baseline with linear regression — which failed to perform adequately (R² ≈ -5e22%, RMSE ≈ $810M) — I transitioned to XGBoost regression with K-Fold cross-validation for robust model evaluation. The final XGBoost model achieved a mean R² of 30.78% (std 1.58%) and a mean RMSE of 30.32% (std 0.32%), indicating stable and meaningful predictive ability given the noise and complexity of financial loan data. Diagnostics including residual plots and QQ plots confirmed that the model generalized appropriately without major distributional issues or significant overfitting.

Based on the model's risk scores and predicted interest rates, I simulated a lending strategy offering interest rates 0.01 lower than predicted to the safest 5% of borrowers, resulting in a projected additional profit of approximately $860,152.

Key constraints and assumptions of this project include:

- The model was trained only on historical low-risk loan data, assuming past borrower behaviors are predictive of future outcomes.
- No time-based train/test splits were used; random sampling was assumed sufficient due to the lack of explicit temporal features.
- Profitability calculations assumed that the loan amount and risk behavior remain stable under the proposed adjusted interest rates.
- Residual normality was not assumed, as XGBoost is non-parametric and robust to non-normal error structures.

Through this project, I strengthened my practical skills in real-world model validation (early stopping, K-Fold CV), residual diagnostics, scaling transformations, feature engineering, and translating predictive models into business impact. Beyond modeling, I learned the importance of systematically handling assumptions, evaluating model stability across folds, and critically assessing the link between model outputs and real-world business decisions.

## TLDR:
We aim to predict LendingClub interest rates, design a competitive lending strategy focused on safe borrowers, balance profitability and risk, and ensure model transparency — with evaluation grounded in both predictive accuracy and business outcome simulations. 
