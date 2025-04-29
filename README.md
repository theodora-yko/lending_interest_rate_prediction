# lending_interest_rate_prediction

This project simulates how an underwriting engine like Slope's could offer more competitive rates to low-risk borrowers using Lending Club data.

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

## TLDR:

We aim to predict LendingClub interest rates, design a competitive lending strategy focused on safe borrowers, balance profitability and risk, and ensure model transparency — with evaluation grounded in both predictive accuracy and business outcome simulations.