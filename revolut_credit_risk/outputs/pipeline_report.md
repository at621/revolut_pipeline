# Credit Risk Model Development Report
_Generated: 2026-02-07 20:13:36_

## 1. Data Summary

- Customers: 10,000
- Accounts: 20,072
- Transactions: 1,203,641
- Applications: 8,000
- Default rate: 8.4%


## 2. Feature Generation (DFS)

- Features generated: 104
- Depth: 3
- Observations: 8000


## 3. Binning & WoE Summary

| Feature | IV | Bins | Monotonic Trend | Status |
|---|---|---|---|---|
| requested_amount | 0.0281 | 4 | auto_asc_desc | OPTIMAL |
| customers.age | 0.0298 | 5 | auto_asc_desc | OPTIMAL |
| customers.MAX(credit_applications.requested_amount) | 0.0281 | 4 | auto_asc_desc | OPTIMAL |
| customers.MEAN(credit_applications.requested_amount) | 0.0281 | 4 | auto_asc_desc | OPTIMAL |
| customers.MIN(credit_applications.requested_amount) | 0.0281 | 4 | auto_asc_desc | OPTIMAL |
| customers.SUM(credit_applications.requested_amount) | 0.0281 | 4 | auto_asc_desc | OPTIMAL |
| customers.COUNT(accounts) | 0.0172 | 5 | auto_asc_desc | OPTIMAL |
| customers.NUM_UNIQUE(accounts.account_type) | 0.0154 | 4 | auto_asc_desc | OPTIMAL |
| customers.NUM_UNIQUE(accounts.currency) | 0.0178 | 5 | auto_asc_desc | OPTIMAL |
| customers.COUNT(transactions) | 0.0157 | 4 | auto_asc_desc | OPTIMAL |
| customers.MAX(transactions.amount) | 0.2980 | 7 | auto_asc_desc | OPTIMAL |
| customers.MEAN(transactions.amount) | 0.2644 | 7 | auto_asc_desc | OPTIMAL |
| customers.MIN(transactions.amount) | 0.0166 | 5 | auto_asc_desc | OPTIMAL |
| customers.NUM_UNIQUE(transactions.category) | 0.0121 | 4 | auto_asc_desc | OPTIMAL |
| customers.NUM_UNIQUE(transactions.merchant_name) | 0.0001 | 4 | auto_asc_desc | OPTIMAL |
| customers.NUM_UNIQUE(transactions.transaction_state) | 0.0120 | 3 | auto_asc_desc | OPTIMAL |
| customers.SKEW(transactions.amount) | 0.0286 | 3 | auto_asc_desc | OPTIMAL |
| customers.STD(transactions.amount) | 0.3970 | 8 | auto_asc_desc | OPTIMAL |
| customers.SUM(transactions.amount) | 0.1651 | 7 | auto_asc_desc | OPTIMAL |
| customers.PERCENT_TRUE(credit_applications.IS_WEEKEND(application_date)) | 0.0037 | 3 | auto_asc_desc | OPTIMAL |
| customers.MAX(accounts.COUNT(transactions)) | 0.0148 | 6 | auto_asc_desc | OPTIMAL |
| customers.MAX(accounts.MEAN(transactions.amount)) | 0.2157 | 6 | auto_asc_desc | OPTIMAL |
| customers.MAX(accounts.MIN(transactions.amount)) | 0.0147 | 4 | auto_asc_desc | OPTIMAL |
| customers.MAX(accounts.NUM_UNIQUE(transactions.category)) | 0.0120 | 4 | auto_asc_desc | OPTIMAL |
| customers.MAX(accounts.NUM_UNIQUE(transactions.merchant_name)) | 0.0009 | 4 | auto_asc_desc | OPTIMAL |
| customers.MAX(accounts.NUM_UNIQUE(transactions.transaction_state)) | 0.0120 | 3 | auto_asc_desc | OPTIMAL |
| customers.MAX(accounts.SKEW(transactions.amount)) | 0.0345 | 5 | auto_asc_desc | OPTIMAL |
| customers.MAX(accounts.STD(transactions.amount)) | 0.3464 | 6 | auto_asc_desc | OPTIMAL |
| customers.MAX(accounts.SUM(transactions.amount)) | 0.1822 | 8 | auto_asc_desc | OPTIMAL |
| customers.MEAN(accounts.COUNT(transactions)) | 0.0186 | 4 | auto_asc_desc | OPTIMAL |
| customers.MEAN(accounts.MAX(transactions.amount)) | 0.2580 | 5 | auto_asc_desc | OPTIMAL |
| customers.MEAN(accounts.MEAN(transactions.amount)) | 0.2702 | 6 | auto_asc_desc | OPTIMAL |
| customers.MEAN(accounts.MIN(transactions.amount)) | 0.0102 | 5 | auto_asc_desc | OPTIMAL |
| customers.MEAN(accounts.NUM_UNIQUE(transactions.category)) | 0.0124 | 5 | auto_asc_desc | OPTIMAL |
| customers.MEAN(accounts.NUM_UNIQUE(transactions.merchant_name)) | 0.0145 | 5 | auto_asc_desc | OPTIMAL |
| customers.MEAN(accounts.NUM_UNIQUE(transactions.transaction_state)) | 0.0120 | 3 | auto_asc_desc | OPTIMAL |
| customers.MEAN(accounts.SKEW(transactions.amount)) | 0.0515 | 7 | auto_asc_desc | OPTIMAL |
| customers.MEAN(accounts.STD(transactions.amount)) | 0.3041 | 7 | auto_asc_desc | OPTIMAL |
| customers.MEAN(accounts.SUM(transactions.amount)) | 0.1914 | 6 | auto_asc_desc | OPTIMAL |
| customers.MIN(accounts.COUNT(transactions)) | 0.0022 | 4 | auto_asc_desc | OPTIMAL |
| customers.MIN(accounts.MAX(transactions.amount)) | 0.2006 | 5 | auto_asc_desc | OPTIMAL |
| customers.MIN(accounts.MEAN(transactions.amount)) | 0.1924 | 8 | auto_asc_desc | OPTIMAL |
| customers.MIN(accounts.NUM_UNIQUE(transactions.category)) | 0.0055 | 5 | auto_asc_desc | OPTIMAL |
| customers.MIN(accounts.NUM_UNIQUE(transactions.merchant_name)) | 0.0016 | 4 | auto_asc_desc | OPTIMAL |
| customers.MIN(accounts.NUM_UNIQUE(transactions.transaction_state)) | 0.0120 | 3 | auto_asc_desc | OPTIMAL |
| customers.MIN(accounts.SKEW(transactions.amount)) | 0.0514 | 7 | auto_asc_desc | OPTIMAL |
| customers.MIN(accounts.STD(transactions.amount)) | 0.2233 | 6 | auto_asc_desc | OPTIMAL |
| customers.MIN(accounts.SUM(transactions.amount)) | 0.1612 | 7 | auto_asc_desc | OPTIMAL |
| customers.NUM_UNIQUE(accounts.MONTH(open_date)) | 0.0161 | 4 | auto_asc_desc | OPTIMAL |
| customers.NUM_UNIQUE(accounts.WEEKDAY(open_date)) | 0.0183 | 5 | auto_asc_desc | OPTIMAL |

## 3b. Bivariate Analysis

|   # | feature                                                                  |          iv | iv_strength   |   univariate_gini |   n_bins |
|----:|:-------------------------------------------------------------------------|------------:|:--------------|------------------:|---------:|
|   1 | customers.STD(transactions.amount)                                       | 0.397039    | Strong        |       -0.284922   |        8 |
|   2 | customers.MAX(accounts.STD(transactions.amount))                         | 0.346413    | Strong        |       -0.273441   |        6 |
|   3 | customers.MEAN(transactions.amount WHERE category = Salary)              | 0.330143    | Strong        |       -0.297287   |        7 |
|   4 | customers.MEAN(accounts.STD(transactions.amount))                        | 0.304129    | Strong        |       -0.269465   |        7 |
|   5 | customers.MAX(transactions.amount)                                       | 0.297958    | Medium        |       -0.27428    |        7 |
|   6 | customers.MEAN(accounts.MEAN(transactions.amount))                       | 0.270167    | Medium        |       -0.253112   |        6 |
|   7 | customers.MEAN(transactions.amount)                                      | 0.264442    | Medium        |       -0.256824   |        7 |
|   8 | customers.MEAN(accounts.MAX(transactions.amount))                        | 0.258004    | Medium        |       -0.242473   |        5 |
|   9 | customers.MEAN(transactions.amount WHERE transaction_state = COMPLETED)  | 0.244974    | Medium        |       -0.251288   |        8 |
|  10 | customers.SUM(accounts.MEAN(transactions.amount))                        | 0.238535    | Medium        |       -0.246186   |        5 |
|  11 | customers.MIN(accounts.STD(transactions.amount))                         | 0.223272    | Medium        |       -0.223104   |        6 |
|  12 | customers.MAX(accounts.MEAN(transactions.amount))                        | 0.215711    | Medium        |       -0.238965   |        6 |
|  13 | customers.MIN(accounts.MAX(transactions.amount))                         | 0.200649    | Medium        |       -0.215931   |        5 |
|  14 | customers.MIN(accounts.MEAN(transactions.amount))                        | 0.192413    | Medium        |       -0.207845   |        8 |
|  15 | customers.MEAN(accounts.SUM(transactions.amount))                        | 0.191428    | Medium        |       -0.217263   |        6 |
|  16 | customers.MAX(accounts.SUM(transactions.amount))                         | 0.182208    | Medium        |       -0.215948   |        8 |
|  17 | customers.SUM(transactions.amount WHERE transaction_state = COMPLETED)   | 0.181313    | Medium        |       -0.226747   |        8 |
|  18 | customers.SUM(accounts.STD(transactions.amount))                         | 0.169305    | Medium        |       -0.205748   |        6 |
|  19 | customers.SUM(transactions.amount)                                       | 0.165063    | Medium        |       -0.212098   |        7 |
|  20 | customers.MIN(accounts.SUM(transactions.amount))                         | 0.161222    | Medium        |       -0.19913    |        7 |
|  21 | customers.SUM(accounts.MAX(transactions.amount))                         | 0.160144    | Medium        |       -0.204455   |        5 |
|  22 | customers.SUM(transactions.amount WHERE category = Salary)               | 0.127527    | Medium        |       -0.171906   |        8 |
|  23 | customers.SUM(transactions.amount WHERE category = Travel)               | 0.105278    | Medium        |       -0.156212   |        7 |
|  24 | customers.SUM(transactions.amount WHERE category = Entertainment)        | 0.0996198   | Weak          |       -0.161754   |        8 |
|  25 | customers.MEAN(transactions.amount WHERE category = Groceries)           | 0.0976664   | Weak          |       -0.143689   |        6 |
|  26 | customers.MEAN(transactions.amount WHERE category = Travel)              | 0.0775736   | Weak          |       -0.140023   |        7 |
|  27 | customers.SUM(transactions.amount WHERE category = Groceries)            | 0.0707065   | Weak          |       -0.120005   |        5 |
|  28 | customers.COUNT(transactions WHERE category = Salary)                    | 0.0628018   | Weak          |       -0.106909   |        8 |
|  29 | customers.STD(accounts.SUM(transactions.amount))                         | 0.0600769   | Weak          |       -0.0955418  |        5 |
|  30 | customers.SUM(accounts.SKEW(transactions.amount))                        | 0.059117    | Weak          |       -0.122923   |        6 |
|  31 | customers.MEAN(transactions.amount WHERE category = Entertainment)       | 0.0564885   | Weak          |       -0.130098   |        6 |
|  32 | customers.MEAN(accounts.SKEW(transactions.amount))                       | 0.0514575   | Weak          |       -0.123038   |        7 |
|  33 | customers.MIN(accounts.SKEW(transactions.amount))                        | 0.0514039   | Weak          |       -0.123202   |        7 |
|  34 | customers.SKEW(accounts.MIN(transactions.amount))                        | 0.046194    | Weak          |       -0.0783724  |        5 |
|  35 | customers.COUNT(transactions WHERE category = Entertainment)             | 0.0449712   | Weak          |       -0.092632   |        7 |
|  36 | customers.STD(accounts.MEAN(transactions.amount))                        | 0.0426672   | Weak          |       -0.0801378  |        5 |
|  37 | customers.MAX(accounts.SKEW(transactions.amount))                        | 0.0344878   | Weak          |       -0.0951896  |        5 |
|  38 | customers.age                                                            | 0.0297599   | Weak          |       -0.0770946  |        5 |
|  39 | customers.SKEW(transactions.amount)                                      | 0.0286385   | Weak          |       -0.0760939  |        3 |
|  40 | customers.COUNT(transactions WHERE category = Travel)                    | 0.0286348   | Weak          |       -0.09115    |        6 |
|  41 | customers.MAX(credit_applications.requested_amount)                      | 0.0280818   | Weak          |       -0.0739586  |        4 |
|  42 | requested_amount                                                         | 0.0280818   | Weak          |       -0.0739586  |        4 |
|  43 | customers.MEAN(credit_applications.requested_amount)                     | 0.0280818   | Weak          |       -0.0739586  |        4 |
|  44 | customers.SUM(credit_applications.requested_amount)                      | 0.0280818   | Weak          |       -0.0739586  |        4 |
|  45 | customers.MIN(credit_applications.requested_amount)                      | 0.0280818   | Weak          |       -0.0739586  |        4 |
|  46 | customers.STD(accounts.MIN(transactions.amount))                         | 0.0205666   | Weak          |       -0.0600365  |        6 |
|  47 | customers.COUNT(transactions WHERE category = Groceries)                 | 0.0201328   | Weak          |       -0.0679355  |        4 |
|  48 | customers.NUM_UNIQUE(accounts.YEAR(open_date))                           | 0.0194443   | Poor          |       -0.0533895  |        4 |
|  49 | customers.MEAN(accounts.COUNT(transactions))                             | 0.0186302   | Poor          |       -0.0572496  |        4 |
|  50 | customers.NUM_UNIQUE(accounts.WEEKDAY(open_date))                        | 0.0182602   | Poor          |       -0.0561063  |        5 |
|  51 | customers.COUNT(transactions WHERE transaction_state = COMPLETED)        | 0.018245    | Poor          |       -0.0587094  |        4 |
|  52 | customers.NUM_UNIQUE(accounts.currency)                                  | 0.0177535   | Poor          |       -0.0609425  |        5 |
|  53 | customers.STD(accounts.MAX(transactions.amount))                         | 0.0176084   | Poor          |       -0.05726    |        5 |
|  54 | customers.COUNT(accounts)                                                | 0.017227    | Poor          |       -0.0573811  |        5 |
|  55 | customers.MIN(transactions.amount)                                       | 0.0165611   | Poor          |       -0.0594292  |        5 |
|  56 | customers.NUM_UNIQUE(accounts.MONTH(open_date))                          | 0.016141    | Poor          |       -0.0533797  |        4 |
|  57 | customers.COUNT(transactions)                                            | 0.0156782   | Poor          |       -0.0541173  |        4 |
|  58 | customers.NUM_UNIQUE(transactions.accounts.currency)                     | 0.0154668   | Poor          |       -0.0603407  |        5 |
|  59 | customers.NUM_UNIQUE(accounts.account_type)                              | 0.0153559   | Poor          |       -0.0488115  |        4 |
|  60 | customers.MAX(accounts.COUNT(transactions))                              | 0.0147662   | Poor          |       -0.0540656  |        6 |
|  61 | customers.MAX(accounts.MIN(transactions.amount))                         | 0.0146589   | Poor          |       -0.0484698  |        4 |
|  62 | customers.MEAN(accounts.NUM_UNIQUE(transactions.merchant_name))          | 0.0145212   | Poor          |       -0.0495374  |        5 |
|  63 | customers.SKEW(accounts.SUM(transactions.amount))                        | 0.0145157   | Poor          |       -0.0437639  |        4 |
|  64 | customers.SUM(transactions.amount WHERE category = Rent)                 | 0.0140042   | Poor          |       -0.0516513  |        5 |
|  65 | customers.SUM(accounts.NUM_UNIQUE(transactions.merchant_name))           | 0.0139139   | Poor          |       -0.0528025  |        4 |
|  66 | customers.SUM(accounts.NUM_UNIQUE(transactions.category))                | 0.0138211   | Poor          |       -0.0531879  |        5 |
|  67 | customers.STD(accounts.SKEW(transactions.amount))                        | 0.0136864   | Poor          |       -0.0395725  |        3 |
|  68 | customers.NUM_UNIQUE(transactions.accounts.account_type)                 | 0.0127636   | Poor          |       -0.0475349  |        4 |
|  69 | customers.MEAN(accounts.NUM_UNIQUE(transactions.category))               | 0.0124339   | Poor          |       -0.0445126  |        5 |
|  70 | customers.NUM_UNIQUE(transactions.category)                              | 0.0120787   | Poor          |       -0.0380408  |        4 |
|  71 | customers.MAX(accounts.NUM_UNIQUE(transactions.category))                | 0.0119854   | Poor          |       -0.0369922  |        4 |
|  72 | customers.SUM(accounts.NUM_UNIQUE(transactions.transaction_state))       | 0.0119573   | Poor          |       -0.0365195  |        4 |
|  73 | customers.NUM_UNIQUE(transactions.transaction_state)                     | 0.0119548   | Poor          |       -0.0358919  |        3 |
|  74 | customers.MAX(accounts.NUM_UNIQUE(transactions.transaction_state))       | 0.0119548   | Poor          |       -0.0358919  |        3 |
|  75 | customers.MEAN(accounts.NUM_UNIQUE(transactions.transaction_state))      | 0.0119548   | Poor          |       -0.0358919  |        3 |
|  76 | customers.MIN(accounts.NUM_UNIQUE(transactions.transaction_state))       | 0.0119548   | Poor          |       -0.0358919  |        3 |
|  77 | customers.MEAN(accounts.MIN(transactions.amount))                        | 0.0101559   | Poor          |       -0.0432949  |        5 |
|  78 | customers.STD(accounts.COUNT(transactions))                              | 0.00851847  | Poor          |       -0.0473597  |        6 |
|  79 | customers.COUNT(transactions WHERE category = Rent)                      | 0.0073954   | Poor          |       -0.0378779  |        5 |
|  80 | customers.STD(accounts.NUM_UNIQUE(transactions.category))                | 0.00713239  | Poor          |       -0.042057   |        4 |
|  81 | customers.SKEW(accounts.MAX(transactions.amount))                        | 0.00669496  | Poor          |       -0.0223153  |        4 |
|  82 | customers.SKEW(accounts.MEAN(transactions.amount))                       | 0.00632241  | Poor          |       -0.0299187  |        4 |
|  83 | customers.MIN(accounts.NUM_UNIQUE(transactions.category))                | 0.00552868  | Poor          |       -0.0386985  |        5 |
|  84 | customers.PERCENT_TRUE(credit_applications.IS_WEEKEND(application_date)) | 0.00372958  | Poor          |       -0.027181   |        3 |
|  85 | customers.STD(accounts.NUM_UNIQUE(transactions.merchant_name))           | 0.0037028   | Poor          |       -0.0299402  |        4 |
|  86 | customers.SKEW(accounts.COUNT(transactions))                             | 0.0036942   | Poor          |       -0.0156702  |        5 |
|  87 | customers.PERCENT_TRUE(accounts.IS_WEEKEND(open_date))                   | 0.00300452  | Poor          |       -0.0190269  |        3 |
|  88 | customers.SKEW(accounts.NUM_UNIQUE(transactions.merchant_name))          | 0.00233054  | Poor          |       -0.0140924  |        3 |
|  89 | customers.MIN(accounts.COUNT(transactions))                              | 0.00222982  | Poor          |       -0.0240277  |        4 |
|  90 | customers.MEAN(transactions.amount WHERE category = Rent)                | 0.00210297  | Poor          |       -0.0242453  |        5 |
|  91 | customers.MIN(accounts.NUM_UNIQUE(transactions.merchant_name))           | 0.0016472   | Poor          |       -0.0209968  |        4 |
|  92 | customers.MAX(accounts.NUM_UNIQUE(transactions.merchant_name))           | 0.000909905 | Poor          |       -0.0152811  |        4 |
|  93 | customers.SKEW(accounts.STD(transactions.amount))                        | 0.000803161 | Poor          |       -0.00898697 |        3 |
|  94 | customers.SKEW(accounts.NUM_UNIQUE(transactions.transaction_state))      | 0.000708559 | Poor          |       -0.00998702 |        4 |
|  95 | customers.SUM(accounts.MIN(transactions.amount))                         | 0.000214794 | Poor          |       -0.00630889 |        3 |
|  96 | customers.NUM_UNIQUE(transactions.merchant_name)                         | 0.000105189 | Poor          |       -0.0052572  |        4 |
|  97 | customers.SKEW(accounts.NUM_UNIQUE(transactions.category))               | 0           | Poor          |      nan          |        2 |
|  98 | customers.STD(accounts.NUM_UNIQUE(transactions.transaction_state))       | 0           | Poor          |      nan          |        2 |

## 4. MIV Feature Selection

| Step | Feature Added | MIV | AUC (Train) | AUC (Test) |
|---|---|---|---|---|
| 1 | customers.STD(transactions.amount) | 0.3970 | 0.6425 | 0.6705 |
| 2 | customers.SKEW(accounts.MIN(transactions.amount)) | 0.0240 | 0.6500 | 0.6435 |
| 3 | customers.COUNT(transactions WHERE category = Salary) | 0.0152 | 0.6562 | 0.6441 |

- Stopping reason: MIV (0.0000) below threshold (0.01)


## 5. Final Model — statsmodels Summary

```
                           Logit Regression Results                           
==============================================================================
Dep. Variable:             is_default   No. Observations:                 4800
Model:                          Logit   Df Residuals:                     4796
Method:                           MLE   Df Model:                            3
Date:                Sat, 07 Feb 2026   Pseudo R-squ.:                 0.04530
Time:                        20:17:35   Log-Likelihood:                -1237.4
converged:                       True   LL-Null:                       -1296.2
Covariance Type:            nonrobust   LLR p-value:                 2.760e-25
=============================================================================================================================
                                                                coef    std err          z      P>|z|      [0.025      0.975]
-----------------------------------------------------------------------------------------------------------------------------
const                                                        -2.4911      0.058    -43.200      0.000      -2.604      -2.378
woe_customers.STD(transactions.amount)                       -0.9826      0.130     -7.531      0.000      -1.238      -0.727
woe_customers.SKEW(accounts.MIN(transactions.amount))        -0.9157      0.299     -3.067      0.002      -1.501      -0.331
woe_customers.COUNT(transactions WHERE category = Salary)     0.0461      0.254      0.182      0.856      -0.451       0.543
=============================================================================================================================
```

## 6. Scorecard Points

| feature                                               | bin                |     woe |   points |
|:------------------------------------------------------|:-------------------|--------:|---------:|
| customers.STD(transactions.amount)                    | (-inf, 956.81)     | -0.4514 |    125.6 |
| customers.STD(transactions.amount)                    | [956.81, 1171.79)  | -0.4308 |    126.2 |
| customers.STD(transactions.amount)                    | [1171.79, 1397.73) | -0.1602 |    133.9 |
| customers.STD(transactions.amount)                    | [1397.73, 1693.12) |  0.1802 |    143.5 |
| customers.STD(transactions.amount)                    | [1693.12, 1863.21) |  0.3685 |    148.9 |
| customers.STD(transactions.amount)                    | [1863.21, 2341.83) |  1.1248 |    170.3 |
| customers.STD(transactions.amount)                    | [2341.83, inf)     |  3.0057 |    223.6 |
| customers.SKEW(accounts.MIN(transactions.amount))     | (-inf, -1.00)      | -0.3216 |    129.9 |
| customers.SKEW(accounts.MIN(transactions.amount))     | [-1.00, 0.70)      | -0.037  |    137.4 |
| customers.SKEW(accounts.MIN(transactions.amount))     | [0.70, 1.52)       |  0.3946 |    148.8 |
| customers.SKEW(accounts.MIN(transactions.amount))     | [1.52, inf)        |  0.7245 |    157.6 |
| customers.COUNT(transactions WHERE category = Salary) | (-inf, 0.50)       | -0.1909 |    138.7 |
| customers.COUNT(transactions WHERE category = Salary) | [0.50, 4.50)       | -0.0741 |    138.5 |
| customers.COUNT(transactions WHERE category = Salary) | [4.50, 7.50)       | -0.0602 |    138.5 |
| customers.COUNT(transactions WHERE category = Salary) | [7.50, 10.50)      |  0.0365 |    138.4 |
| customers.COUNT(transactions WHERE category = Salary) | [10.50, 12.50)     |  0.152  |    138.2 |
| customers.COUNT(transactions WHERE category = Salary) | [12.50, 17.50)     |  0.6834 |    137.5 |
| customers.COUNT(transactions WHERE category = Salary) | [17.50, inf)       |  0.7407 |    137.4 |

## 9. Benchmarking

| model                   |   gini_train |   gini_test |   gini_oot |
|:------------------------|-------------:|------------:|-----------:|
| WoE Logistic Regression |     0.31239  |    0.288104 |   0.311087 |
| Gradient Boosted Trees  |     0.981295 |    0.277463 |   0.296374 |
| Random Forest           |     0.931745 |    0.351057 |   0.37484  |

## 8. PD Calibration

- Method: Isotonic Regression
- Brier Score (before): 0.0807
- Brier Score (after): 0.0806


## 7. Model Performance

| metric      |     train |      test |       oot |
|:------------|----------:|----------:|----------:|
| AUC         | 0.656195  | 0.644052  | 0.655544  |
| GINI        | 0.31239   | 0.288104  | 0.311087  |
| KS          | 0.241061  | 0.235424  | 0.251411  |
| Brier Score | 0.0691604 | 0.0806948 | 0.0878577 |

## 10. Residual Monitoring

| feature                                                           |        miv |
|:------------------------------------------------------------------|-----------:|
| customers.MEAN(accounts.COUNT(transactions))                      | 0.00390752 |
| customers.SKEW(accounts.SUM(transactions.amount))                 | 0.00278917 |
| customers.COUNT(transactions WHERE category = Entertainment)      | 0          |
| customers.COUNT(transactions WHERE category = Rent)               | 0          |
| customers.COUNT(transactions WHERE category = Travel)             | 0          |
| customers.COUNT(transactions WHERE transaction_state = COMPLETED) | 0          |
| customers.COUNT(accounts)                                         | 0          |
| customers.MAX(accounts.COUNT(transactions))                       | 0          |
| customers.MAX(accounts.MEAN(transactions.amount))                 | 0          |
| customers.MAX(accounts.NUM_UNIQUE(transactions.category))         | 0          |
