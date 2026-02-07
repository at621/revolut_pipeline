# Credit Risk Model Development Report
_Generated: 2026-02-07 21:22:31_

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
| 2 | customers.SUM(transactions.amount WHERE category = Travel) | 0.0862 | 0.6675 | 0.6505 |
| 3 | customers.SUM(accounts.MAX(transactions.amount)) | 0.0694 | 0.6726 | 0.6542 |
| 4 | customers.SUM(transactions.amount WHERE category = Entertainment) | 0.0738 | 0.6873 | 0.6512 |

- Stopping reason: AUC plateau after step 4 (no improvement for 3 steps)


## 5. Final Model — statsmodels Summary

```
                           Logit Regression Results                           
==============================================================================
Dep. Variable:             is_default   No. Observations:                 4800
Model:                          Logit   Df Residuals:                     4795
Method:                           MLE   Df Model:                            4
Date:                Sat, 07 Feb 2026   Pseudo R-squ.:                 0.06476
Time:                        21:26:32   Log-Likelihood:                -1212.2
converged:                       True   LL-Null:                       -1296.2
Covariance Type:            nonrobust   LLR p-value:                 2.996e-35
=========================================================================================================================================
                                                                            coef    std err          z      P>|z|      [0.025      0.975]
-----------------------------------------------------------------------------------------------------------------------------------------
const                                                                    -2.4897      0.058    -42.784      0.000      -2.604      -2.376
woe_customers.STD(transactions.amount)                                   -0.6232      0.130     -4.809      0.000      -0.877      -0.369
woe_customers.SUM(transactions.amount WHERE category = Travel)           -0.8593      0.169     -5.072      0.000      -1.191      -0.527
woe_customers.SUM(accounts.MAX(transactions.amount))                     -0.7846      0.186     -4.229      0.000      -1.148      -0.421
woe_customers.SUM(transactions.amount WHERE category = Entertainment)    -0.8353      0.173     -4.836      0.000      -1.174      -0.497
=========================================================================================================================================
```

## 6. Scorecard Points

| feature                                                           | bin                 |     woe |   points |
|:------------------------------------------------------------------|:--------------------|--------:|---------:|
| customers.STD(transactions.amount)                                | (-inf, 956.81)      | -0.4514 |     95.7 |
| customers.STD(transactions.amount)                                | [956.81, 1171.79)   | -0.4308 |     96.1 |
| customers.STD(transactions.amount)                                | [1171.79, 1397.73)  | -0.1602 |    100.9 |
| customers.STD(transactions.amount)                                | [1397.73, 1693.12)  |  0.1802 |    107.1 |
| customers.STD(transactions.amount)                                | [1693.12, 1863.21)  |  0.3685 |    110.4 |
| customers.STD(transactions.amount)                                | [1863.21, 2341.83)  |  1.1248 |    124   |
| customers.STD(transactions.amount)                                | [2341.83, inf)      |  3.0057 |    157.9 |
| customers.SUM(transactions.amount WHERE category = Travel)        | (-inf, -202.89)     | -0.7181 |     86   |
| customers.SUM(transactions.amount WHERE category = Travel)        | [-202.89, -119.65)  | -0.1977 |     98.9 |
| customers.SUM(transactions.amount WHERE category = Travel)        | [-119.65, -66.01)   | -0.0466 |    102.7 |
| customers.SUM(transactions.amount WHERE category = Travel)        | [-66.01, -31.67)    |  0.0678 |    105.5 |
| customers.SUM(transactions.amount WHERE category = Travel)        | [-31.67, 235.21)    |  0.1579 |    107.7 |
| customers.SUM(transactions.amount WHERE category = Travel)        | [235.21, inf)       |  0.42   |    114.2 |
| customers.SUM(accounts.MAX(transactions.amount))                  | (-inf, 4544.89)     | -0.3593 |     95.7 |
| customers.SUM(accounts.MAX(transactions.amount))                  | [4544.89, 8519.19)  |  0.0415 |    104.8 |
| customers.SUM(accounts.MAX(transactions.amount))                  | [8519.19, 14924.44) |  0.3669 |    112.1 |
| customers.SUM(accounts.MAX(transactions.amount))                  | [14924.44, inf)     |  1.0704 |    128.1 |
| customers.SUM(transactions.amount WHERE category = Entertainment) | (-inf, -358.97)     | -0.8175 |     84.1 |
| customers.SUM(transactions.amount WHERE category = Entertainment) | [-358.97, -103.21)  | -0.2956 |     96.7 |
| customers.SUM(transactions.amount WHERE category = Entertainment) | [-103.21, -57.35)   | -0.0321 |    103   |
| customers.SUM(transactions.amount WHERE category = Entertainment) | [-57.35, -16.32)    |  0.0503 |    105   |
| customers.SUM(transactions.amount WHERE category = Entertainment) | [-16.32, 53.62)     |  0.1705 |    107.9 |
| customers.SUM(transactions.amount WHERE category = Entertainment) | [53.62, 198.25)     |  0.288  |    110.8 |
| customers.SUM(transactions.amount WHERE category = Entertainment) | [198.25, inf)       |  0.3617 |    112.5 |

## 9. Benchmarking

| model                   |   gini_train |   gini_test |   gini_oot |
|:------------------------|-------------:|------------:|-----------:|
| WoE Logistic Regression |     0.37456  |    0.302417 |   0.363374 |
| Gradient Boosted Trees  |     0.981295 |    0.279504 |   0.298949 |
| Random Forest           |     0.931366 |    0.342171 |   0.371942 |

## 8. PD Calibration

- Method: Isotonic Regression
- Brier Score (before): 0.0802
- Brier Score (after): 0.0807


## 7. Model Performance

| metric      |     train |      test |       oot |
|:------------|----------:|----------:|----------:|
| AUC         | 0.68728   | 0.651209  | 0.681687  |
| GINI        | 0.37456   | 0.302417  | 0.363374  |
| KS          | 0.277909  | 0.291743  | 0.295222  |
| Brier Score | 0.0679503 | 0.0802458 | 0.0855366 |

## 10. Residual Monitoring

| feature                                                        |       miv |    p_value |
|:---------------------------------------------------------------|----------:|-----------:|
| customers.MEAN(transactions.amount WHERE category = Groceries) | 0.0577251 | 0.00999302 |
| customers.MEAN(transactions.amount WHERE category = Salary)    | 0.0451467 | 0.815309   |
| customers.SUM(transactions.amount WHERE category = Groceries)  | 0.0417705 | 0.0529225  |
| customers.MIN(accounts.SUM(transactions.amount))               | 0.0408836 | 0.558281   |
| customers.SKEW(accounts.MIN(transactions.amount))              | 0.0364809 | 0.0117162  |
| customers.MEAN(accounts.MAX(transactions.amount))              | 0.0357597 | 0.667013   |
| customers.MEAN(accounts.SUM(transactions.amount))              | 0.0330182 | 0.540187   |
| customers.MAX(accounts.STD(transactions.amount))               | 0.0328967 | 0.443825   |
| customers.MIN(accounts.MAX(transactions.amount))               | 0.0304845 | 0.686626   |
| customers.MIN(credit_applications.requested_amount)            | 0.0290314 | 0.00738762 |
