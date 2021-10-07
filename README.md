# dais2021_full_ml_lifecycle
Demo assets for DAIS 2021 [Learn to use Databricks for the full ML lifecycle](https://databricks.com/session_na21/learn-to-use-databricks-for-the-full-ml-lifecycle)


Data is available for free on the web.  In a Databricks notebook run the following in an empty cell:

```
%sh
wget https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
```

This will download the data to the machine your notebook is attached to.  From there you can read the data in with Spark, pandas, or R.

```
%python
import pandas as pd
pdf = pd.read_csv("/databricks/driver/Telco-Customer-Churn.csv")
display(pdf)
```

