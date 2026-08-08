#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import ttest_rel, f_oneway, f
from statsmodels.stats.multicomp import pairwise_tukeyhsd

data = pd.read_excel('ensemble learning data.xlsx')
X = data[['Strain Rate', 'NaOH Concentration', 'Soaking Time']]
y = data['Load'].values.reshape(-1, 1)
scaler = StandardScaler()
y_scaled = scaler.fit_transform(y).flatten()
models = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Linear Regression': LinearRegression(),
    'Lasso': Lasso(random_state=42),
    'Ridge': Ridge(random_state=42)
}
meta_learner = LinearRegression()
kf = KFold(n_splits=8, shuffle=True, random_state=42)
rmse_scores = {name: [] for name in models.keys()}
rmse_scores['Ensemble'] = []
for train_idx, test_idx in kf.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]
    val_size = int(0.2 * len(X_train))
    X_val, X_subtrain = X_train[:val_size], X_train[val_size:]
    y_val, y_subtrain = y_train[:val_size], y_train[val_size:]
    val_preds = pd.DataFrame()
    test_preds = pd.DataFrame()
    for name, model in models.items():
        model.fit(X_subtrain, y_subtrain)
        val_preds[name] = model.predict(X_val)
        test_preds[name] = model.predict(X_test)
    meta_learner.fit(val_preds, y_val)
    ensemble_preds = meta_learner.predict(test_preds)
    for name in models.keys():
        rmse = np.sqrt(mean_squared_error(y_test, test_preds[name]))
        rmse_scores[name].append(rmse)
    rmse_ens = np.sqrt(mean_squared_error(y_test, ensemble_preds))
    rmse_scores['Ensemble'].append(rmse_ens)
print("\n--- Paired t-tests (Ensemble vs Base Models) ---")
for name in models.keys():
    t_stat, p_val = ttest_rel(rmse_scores['Ensemble'], rmse_scores[name])
    print(f"Ensemble vs {name}: t = {t_stat:.3f}, p = {p_val:.5f}")
all_scores = [rmse_scores[name] for name in rmse_scores.keys()]
f_stat, p_val = f_oneway(*all_scores)
all_rmse_values = np.concatenate(all_scores)
grand_mean = np.mean(all_rmse_values)
ssb = sum([len(scores) * (np.mean(scores) - grand_mean) ** 2 for scores in all_scores])
ssw = sum([sum((scores - np.mean(scores))**2) for scores in all_scores])
sst = ssb + ssw
df_between = len(all_scores) - 1
df_within = len(all_rmse_values) - len(all_scores)
df_total = len(all_rmse_values) - 1
ms_between = ssb / df_between
ms_within = ssw / df_within
f_crit = f.ppf(1-0.05, df_between, df_within)
print("\n--- One-way ANOVA ---")
anova_table = pd.DataFrame({
    "Source of Variation": ["Between Groups", "Within Groups", "Total"],
    "SS": [ssb, ssw, sst],
    "df": [df_between, df_within, df_total],
    "MS": [ms_between, ms_within, ""],
    "F": [f_stat, "", ""],
    "P-value": [p_val, "", ""],
    "F crit": [f_crit, "", ""]
})
print(anova_table.to_string(index=False))
model_names = []
for name, scores in rmse_scores.items():
    model_names.extend([name]*len(scores))
df = pd.DataFrame({'RMSE': all_rmse_values, 'Model': model_names})
print("\n--- Tukey HSD Post-hoc Test ---")
tukey = pairwise_tukeyhsd(endog=df['RMSE'], groups=df['Model'], alpha=0.05)
print(tukey)
df.to_excel("anova_ttest_results.xlsx", index=False)
print("\nResults saved in: 'anova_ttest_results.xlsx'")


# In[ ]:





# In[ ]:




