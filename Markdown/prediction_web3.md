### Feature Importance 

The approach I decided to use, as I feel it is more reliable then any other is called **Permutation Importance** which cleverly uses some data that has gone *unused* at when random samples are selected for each Decision Tree (this stage is called "bootstrap sampling" or "bootstrapping")

These observations that were not randomly selected for each Decision Tree are known as *Out of Bag* observations and these can be used for testing the accuracy of each particular Decision Tree.

For each Decision Tree, all of the *Out of Bag* observations are gathered and then passed through.  Once all of these observations have been run through the Decision Tree, we obtain an accuracy score for these predictions, which in the case of a regression problem could be Mean Squared Error or r-squared.

In order to understand the *importance*, we *randomise* the values within one of the input variables - a process that essentially destroys any relationship that might exist between that input variable and the output variable - and run that updated data through the Decision Tree again, obtaining a second accuracy score.  The difference between the original accuracy and the new accuracy gives us a view on how important that particular variable is for predicting the output.

*Permutation Importance* is often preferred over *Feature Importance* which can at times inflate the importance of numerical features. Both are useful, and in most cases will give fairly similar results.


```python

# calculate permutation importance
result = permutation_importance(clf_full, X_test, y_test, n_repeats=10, random_state=42)

permutation_importance = pd.DataFrame(result["importances_mean"])
feature_names = pd.DataFrame(X.columns)
permutation_importance_summary = pd.concat([feature_names, permutation_importance], axis=1)
permutation_importance_summary.columns = ["input variable", "importance"]
permutation_importance_summary.sort_values(by = "importance", inplace=True)

# plot permutation importance
plt.barh(permutation_importance_summary["input_variable"],permutation_importance_summary["permutation_importance"])
plt.title("Permutation Importance of Random Forest")
plt.xlabel("Permutation Importance")
plt.tight_layout()
plt.show()

```

That code gives a *Permutation Importance* plot!
