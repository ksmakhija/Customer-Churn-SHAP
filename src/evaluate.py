from sklearn.metrics import roc_auc_score

def evaluate_model(model, X_test, y_test):
    preds = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, preds)
