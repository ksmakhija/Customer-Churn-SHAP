from xgboost import XGBClassifier

def train_model(X, y):
    model = XGBClassifier(
        eval_metric="logloss",
        max_depth=4,
        n_estimators=150,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X, y)
    return model
