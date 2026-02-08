def explain_features(feature_names, model):
    importance = model.feature_importances_
    explanation = dict(zip(feature_names, importance))
    return sorted(explanation.items(), key=lambda x: x[1], reverse=True)
