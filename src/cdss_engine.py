def cdss(risk_score):
    if risk_score >= 0.8:
        return "🚨 CRITICAL RISK: Immediate medical attention required."
    elif risk_score >= 0.5:
        return "⚠️ HIGH RISK: Doctor consultation strongly advised."
    elif risk_score >= 0.3:
        return "🟡 MODERATE RISK: Lifestyle changes and monitoring recommended."
    else:
        return "🟢 LOW RISK: Maintain healthy habits."
