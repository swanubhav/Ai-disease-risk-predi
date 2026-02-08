def ensemble_risk(ml_probability, dl_probability):
    return round((0.6 * ml_probability + 0.4 * dl_probability), 3)
