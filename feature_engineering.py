
def add_engineered_features(X_raw):
    X = X_raw.copy()

    # Columnas académicas originales
    grade_1 = "Curricular units 1st sem (grade)"
    grade_2 = "Curricular units 2nd sem (grade)"

    approved_1 = "Curricular units 1st sem (approved)"
    approved_2 = "Curricular units 2nd sem (approved)"

    evaluations_1 = "Curricular units 1st sem (evaluations)"
    evaluations_2 = "Curricular units 2nd sem (evaluations)"

    # Promedio de notas de ambos semestres
    if grade_1 in X.columns and grade_2 in X.columns:
        X["Mean semester grade"] = X[
            [grade_1, grade_2]
        ].mean(axis=1)

        X["Grade change"] = X[grade_2] - X[grade_1]

    # Tasa de aprobación del primer semestre
    if approved_1 in X.columns and evaluations_1 in X.columns:
        X["Approval rate 1"] = (
            X[approved_1]
            / X[evaluations_1].replace(0, float("nan"))
        )

    # Tasa de aprobación del segundo semestre
    if approved_2 in X.columns and evaluations_2 in X.columns:
        X["Approval rate 2"] = (
            X[approved_2]
            / X[evaluations_2].replace(0, float("nan"))
        )

    # Cambio de unidades aprobadas entre semestres
    if approved_1 in X.columns and approved_2 in X.columns:
        X["Approved change"] = X[approved_2] - X[approved_1]

    return X
