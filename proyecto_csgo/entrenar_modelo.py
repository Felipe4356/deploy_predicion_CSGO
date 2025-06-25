# limpieza_entrenamiento.py
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. Cargar dataset
df = pd.read_csv("data/Anexo ET_demo_round_traces_2022 .csv", sep=";")  # Note the space before .csv


# 2. Copia de respaldo
df_backup = df.copy()

# 3. Limpieza
df_backup.drop(columns=['Unnamed: 0', 'AbnormalMatch', 'FirstKillTime', 'TimeAlive', 'TravelledDistance'], inplace=True)
df_backup.dropna(inplace=True)
df_backup = df_backup[df_backup['MatchKills'] <= 28]
df_backup = df_backup[df_backup['MatchAssists'] <= 8]
df_backup = df_backup[(df_backup['RoundId'] >= 1) & (df_backup['RoundId'] <= 30)]

# 4. Transformaciones
le = LabelEncoder()
df_backup['Team'] = le.fit_transform(df_backup['Team'])
df_backup['Map'] = le.fit_transform(df_backup['Map'])

df_backup['RoundWinner'] = df_backup['RoundWinner'].astype(bool).replace({True: 1, False: 0})
df_backup['MatchWinner'] = df_backup['MatchWinner'].astype(bool).replace({True: 1, False: 0})
df_backup['Survived'] = df_backup['Survived'].astype(bool).replace({True: 1, False: 0})

# 5. Modelo de regresión
X_reg = df_backup[['TeamStartingEquipmentValue']]
y_reg = df_backup[['RoundStartingEquipmentValue']]
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

modelo_reg = LinearRegression()
modelo_reg.fit(X_train_reg, y_train_reg)

# Guardar modelo regresión
joblib.dump(modelo_reg, "models/modelo_regresion.pkl")
joblib.dump(X_reg.columns.tolist(), "models/columnas_regresion.pkl")

# 6. Modelo de clasificación
X_clf= df_backup[['TeamStartingEquipmentValue','MatchKills','MatchAssists','Map']]
y_clf= df_backup['MatchWinner']

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

modelo_clf = RandomForestClassifier(    
       n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=3,
    class_weight="balanced",
    max_features='sqrt',
    oob_score=True,
    random_state=42,
    verbose=2)

modelo_clf.fit(X_train_clf, y_train_clf)

y_pred = modelo_clf.predict(X_test_clf)
print(classification_report(y_test_clf, y_pred))

# Guardar modelo clasificación
joblib.dump(modelo_clf, "models/modelo_clasificacion.pkl")
joblib.dump(X_clf.columns.tolist(), "models/columnas_clasificacion.pkl")
print("✅ Modelos entrenados y guardados con éxito ( RandomForestClassifier y Regresión).")

