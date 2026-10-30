from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from feature_engineering import add_engineered_features

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Alerta académica ML', page_icon='🎓', layout='wide')

@st.cache_resource
def load_bundle():
    return joblib.load(ROOT/'modelo_academico.joblib')

bundle=load_bundle()
model=bundle['pipeline']
features=bundle['original_features']
defaults=bundle['defaults']
ranges=bundle['ranges']
choices=bundle['choices']
metrics=bundle['metrics']
classes=bundle['classes']

st.title('🎓 Analítica académica: riesgo de deserción y resultado estudiantil')
st.caption('Modelo clásico de Machine Learning | Random Forest multiclase | ACA N.° 1 - Aprendizaje Automático')

with st.sidebar:
    st.header('Uso responsable')
    st.info('La salida orienta priorización de acompañamiento. No debe usarse como decisión académica automática ni como diagnóstico causal.')
    st.markdown('**Dataset:** UCI Predict Students\' Dropout and Academic Success')
    st.markdown('**Clases:** Dropout · Enrolled · Graduate')
    st.markdown('**Modelo:** Random Forest balanceado')

TAB1,TAB2,TAB3,TAB4=st.tabs(['Predicción individual','Predicción por CSV','Desempeño del modelo','Metodología'])

# Friendly labels for the most visible coded fields.
COURSE_LABELS={33:'Biofuel Production Technologies',171:'Animation and Multimedia Design',8014:'Social Service (evening)',9003:'Agronomy',9070:'Communication Design',9085:'Veterinary Nursing',9119:'Informatics Engineering',9130:'Equinculture',9147:'Management',9238:'Social Service',9254:'Tourism',9500:'Nursing',9556:'Oral Hygiene',9670:'Advertising and Marketing Management',9773:'Journalism and Communication',9853:'Basic Education',9991:'Management (evening)'}
BINARY_LABELS={0:'No',1:'Sí'}
GENDER_LABELS={0:'Femenino (código 0)',1:'Masculino (código 1)'}
ATTEND_LABELS={0:'Nocturna',1:'Diurna'}

quick_fields=[
    'Course','Tuition fees up to date','Debtor','Scholarship holder','Age at enrollment','Admission grade',
    'Curricular units 1st sem (enrolled)','Curricular units 1st sem (approved)','Curricular units 1st sem (grade)',
    'Curricular units 2nd sem (enrolled)','Curricular units 2nd sem (approved)','Curricular units 2nd sem (grade)'
]

def select_categorical(col, key):
    vals=choices[col]
    default=defaults[col]
    idx=vals.index(default) if default in vals else 0
    if col=='Course':
        return st.selectbox('Programa/curso', vals, index=idx, key=key, format_func=lambda v:f'{v} - {COURSE_LABELS.get(v,"Código observado")}')
    if col in ['Tuition fees up to date','Debtor','Scholarship holder','Displaced','Educational special needs','International']:
        return st.selectbox(col, vals, index=idx, key=key, format_func=lambda v:BINARY_LABELS.get(v,str(v)))
    if col=='Gender':
        return st.selectbox(col, vals, index=idx, key=key, format_func=lambda v:GENDER_LABELS.get(v,str(v)))
    if col=='Daytime/evening attendance':
        return st.selectbox(col, vals, index=idx, key=key, format_func=lambda v:ATTEND_LABELS.get(v,str(v)))
    return st.selectbox(col, vals, index=idx, key=key)

def numeric_input(col,key):
    r=ranges[col]; default=defaults[col]
    is_int=float(r['min']).is_integer() and float(r['max']).is_integer() and float(default).is_integer()
    if is_int:
        return st.number_input(col,min_value=int(r['min']),max_value=int(r['max']),value=int(default),step=1,key=key)
    return st.number_input(col,min_value=float(r['min']),max_value=float(r['max']),value=float(default),key=key)

def predict_row(rowdict):
    raw=pd.DataFrame([rowdict],columns=features)
    eng=add_engineered_features(raw)
    pred=model.predict(eng)[0]
    proba=model.predict_proba(eng)[0]
    return pred, pd.Series(proba,index=classes,name='Probabilidad')

with TAB1:
    st.subheader('Simulador de caso individual')
    st.write('El modo rápido solicita los predictores académicos y financieros más accionables; las variables restantes se fijan en la mediana o moda del dataset. Para análisis institucionales use la carga CSV con las 36 variables originales.')
    row=dict(defaults)
    c1,c2,c3=st.columns(3)
    for i,col in enumerate(quick_fields):
        with [c1,c2,c3][i%3]:
            if col in choices:
                row[col]=select_categorical(col,f'q_{col}')
            else:
                row[col]=numeric_input(col,f'q_{col}')
    if st.button('Calcular predicción',type='primary'):
        pred,ps=predict_row(row)
        pdrop=float(ps.get('Dropout',0))
        st.metric('Resultado estimado',pred)
        if pdrop>=.50:
            st.error(f'Prioridad de acompañamiento: ALTA · P(Dropout)={pdrop:.1%}')
        elif pdrop>=.30:
            st.warning(f'Prioridad de acompañamiento: MEDIA · P(Dropout)={pdrop:.1%}')
        else:
            st.success(f'Prioridad de acompañamiento: BAJA · P(Dropout)={pdrop:.1%}')
        st.bar_chart(ps)
        st.caption('Los umbrales de priorización son operativos y deben calibrarse con política institucional y datos locales.')

with TAB2:
    st.subheader('Predicción por archivo CSV')
    st.write('El archivo debe contener las 36 columnas originales del dataset. La columna `Target` es opcional y se ignora para predecir.')
    sample=pd.read_csv(ROOT/'data.csv',sep=';').head(5)
    st.download_button('Descargar ejemplo de estructura',sample.to_csv(index=False).encode('utf-8'),'ejemplo_entrada.csv','text/csv')
    up=st.file_uploader('Cargue el CSV',type=['csv'])
    if up is not None:
        try:
            batch = pd.read_csv(up, sep=None, engine="python")
            batch.columns=[c.strip() for c in batch.columns]
            if 'Target' in batch.columns:
                batch=batch.drop(columns='Target')
            missing=[c for c in features if c not in batch.columns]
            if missing:
                st.error('Faltan columnas: '+', '.join(missing))
            else:
                raw=batch[features].copy()
                eng=add_engineered_features(raw)
                pp=model.predict_proba(eng); pr=model.predict(eng)
                result=raw.copy(); result['Prediccion']=pr
                for i,c in enumerate(classes): result[f'P_{c}']=pp[:,i]
                st.success(f'{len(result)} registros procesados correctamente.')
                st.dataframe(result.head(50),use_container_width=True)
                st.download_button('Descargar predicciones',result.to_csv(index=False).encode('utf-8'),'predicciones.csv','text/csv')
        except Exception as e:
            st.exception(e)

with TAB3:
    st.subheader("Evaluación en conjunto de prueba independiente")

    a, b, c, d, e = st.columns(5)

    a.metric("Exactitud", f"{metrics['Accuracy']:.3f}")
    b.metric("F1 macro", f"{metrics['F1 macro']:.3f}")
    c.metric("F1 ponderado", f"{metrics['F1 weighted']:.3f}")
    d.metric("Exactitud balanceada", f"{metrics['Balanced accuracy']:.3f}")
    e.metric("Log loss", f"{metrics['Log loss']:.3f}")

    st.write("### Métricas por clase")

    reporte = {
        "Dropout": {
            "precision": 0.83,
            "recall": 0.72,
            "f1-score": 0.77,
            "support": 284
        },
        "Enrolled": {
            "precision": 0.46,
            "recall": 0.58,
            "f1-score": 0.51,
            "support": 159
        },
        "Graduate": {
            "precision": 0.85,
            "recall": 0.83,
            "f1-score": 0.84,
            "support": 442
        }
    }

    tabla = pd.DataFrame([
        {
            "Clase": clase,
            "Precisión": valores["precision"],
            "Recall": valores["recall"],
            "F1": valores["f1-score"],
            "Casos": valores["support"]
        }
        for clase, valores in reporte.items()
    ])

    st.dataframe(
        tabla.style.format({
            "Precisión": "{:.2f}",
            "Recall": "{:.2f}",
            "F1": "{:.2f}"
        }),
        use_container_width=True
    )

    st.caption(
        "Resultados obtenidos sobre 885 registros del conjunto de prueba. "
        "Las métricas por clase corresponden al reporte de evaluación del modelo."
    )

    if (ROOT / "figures/matriz_confusion.png").exists():
        st.image(
            str(ROOT / "figures/matriz_confusion.png"),
            caption="Matriz de confusión"
        )

    if (ROOT / "figures/importancia_variables.png").exists():
        st.image(
            str(ROOT / "figures/importancia_variables.png"),
            caption="Importancia por permutación"
        )

with TAB4:
    st.subheader('Flujo metodológico')
    st.markdown('''
1. **EDA y auditoría:** 4.424 registros; 36 predictores; sin faltantes ni duplicados.
2. **Preprocesamiento:** codificación one-hot de variables nominales y estandarización de variables numéricas dentro del pipeline.
3. **Ingeniería de características:** tasas de aprobación y cambios de desempeño entre semestres.
4. **Modelado:** comparación de algoritmos clásicos y ajuste de Random Forest con validación cruzada sobre entrenamiento.
5. **Evaluación:** prueba independiente estratificada con Accuracy, F1 macro, Balanced Accuracy, AUC OvR, matriz de confusión y métricas por clase.
6. **Despliegue:** interfaz Streamlit para predicción individual y por lotes.
''')
    st.warning('Validez externa: el dataset corresponde a una institución portuguesa. Antes de adoptar el modelo en otra universidad debe reentrenarse o validarse con datos locales.')
