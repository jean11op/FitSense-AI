"""Funciones de resumen y hallazgos exploratorios."""
import pandas as pd

def missing_summary(df):
    return pd.DataFrame({"faltantes":df.isna().sum(),"porcentaje":100*df.isna().mean()}).sort_values("faltantes",ascending=False)

def descriptive_summary(df):
    numeric=df.select_dtypes(include="number")
    return numeric.describe().T

def correlation_matrix(df):
    return df.select_dtypes(include="number").corr()

def findings(df):
    numeric=df.select_dtypes(include="number"); result=[]
    result.append(f"La tabla contiene {len(df):,} filas y {df.shape[1]} columnas.")
    result.append(f"Se detectaron {int(df.duplicated().sum())} filas duplicadas exactas.")
    result.append(f"Hay {int(df.isna().sum().sum())} celdas faltantes en total ({100*df.isna().mean().mean():.2f}% del conjunto de celdas). Los ceros se mantienen como mediciones válidas.")
    if numeric.notna().any().any():
        corr=numeric.corr().abs().where(lambda x: x<1).stack().sort_values(ascending=False)
        if not corr.empty:
            a,b=corr.index[0]; result.append(f"La asociación lineal absoluta más alta entre variables numéricas es {a}–{b} (r={numeric[[a,b]].corr().iloc[0,1]:.2f}); es descriptiva y no implica causalidad.")
    return result

