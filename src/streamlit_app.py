"""Interfaz de FitSense AI con resultados trazables y vistas públicas pseudonimizadas."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from .config import DATA_ORIGINAL, DEFAULT_FEATURES
from .data_loader import load_activity, load_sleep
from .integration import integrate
from .eda import missing_summary, descriptive_summary, correlation_matrix, findings
from .data_dictionary import dictionary_frame
from .preprocessing import prepare_features
from .kmeans_model import search_k, fit_kmeans
from .dbscan_model import fit_dbscan, k_distances, explore_dbscan
from .pca_model import fit_pca
from .combined_models import fit_pca_kmeans, fit_pca_dbscan, reconstruct_centroids
from .evaluation import clustering_metrics
from .model_comparison import compare_models, common_assigned_comparison
from .kmeans_stability import assess_kmeans_stability
from .dbscan_stability import assess_dbscan_stability
from .profiling import profile_clusters, outlier_days, kmeans_outlier_candidates
from .decision_support import evidence_matrix, decision_guidance
from .result_tracking import (
    dataframe_fingerprint, participant_mapping, pseudonymize_frame,
    preprocessing_fingerprint, pca_fingerprint, result_fingerprint,
    register_result, result_matches, fingerprint, feature_export_frames,
)

NAV=["Inicio","Dataset y calidad","Exploración de datos","Preprocesamiento","K-Means","DBSCAN","Análisis PCA","Modelos combinados","Comparación y estabilidad","Perfiles y conclusiones"]
MODEL_KEYS={"K-Means":"kmeans","DBSCAN":"dbscan","PCA + K-Means":"pca_kmeans","PCA + DBSCAN":"pca_dbscan"}


@st.cache_data(show_spinner=False)
def load_sources():
    return (
        load_activity(DATA_ORIGINAL/"dailyActivity_merged.csv"),
        load_sleep(DATA_ORIGINAL/"sleepDay_merged.csv"),
    )


@st.cache_data(show_spinner=False)
def cached_search_k(X, ks, n_init, seed):
    return search_k(np.asarray(X),ks,n_init,seed)


@st.cache_data(show_spinner=False)
def cached_k_distances(X, min_samples):
    return k_distances(np.asarray(X),min_samples)


@st.cache_data(show_spinner=False)
def cached_dbscan_grid(X, eps_values, min_samples_values):
    return explore_dbscan(np.asarray(X),eps_values,min_samples_values)


@st.cache_data(show_spinner=False)
def cached_pca(X, threshold):
    return fit_pca(np.asarray(X),variance_threshold=float(threshold))


def _mapping_for(df):
    source_id=dataframe_fingerprint(df[["Id","Date"]])
    if st.session_state.get("pseudonym_source_id") != source_id:
        st.session_state["participant_map"]=participant_mapping(df["Id"])
        st.session_state["pseudonym_source_id"]=source_id
    return st.session_state["participant_map"]


def _prepared(df, default_features):
    features=st.session_state.get("feature_selection",default_features)
    missing=st.session_state.get("missing_strategy","median")
    outliers=st.session_state.get("outlier_strategy","keep")
    expected=preprocessing_fingerprint(df,features,missing,outliers)
    current=st.session_state.get("prep")
    if current is None or current.get("configuration_id") != expected:
        current=prepare_features(df,list(features),missing,outliers)
        current["configuration_id"]=expected
        st.session_state["prep"]=current
    return current


def _save_prepared(df,features,missing,outliers):
    prep=prepare_features(df,list(features),missing,outliers)
    prep["configuration_id"]=preprocessing_fingerprint(df,features,missing,outliers)
    st.session_state["feature_selection"]=list(features)
    st.session_state["missing_strategy"]=missing
    st.session_state["outlier_strategy"]=outliers
    st.session_state["prep"]=prep
    return prep


def _results():
    if "model_results" not in st.session_state:
        st.session_state["model_results"]={}
    return st.session_state["model_results"]


def _save_model(name, model, labels, metrics, config, prep, pca_id=None, **extra):
    config_id=result_fingerprint(prep["configuration_id"],name,config,pca_id)
    result=register_result(model,labels,metrics,config,config_id,
        preprocessing_id=prep["configuration_id"],pca_id=pca_id,
        representation="espacio PCA" if pca_id else "espacio original escalado",**extra)
    _results()[name]=result
    return result


def _current_configs(prep):
    pca_id=pca_fingerprint(prep["configuration_id"],st.session_state.get("pca_threshold",.90))
    return {
        "K-Means":result_fingerprint(prep["configuration_id"],"K-Means",{"k":int(st.session_state.get("km_k",3)),"n_init":int(st.session_state.get("km_n_init",20)),"random_state":int(st.session_state.get("km_seed",42))}),
        "DBSCAN":result_fingerprint(prep["configuration_id"],"DBSCAN",{"eps":float(st.session_state.get("db_eps",.8)),"min_samples":int(st.session_state.get("db_min_samples",5))}),
        "PCA + K-Means":result_fingerprint(prep["configuration_id"],"PCA + K-Means",{"k":int(st.session_state.get("combined_k",3)),"n_init":20,"random_state":42},pca_id),
        "PCA + DBSCAN":result_fingerprint(prep["configuration_id"],"PCA + DBSCAN",{"eps":float(st.session_state.get("combined_eps",.8)),"min_samples":int(st.session_state.get("combined_min_samples",5))},pca_id),
    }


def _valid_results(prep):
    expected=_current_configs(prep); rows={}
    expected_pca=pca_fingerprint(prep["configuration_id"],st.session_state.get("pca_threshold",.90))
    current_pca=st.session_state.get("pca_result")
    for display_name,key in MODEL_KEYS.items():
        result=_results().get(key)
        if result_matches(result,expected[display_name],len(prep["scaled"])) and result.get("preprocessing_id")==prep["configuration_id"]:
            if display_name.startswith("PCA +") and (result.get("pca_id")!=expected_pca or not current_pca or current_pca.get("configuration_id")!=expected_pca or current_pca.get("preprocessing_id")!=prep["configuration_id"]):
                continue
            rows[display_name]=result
    return rows


def _artifact_warning(prep,name):
    result=_results().get(MODEL_KEYS[name])
    if result is None:
        st.info("Aún no hay un resultado guardado para esta alternativa.")
    elif result.get("preprocessing_id") != prep["configuration_id"]:
        st.warning("El preprocesamiento cambió. Este resultado pertenece a otra configuración y queda excluido de comparación y perfiles.")
    elif result.get("configuration_id") != _current_configs(prep)[name]:
        st.warning("Los parámetros visibles difieren del último entrenamiento. Vuelva a ejecutar el modelo para habilitar sus resultados.")


def run():
    st.set_page_config(page_title="FitSense AI",page_icon="🏃",layout="wide")
    st.sidebar.title("FitSense AI")
    page=st.sidebar.radio("Navegación",NAV)
    st.sidebar.caption("Perfiles de actividad física con dispositivos vestibles")
    try:
        activity,sleep=load_sources()
        df,a0,s0,audit,_,_,sleep_duplicates=integrate(activity,sleep)
    except Exception as exc:
        st.error(f"No se pudieron cargar e integrar los datos: {exc}"); st.stop()
    mapping=_mapping_for(df)
    numerical=df.select_dtypes(include="number").columns.tolist()
    defaults=[c for c in DEFAULT_FEATURES if c in numerical] or numerical[:min(6,len(numerical))]
    st.title(page)

    if page=="Inicio":
        st.subheader("Perfiles de actividad física con dispositivos vestibles")
        st.write("FitSense AI compara K-Means, DBSCAN, PCA + K-Means y PCA + DBSCAN para describir rutinas diarias de actividad y descanso. No realiza diagnósticos médicos.")
        st.markdown("**Pregunta:** ¿Qué rutinas de actividad física y descanso se pueden diferenciar y qué días presentan comportamientos atípicos?")
        c1,c2,c3,c4=st.columns(4); c1.metric("Días integrados",f"{len(df):,}"); c2.metric("Participantes",df.Id.nunique()); c3.metric("Actividad",f"{len(activity):,} filas"); c4.metric("Sueño",f"{len(sleep):,} filas")
        st.info("El aprendizaje no supervisado descubre estructura sin etiquetas objetivo; no predice una clase conocida.")
        st.dataframe(audit,width="stretch",hide_index=True)

    elif page=="Dataset y calidad":
        st.write("Fuentes originales CSV. Id se conserva internamente y se reemplaza por un pseudónimo en las tablas públicas.")
        m1,m2,m3=st.columns(3); m1.metric("Actividad",f"{activity.shape[0]} × {activity.shape[1]-1}"); m2.metric("Sueño",f"{sleep.shape[0]} × {sleep.shape[1]-1}"); m3.metric("Duplicados exactos de sueño descartados",len(sleep_duplicates))
        structure=pd.DataFrame({"variable":[c for c in df.columns if c!="Id"],"tipo":[str(df[c].dtype) for c in df.columns if c!="Id"],"faltantes":[int(df[c].isna().sum()) for c in df.columns if c!="Id"],"faltantes (%)":[round(100*df[c].isna().mean(),2) for c in df.columns if c!="Id"]})
        st.subheader("Estructura y calidad persona-día"); st.dataframe(structure,width="stretch",hide_index=True)
        st.caption(f"Duplicados exactos de actividad: {activity.duplicated(subset=[c for c in activity.columns if c!='ActivityDate']).sum()}. Duplicados persona-día integrados: {df.duplicated(['Id','Date']).sum()}.")
        st.dataframe(audit,width="stretch",hide_index=True); st.subheader("Diccionario de datos"); st.dataframe(dictionary_frame(),width="stretch",hide_index=True)
        upload_a=st.file_uploader("CSV alternativo de actividad",type="csv",key="ua"); upload_s=st.file_uploader("CSV alternativo de sueño",type="csv",key="us")
        if upload_a and upload_s:
            try:
                aa=pd.read_csv(upload_a,dtype={"Id":"string"}); ss=pd.read_csv(upload_s,dtype={"Id":"string"})
                merged,*_=integrate(aa,ss); alt_map=participant_mapping(merged.Id)
                st.success(f"Integración alternativa: {len(merged)} observaciones.")
                st.dataframe(pseudonymize_frame(merged.head(30),alt_map),width="stretch",hide_index=True)
            except Exception as exc: st.error(f"No se pudo integrar: {exc}")
        st.subheader("Vista integrada pseudonimizada"); st.dataframe(pseudonymize_frame(df.head(50),mapping),width="stretch",hide_index=True)

    elif page=="Exploración de datos":
        st.write("Hallazgos descriptivos; correlación no implica causalidad.")
        for item in findings(df): st.markdown(f"- {item}")
        st.subheader("Estadísticas descriptivas"); st.dataframe(descriptive_summary(df).round(2),width="stretch")
        cols=st.multiselect("Variables numéricas",numerical,default=defaults[:4],key="eda_vars")
        if cols:
            tab1,tab2=st.tabs(["Distribuciones","Correlación"])
            with tab1:
                col=st.selectbox("Variable",cols,key="eda_hist_var"); st.plotly_chart(px.histogram(df,x=col,marginal="box",title=f"Distribución de {col}"),width="stretch")
            with tab2: st.plotly_chart(px.imshow(correlation_matrix(df[cols]),text_auto=".2f",color_continuous_scale="RdBu_r",zmin=-1,zmax=1),width="stretch")
        st.subheader("Valores faltantes"); st.dataframe(missing_summary(df),width="stretch")

    elif page=="Preprocesamiento":
        features=st.multiselect("Variables para clustering (sin Id ni fecha)",numerical,default=st.session_state.get("feature_selection",defaults),key="feature_selection_widget")
        missing=st.selectbox("Tratamiento de faltantes",["median","mean","drop","zero"],index=["median","mean","drop","zero"].index(st.session_state.get("missing_strategy","median")),format_func=lambda x:{"median":"Imputar mediana","mean":"Imputar media","drop":"Excluir filas incompletas","zero":"Imputar cero (explícito)"}[x],key="missing_strategy_widget")
        outlier=st.selectbox("Valores extremos",["keep","iqr","winsorize"],format_func=lambda x:{"keep":"Conservar","iqr":"Limitar a límites de Tukey (1.5×IQR)","winsorize":"Winsorizar en percentiles 5 y 95"}[x],key="outlier_strategy_widget")
        prep=_save_prepared(df,features,missing,outlier)
        exports=feature_export_frames(prep["metadata"],prep["raw"],prep["scaled"],mapping)
        raw_export=exports["antes_de_escalar"]; scaled_export=exports["estandarizada"]
        st.metric("Observaciones preparadas",len(prep["scaled"]))
        st.write("Id pseudónimo y fecha son metadatos asociados; no se incluyen en la matriz de características. StandardScaler centra las características y las lleva a desviación estándar poblacional aproximadamente uno.")
        tab_raw,tab_scaled=st.tabs(["Características antes de escalar","Matriz estandarizada usada por los modelos"])
        with tab_raw:
            st.dataframe(raw_export.head(30),width="stretch",hide_index=True)
            st.download_button("Descargar características antes de escalar",raw_export.to_csv(index=False).encode("utf-8-sig"),"caracteristicas_pre_escalado.csv","text/csv")
        with tab_scaled:
            st.dataframe(scaled_export.head(30),width="stretch",hide_index=True)
            st.download_button("Descargar matriz estandarizada",scaled_export.to_csv(index=False).encode("utf-8-sig"),"preprocessed_features.csv","text/csv")
            st.caption("Media observada: aproximadamente 0. Desviación estándar poblacional: aproximadamente 1.")
            st.dataframe(pd.DataFrame({"media":prep["scaled"].mean(),"desviación poblacional":prep["scaled"].std(ddof=0)}).round(4),width="stretch")
        config={"variables":features,"faltantes":missing,"extremos":outlier,"escalado":"StandardScaler","configuracion_id":prep["configuration_id"]}
        st.json(config); st.download_button("Descargar registro de transformaciones",json.dumps(config,ensure_ascii=False,indent=2).encode(),"preprocessing_config.json","application/json")
        if prep["bounds"]: st.json(prep["bounds"])

    elif page=="K-Means":
        prep=_prepared(df,defaults); X=prep["scaled"].to_numpy(); n=len(X); max_k=min(10,n-1)
        if max_k<2: st.warning("Se requieren al menos tres observaciones."); return
        k=st.slider("K elegido por el analista",2,max_k,min(3,max_k),key="km_k")
        ninit=st.number_input("n_init",10,100,20,key="km_n_init"); seed=st.number_input("Semilla",0,100000,42,key="km_seed")
        if st.button("Explorar K=2…10",key="run_k_search"):
            ks=tuple(range(2,max_k+1)); result=cached_search_k(X,ks,int(ninit),int(seed))
            st.session_state["k_search"]={"prep_id":prep["configuration_id"],"parameters":{"K":ks,"n_init":int(ninit),"seed":int(seed)},"table":result}
        search=st.session_state.get("k_search")
        if search and search["prep_id"]==prep["configuration_id"]:
            st.caption(f"Exploración guardada: K={search['parameters']['K'][0]}–{search['parameters']['K'][-1]}, n_init={search['parameters']['n_init']}, semilla={search['parameters']['seed']}.")
            st.plotly_chart(px.line(search["table"],x="k",y="inertia",markers=True,title="Método del codo"),width="stretch")
            st.plotly_chart(px.line(search["table"],x="k",y="silhouette",markers=True,title="Silhouette por K"),width="stretch")
            st.dataframe(search["table"].round(4),width="stretch",hide_index=True)
        elif search: st.warning("La búsqueda guardada pertenece a otro preprocesamiento. Ejecute de nuevo la exploración.")
        if st.button("Ejecutar K-Means",key="fit_kmeans"):
            model,labels=fit_kmeans(X,k,int(ninit),int(seed)); metrics=clustering_metrics(X,labels,noise_label=None)
            config={"k":int(k),"n_init":int(ninit),"random_state":int(seed)}
            _save_model("kmeans",model,labels,metrics,config,prep)
            st.session_state.pop("stability_kmeans",None)
        expected=_current_configs(prep)["K-Means"]; result=_results().get("kmeans")
        if result_matches(result,expected,n) and result.get("preprocessing_id")==prep["configuration_id"]:
            labels=result["labels"]; st.json(result["metrics"]); st.caption(f"Configuración {result['configuration_id'][:12]} · ejecutado {result['executed_at']}")
            st.bar_chart(pd.Series(labels).value_counts().sort_index())
            if X.shape[1]>1: st.plotly_chart(px.scatter(x=X[:,0],y=X[:,1],color=labels.astype(str),title="K-Means · dos primeras variables escaladas",labels={"x":prep["features"][0],"y":prep["features"][1]}),width="stretch")
            centers=result["model"].cluster_centers_; raw_centers=prep["scaler"].inverse_transform(centers)
            st.dataframe(pd.DataFrame(raw_centers,columns=prep["features"]).assign(cluster=range(len(centers))),width="stretch")
            if st.button("Evaluar estabilidad K-Means",key="stability_kmeans_button"):
                summary,ari=assess_kmeans_stability(X,k,seeds=(11,23,42,77,101),n_init=int(ninit))
                st.session_state["stability_kmeans"]={"configuration_id":expected,"summary":summary,"ari":ari}
                st.session_state.setdefault("stability_results",{})["K-Means"]={"configuration_id":expected,"summary":f"ARI semillas medio={ari['ari_mean']:.3f} (DE={ari['ari_std']:.3f})"}
            stability=st.session_state.get("stability_kmeans")
            if stability and stability["configuration_id"]==expected:
                st.subheader("Estabilidad por semillas"); st.dataframe(stability["summary"],width="stretch"); st.write(stability["ari"])
        else: _artifact_warning(prep,"K-Means")

    elif page=="DBSCAN":
        prep=_prepared(df,defaults); X=prep["scaled"].to_numpy(); n=len(X)
        ms_curve=st.slider("min_samples para curva k-distances",2,min(30,n),5,key="db_curve_min")
        kd=cached_k_distances(X,int(ms_curve)); st.plotly_chart(px.line(x=np.arange(len(kd)),y=kd,title="Curva k-distances",labels={"x":"Observaciones ordenadas","y":"Distancia k-ésimo vecino"}),width="stretch")
        eps=st.number_input("eps",min_value=.01,max_value=20.,value=.8,step=.05,key="db_eps"); min_samples=st.slider("min_samples",2,min(30,n),5,key="db_min_samples")
        if st.button("Explorar grilla DBSCAN",key="db_grid_button"):
            eps_values=tuple(np.round(np.linspace(max(.1,eps*.5),eps*1.5,6),4))
            ms_values=tuple(sorted({max(2,min_samples-1),min_samples,min(30,min_samples+1)}))
            grid=cached_dbscan_grid(X,eps_values,ms_values)
            st.session_state["db_grid"]={"prep_id":prep["configuration_id"],"parameters":{"eps":eps_values,"min_samples":ms_values},"table":grid}
        grid=st.session_state.get("db_grid")
        if grid and grid["prep_id"]==prep["configuration_id"]:
            st.caption(f"Grilla guardada: eps={grid['parameters']['eps']}; min_samples={grid['parameters']['min_samples']}.")
            st.dataframe(grid["table"].round(3),width="stretch",hide_index=True)
        elif grid: st.warning("La grilla guardada pertenece a otro preprocesamiento. Explórela nuevamente.")
        if st.button("Ejecutar DBSCAN",key="fit_dbscan"):
            model,labels,metrics,kinds=fit_dbscan(X,float(eps),int(min_samples))
            config={"eps":float(eps),"min_samples":int(min_samples)}
            _save_model("dbscan",model,labels,metrics,config,prep,kinds=kinds)
            st.session_state.pop("stability_dbscan",None)
        expected=_current_configs(prep)["DBSCAN"]; result=_results().get("dbscan")
        if result_matches(result,expected,n) and result.get("preprocessing_id")==prep["configuration_id"]:
            labels=result["labels"]; st.json(result["metrics"]); st.write("Ruido = -1; las métricas internas excluyen ruido.")
            st.bar_chart(pd.Series(labels).value_counts().sort_index()); st.write("Observaciones por tipo:",pd.Series(result["kinds"]).value_counts().to_dict())
            if X.shape[1]>1: st.plotly_chart(px.scatter(x=X[:,0],y=X[:,1],color=labels.astype(str),title="DBSCAN · dos primeras variables escaladas",labels={"x":prep["features"][0],"y":prep["features"][1]}),width="stretch")
            if st.button("Evaluar estabilidad DBSCAN",key="stability_dbscan_button"):
                eps_values=[max(.1,eps*.75),eps,eps*1.25]
                ms_values=sorted({max(2,min_samples-1),min_samples,min(30,min_samples+1)})
                configs,pairs=assess_dbscan_stability(X,eps_values,ms_values)
                st.session_state["stability_dbscan"]={"configuration_id":expected,"configs":configs,"pairs":pairs}
                st.session_state.setdefault("stability_results",{})["DBSCAN"]={"configuration_id":expected,"summary":f"ARI incluyendo ruido medio={pairs['ari_including_noise'].mean():.3f}"}
            stability=st.session_state.get("stability_dbscan")
            if stability and stability["configuration_id"]==expected:
                st.dataframe(stability["configs"],width="stretch"); st.dataframe(stability["pairs"],width="stretch")
        else: _artifact_warning(prep,"DBSCAN")

    elif page=="Análisis PCA":
        prep=_prepared(df,defaults); X=prep["scaled"].to_numpy()
        threshold=st.slider("Varianza objetivo",.5,.99,.9,.01,key="pca_threshold")
        expected=pca_fingerprint(prep["configuration_id"],threshold); pca_result=st.session_state.get("pca_result")
        if st.button("Ajustar PCA",key="fit_pca_button"):
            model,scores,info=cached_pca(X,float(threshold))
            pca_result={"model":model,"scores":scores,"info":info,"configuration_id":expected,"preprocessing_id":prep["configuration_id"],"threshold":float(threshold)}
            st.session_state["pca_result"]=pca_result
        if pca_result and pca_result["configuration_id"]==expected and pca_result["preprocessing_id"]==prep["configuration_id"]:
            model,scores,info=pca_result["model"],pca_result["scores"],pca_result["info"]
            c1,c2,c3=st.columns(3); c1.metric("Componentes",info["n_components"]); c2.metric("Varianza retenida",f"{info['variance_retained']:.1%}"); c3.metric("RMSE estandarizado",f"{info['reconstruction_rmse_scaled']:.3f}")
            st.plotly_chart(px.bar(x=np.arange(1,len(info["explained_variance_ratio"])+1),y=info["explained_variance_ratio"],labels={"x":"Componente","y":"Varianza explicada"},title="Scree plot"),width="stretch")
            st.plotly_chart(px.line(x=np.arange(1,len(info["cumulative_variance"])+1),y=info["cumulative_variance"],markers=True,title="Varianza acumulada"),width="stretch")
            st.dataframe(info["loadings"].set_axis(prep["features"],axis=0),width="stretch")
            if scores.shape[1]>=2: st.plotly_chart(px.scatter(x=scores[:,0],y=scores[:,1],labels={"x":"PC1","y":"PC2"},title="Observaciones en PC1 frente a PC2"),width="stretch")
        elif pca_result: st.warning("PCA guardado pertenece a otra configuración. Ajuste PCA para el umbral y preprocesamiento actuales.")
        else: st.info("Ajuste PCA para calcular componentes, varianza, cargas y reconstrucción.")

    elif page=="Modelos combinados":
        prep=_prepared(df,defaults); X=prep["scaled"].to_numpy()
        threshold=st.slider("Varianza objetivo PCA compartido",.5,.99,.9,.01,key="pca_threshold")
        pca_id=pca_fingerprint(prep["configuration_id"],threshold)
        k=st.slider("K para PCA + K-Means",2,min(10,len(X)-1),min(3,len(X)-1),key="combined_k")
        eps=st.number_input("eps para PCA + DBSCAN",.01,20.,.8,.05,key="combined_eps"); ms=st.slider("min_samples combinado",2,min(30,len(X)),5,key="combined_min_samples")
        pca_result=st.session_state.get("pca_result")
        pca_valid=bool(pca_result and pca_result["configuration_id"]==pca_id and pca_result["preprocessing_id"]==prep["configuration_id"])
        if not pca_valid: st.info("Los dos modelos combinados usarán exactamente la misma transformación PCA calculada al ejecutar el botón.")
        if st.button("Ejecutar ambos modelos combinados",key="fit_combined"):
            if not pca_valid:
                pca_model,scores,info=cached_pca(X,float(threshold))
                pca_result={"model":pca_model,"scores":scores,"info":info,"configuration_id":pca_id,"preprocessing_id":prep["configuration_id"],"threshold":float(threshold)}
                st.session_state["pca_result"]=pca_result
            pca_model,scores,info=pca_result["model"],pca_result["scores"],pca_result["info"]
            km,lk=fit_pca_kmeans(scores,k); km_metrics=clustering_metrics(X,lk,noise_label=None)
            db,ld,db_metrics,kinds=fit_pca_dbscan(scores,eps,ms); db_metrics_original=clustering_metrics(X,ld)
            _save_model("pca_kmeans",km,lk,km_metrics,{"k":int(k),"n_init":20,"random_state":42},prep,pca_id)
            _save_model("pca_dbscan",db,ld,db_metrics_original,{"eps":float(eps),"min_samples":int(ms)},prep,pca_id,kinds=kinds,metrics_pca=db_metrics)
            configs_now=_current_configs(prep)
            st.session_state["pca_diagnostic_metrics"]={
                "configuration_ids":{"PCA + K-Means":configs_now["PCA + K-Means"],"PCA + DBSCAN":configs_now["PCA + DBSCAN"]},
                "metrics":{"pca_kmeans":clustering_metrics(scores,lk,noise_label=None),"pca_dbscan":db_metrics},
            }
        current=_valid_results(prep)
        if "PCA + K-Means" in current:
            result=current["PCA + K-Means"]; scores=pca_result["scores"]
            st.plotly_chart(px.scatter(x=scores[:,0],y=scores[:,1],color=result["labels"].astype(str),title="PCA + K-Means",labels={"x":"PC1","y":"PC2"}),width="stretch")
            centers=reconstruct_centroids(result["model"],pca_result["model"],prep["scaler"])
            st.subheader("Centroides reconstruidos a unidades originales"); st.dataframe(pd.DataFrame(centers,columns=prep["features"]).assign(cluster=range(len(centers))),width="stretch")
            st.json(result["metrics"])
        if "PCA + DBSCAN" in current:
            result=current["PCA + DBSCAN"]; scores=pca_result["scores"]
            st.plotly_chart(px.scatter(x=scores[:,0],y=scores[:,1],color=result["labels"].astype(str),title="PCA + DBSCAN",labels={"x":"PC1","y":"PC2"}),width="stretch")
            st.json(result["metrics"])
        if pca_result and pca_result.get("preprocessing_id")==prep["configuration_id"]:
            st.caption(f"PCA compartido: {pca_result['info']['n_components']} componentes, {pca_result['info']['variance_retained']:.1%} de varianza.")
        st.info("Las métricas principales de ambos modelos combinados se calculan sobre la representación original escalada; PCA se muestra como diagnóstico secundario.")

    elif page=="Comparación y estabilidad":
        prep=_prepared(df,defaults); valid=_valid_results(prep)
        if not valid:
            st.info("Ejecute modelos con la configuración de preprocesamiento actual. Los resultados de otra configuración no se compararán.")
        else:
            labels={name:result["labels"] for name,result in valid.items()}
            assigned=compare_models(prep["scaled"].to_numpy(),labels)
            st.subheader("A. Métricas sobre observaciones asignadas")
            st.caption("Los soportes difieren: DBSCAN excluye ruido. No ordene las métricas como si se hubieran calculado sobre las mismas observaciones.")
            st.dataframe(assigned.round(4),width="stretch",hide_index=True)
            st.download_button("Descargar comparación sobre asignados",assigned.to_csv(index=False).encode("utf-8-sig"),"comparacion_asignados.csv","text/csv")
            pca_result=st.session_state.get("pca_result")
            combined_valid=all(name in valid for name in MODEL_KEYS)
            if combined_valid:
                common=common_assigned_comparison(prep["scaled"].to_numpy(),labels)
                st.subheader("B. Las cuatro alternativas sobre soporte común")
                st.write(f"Observaciones no consideradas ruido por ambos DBSCAN: **{common['n_common']}** de {len(prep['scaled'])}.")
                if common["valid"]:
                    st.dataframe(common["table"].round(4),width="stretch",hide_index=True)
                    st.download_button("Descargar comparación común",common["table"].to_csv(index=False).encode("utf-8-sig"),"comparacion_soporte_comun.csv","text/csv")
                else: st.warning(f"Comparación común no disponible: {common['reason']}")
                expected_pca=pca_fingerprint(prep["configuration_id"],st.session_state.get("pca_threshold",.90))
                diagnostic=st.session_state.get("pca_diagnostic_metrics")
                diagnostic_current=bool(diagnostic and all(
                    name in valid and diagnostic["configuration_ids"].get(name)==valid[name]["configuration_id"]
                    for name in ("PCA + K-Means","PCA + DBSCAN")
                ))
                if pca_result and pca_result["configuration_id"]==expected_pca and diagnostic_current:
                    st.subheader("Diagnóstico secundario en PCA")
                    st.json(diagnostic["metrics"])
            else: st.info("La comparación común requiere resultados vigentes de las cuatro alternativas.")
        stale=[name for name in MODEL_KEYS if name not in valid and _results().get(MODEL_KEYS[name])]
        if stale: st.warning("Resultados excluidos por configuración anterior o parámetros modificados: "+", ".join(stale))

    else:
        _render_profiles_and_decision(df,defaults,mapping)


def _render_profiles_and_decision(df,defaults,mapping):
    prep=_prepared(df,defaults); valid=_valid_results(prep)
    if not valid:
        st.info("Ejecute modelos con el preprocesamiento actual para generar perfiles y evidencia de decisión.")
        return
    labels={name:result["labels"] for name,result in valid.items()}
    options=list(valid)
    selected=st.selectbox("Modelo para perfiles",options,key="profile_model")
    result=valid[selected]
    profiles=profile_clusters(prep["raw"],result["labels"])
    st.subheader(f"Perfiles de {selected}"); st.dataframe(profiles,width="stretch")
    feature_deviations=prep["scaled"].copy(); feature_deviations["cluster"]=result["labels"]
    for cluster,means in feature_deviations.groupby("cluster").mean(numeric_only=True).iterrows():
        above=means[means>=.25].sort_values(ascending=False).head(3).index.tolist(); below=means[means<=-.25].sort_values().head(3).index.tolist()
        st.markdown(f"**Grupo {cluster}:** sobre la media global en {', '.join(above) if above else 'ninguna variable supera el umbral descriptivo'}; bajo la media global en {', '.join(below) if below else 'ninguna variable supera el umbral descriptivo'}. Valide las diferencias frente a unidades y medianas.")
    public_meta=pseudonymize_frame(prep["metadata"],mapping)
    st.subheader("Días potencialmente inusuales")
    percentile=st.slider("Percentil de distancia a centroides para K-Means",80,100,95,key="outlier_percentile")
    if selected in {"K-Means","PCA + K-Means"}:
        candidates,threshold=kmeans_outlier_candidates(prep["scaled"].to_numpy(),result["labels"],public_meta,percentile)
        st.caption(f"Umbral empírico de distancia: percentil {percentile} = {threshold:.4f} en el espacio original escalado.")
        if candidates.empty: st.info("No hay observaciones por encima del umbral seleccionado.")
        else: st.dataframe(candidates,width="stretch",hide_index=True)
    elif selected in {"DBSCAN","PCA + DBSCAN"}:
        noise=outlier_days(public_meta,result["labels"])
        if noise.empty: st.info("No se identificaron observaciones con etiqueta de ruido (−1).")
        else: st.dataframe(noise,width="stretch",hide_index=True)
    st.caption("Son candidatos a días inusuales; no implican error confirmado ni interpretación médica.")

    st.divider(); st.header("Comparación de evidencia y decisión humana")
    if len(valid)==len(MODEL_KEYS):
        assigned=compare_models(prep["scaled"].to_numpy(),labels)
        common=common_assigned_comparison(prep["scaled"].to_numpy(),labels)
        metrics_table=common["table"] if common["valid"] else assigned
    else:
        metrics_table=compare_models(prep["scaled"].to_numpy(),labels)
    stability={}
    for name in valid:
        artifact=st.session_state.get("stability_results",{}).get(name)
        if artifact and artifact.get("configuration_id")==valid[name]["configuration_id"]:
            stability[name]=artifact["summary"]
    if st.button("Evaluar estabilidad de resultados vigentes",key="decision_stability"):
        updated=st.session_state.get("stability_results",{}).copy()
        pca_result=st.session_state.get("pca_result")
        for name,result in valid.items():
            labels_for_model=result["labels"]
            if name=="K-Means":
                _,ari=assess_kmeans_stability(prep["scaled"].to_numpy(),int(result["configuration"]["k"]),seeds=(11,23,42,77,101),n_init=int(result["configuration"]["n_init"]))
                summary=f"ARI semillas medio={ari['ari_mean']:.3f} (DE={ari['ari_std']:.3f})"
            elif name=="PCA + K-Means":
                scores=pca_result["scores"]
                _,ari=assess_kmeans_stability(scores,int(result["configuration"]["k"]),seeds=(11,23,42,77,101),n_init=20)
                summary=f"ARI semillas medio={ari['ari_mean']:.3f} (DE={ari['ari_std']:.3f})"
            else:
                X=prep["scaled"].to_numpy() if name=="DBSCAN" else pca_result["scores"]
                cfg=result["configuration"]; eps=float(cfg["eps"])
                ms=int(cfg["min_samples"])
                configs,pairs=assess_dbscan_stability(X,[max(.05,eps*.75),eps,eps*1.25],sorted({max(2,ms-1),ms,min(30,ms+1)}))
                summary=f"ARI incluyendo ruido medio={pairs['ari_including_noise'].mean():.3f}; ruido={configs['noise_pct'].min():.1f}–{configs['noise_pct'].max():.1f}%"
            updated[name]={"configuration_id":result["configuration_id"],"summary":summary}
        st.session_state["stability_results"]=updated
        stability={name:updated[name]["summary"] for name in valid}
    pca_result=st.session_state.get("pca_result")
    pca_variance=pca_result["info"]["variance_retained"] if pca_result and pca_result.get("preprocessing_id")==prep["configuration_id"] else None
    evidence=evidence_matrix(metrics_table,stability,pca_variance)
    display_cols=[c for c in ["modelo","n_clusters","n_evaluated","n_noise","coverage_pct","silhouette","davies_bouldin","calinski_harabasz","estabilidad","varianza_pca","ventajas","limitaciones"] if c in evidence]
    st.dataframe(evidence[display_cols],width="stretch",hide_index=True)
    st.markdown("**Limitaciones:** 33 participantes de una muestra de conveniencia; periodo corto de un mes; registros de sueño incompletos respecto a actividad; medidas proceden de dispositivos y no son diagnósticas; DBSCAN cubre solo las filas asignadas; PCA pierde varianza; la generalización a otras poblaciones no está demostrada.")
    st.markdown("**Lectura de alternativas:** K-Means asigna todos los días y facilita centroides, pero requiere K y es sensible a extremos. DBSCAN marca ruido sin fijar K, pero es sensible a `eps`/`min_samples` y deja soporte variable. Las variantes PCA reducen dimensiones con pérdida de varianza y requieren interpretar cargas.")
    chosen=st.selectbox("Alternativa que el equipo considera más adecuada",["Sin definir",*MODEL_KEYS],key="decision_choice")
    arguments=st.text_area("Argumentos académicos de la elección",key="decision_arguments")
    limitations=st.text_area("Limitaciones específicas de la decisión",key="decision_limitations")
    if st.button("Guardar justificación del equipo",key="save_decision"):
        decision=decision_guidance(evidence,chosen,arguments,limitations)
        decision["configuration_id"]=fingerprint({name:result["configuration_id"] for name,result in valid.items()})
        st.session_state["decision_log"]=decision
    decision=st.session_state.get("decision_log")
    current_decision_id=fingerprint({name:result["configuration_id"] for name,result in valid.items()})
    if decision and decision.get("configuration_id")==current_decision_id:
        st.info(decision["note"]); st.write("Alternativa documentada:",decision["selected_model"])
        st.download_button("Descargar evidencia de decisión",evidence.to_csv(index=False).encode("utf-8-sig"),"matriz_evidencia.csv","text/csv")
    elif decision:
        st.warning("La justificación guardada corresponde a otra configuración de modelos. Revísela y guárdela nuevamente.")
