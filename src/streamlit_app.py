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
from .model_comparison import compare_models, common_assigned_comparison, individual_comparison
from .kmeans_stability import assess_kmeans_stability
from .dbscan_stability import assess_dbscan_stability
from .profiling import profile_details, outlier_days, kmeans_outlier_candidates
from .decision_support import evidence_matrix, decision_guidance
from .result_tracking import (
    dataframe_fingerprint, participant_mapping, pseudonymize_frame,
    preprocessing_fingerprint, pca_fingerprint, result_fingerprint,
    register_result, result_matches, fingerprint, feature_export_frames,
)

NAV=["Inicio","Dataset y calidad","Exploración de datos","Preprocesamiento","K-Means","DBSCAN","Análisis PCA","Modelos combinados","Comparación y estabilidad","Perfiles y conclusiones"]
MODEL_KEYS={"K-Means":"kmeans","DBSCAN":"dbscan","PCA + K-Means":"pca_kmeans","PCA + DBSCAN":"pca_dbscan"}
MODEL_DISPLAY_NAMES={value:key for key,value in MODEL_KEYS.items()}


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
    config_id=result_fingerprint(prep["configuration_id"],MODEL_DISPLAY_NAMES.get(name,name),config,pca_id)
    result=register_result(model,labels,metrics,config,config_id,
        preprocessing_id=prep["configuration_id"],pca_id=pca_id,
        representation="espacio PCA" if pca_id else "espacio original escalado",**extra)
    _results()[name]=result
    return result


def _current_configs(prep):
    active=st.session_state.get("active_model_parameters",{})
    km=active.get("K-Means",{})
    db=active.get("DBSCAN",{})
    pkm=active.get("PCA + K-Means",{})
    pdb=active.get("PCA + DBSCAN",{})
    threshold=float(active.get("pca_threshold",st.session_state.get("pca_threshold",.90)))
    pca_id=pca_fingerprint(prep["configuration_id"],threshold)
    return {
        "K-Means":result_fingerprint(prep["configuration_id"],"K-Means",{"k":int(km.get("k",3)),"n_init":int(km.get("n_init",20)),"random_state":int(km.get("random_state",42))}),
        "DBSCAN":result_fingerprint(prep["configuration_id"],"DBSCAN",{"eps":float(db.get("eps",.8)),"min_samples":int(db.get("min_samples",5))}),
        "PCA + K-Means":result_fingerprint(prep["configuration_id"],"PCA + K-Means",{"k":int(pkm.get("k",4)),"n_init":int(pkm.get("n_init",20)),"random_state":int(pkm.get("random_state",42))},pca_id),
        "PCA + DBSCAN":result_fingerprint(prep["configuration_id"],"PCA + DBSCAN",{"eps":float(pdb.get("eps",.8)),"min_samples":int(pdb.get("min_samples",5))},pca_id),
    }


def _valid_results(prep):
    expected=_current_configs(prep); rows={}
    expected_pca=pca_fingerprint(prep["configuration_id"],st.session_state.get("active_model_parameters",{}).get("pca_threshold",.90))
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


def _stability_labels(valid):
    labels={}
    stored=st.session_state.get("stability_results",{})
    for name,result in valid.items():
        artifact=stored.get(name)
        if artifact and artifact.get("configuration_id")==result["configuration_id"]:
            method="Semillas" if name.endswith("K-Means") else "Variación de parámetros"
            labels[name]=f"{method}: {artifact['summary']}"
    return labels


def _evaluate_stability(prep, valid):
    updated=st.session_state.get("stability_results",{}).copy()
    pca_result=st.session_state.get("pca_result")
    for name,artifact in valid.items():
        config=artifact["configuration"]
        if name in {"K-Means","PCA + K-Means"}:
            X=prep["scaled"].to_numpy() if name=="K-Means" else pca_result["scores"]
            _,ari=assess_kmeans_stability(X,int(config["k"]),seeds=(11,23,42,77,101),n_init=int(config["n_init"]))
            summary=f"ARI medio entre semillas={ari['ari_mean']:.3f}; DE={ari['ari_std']:.3f}"
        else:
            X=prep["scaled"].to_numpy() if name=="DBSCAN" else pca_result["scores"]
            eps=float(config["eps"]); min_samples=int(config["min_samples"])
            runs,pairs=assess_dbscan_stability(X,[max(.05,eps*.75),eps,eps*1.25],
                sorted({max(2,min_samples-1),min_samples,min(30,min_samples+1)}))
            ari=pairs["ari_including_noise"].mean() if not pairs.empty else float("nan")
            summary=(f"ARI medio al variar parámetros={ari:.3f}; "
                f"ruido entre {runs['noise_pct'].min():.1f}% y {runs['noise_pct'].max():.1f}%")
        updated[name]={"configuration_id":artifact["configuration_id"],"summary":summary}
    st.session_state["stability_results"]=updated
    return _stability_labels(valid)


def _metric_cards(metrics, extra=None):
    values=[("Silhouette",metrics.get("silhouette")),
        ("Davies–Bouldin",metrics.get("davies_bouldin")),
        ("Calinski–Harabasz",metrics.get("calinski_harabasz"))]
    if extra:
        values=[*extra,*values]
    columns=st.columns(len(values))
    for column,(label,value) in zip(columns,values):
        rendered="—" if value is None or pd.isna(value) else f"{value:.4f}" if isinstance(value,(float,np.floating)) else str(value)
        column.metric(label,rendered)


def _style_chart(figure, discrete_x=False):
    figure.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font={"size":15},title_font_size=20,legend_font_size=13,margin={"l":20,"r":20,"t":55,"b":25})
    if discrete_x:
        figure.update_xaxes(tickmode="linear",dtick=1)
    return figure


def run():
    st.set_page_config(page_title="FitSense AI",page_icon="🏃",layout="wide")
    st.markdown("""<style>
    h1 {font-size: 2.05rem !important;}
    h2 {font-size: 1.55rem !important;}
    h3 {font-size: 1.2rem !important;}
    [data-testid="stDataFrame"] {font-size: 0.95rem;}
    [data-testid="stMetricLabel"] {font-size: 0.95rem;}
    </style>""",unsafe_allow_html=True)
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
        with st.expander("Configuración y detalles técnicos del preprocesamiento"):
            st.json(config)
            if prep["bounds"]: st.json(prep["bounds"])
            st.download_button("Descargar registro de transformaciones",json.dumps(config,ensure_ascii=False,indent=2).encode(),"preprocessing_config.json","application/json")

    elif page=="K-Means":
        prep=_prepared(df,defaults); X=prep["scaled"].to_numpy(); n=len(X); max_k=min(10,n-1)
        if max_k<2: st.warning("Se requieren al menos tres observaciones."); return
        active=st.session_state.setdefault("active_model_parameters",{})
        km_defaults=active.get("K-Means",{})
        k=st.slider("K seleccionado por el analista",2,max_k,min(int(km_defaults.get("k",3)),max_k),key="km_k")
        ninit=st.number_input("n_init",10,100,int(km_defaults.get("n_init",20)),key="km_n_init")
        seed=st.number_input("Semilla",0,100000,int(km_defaults.get("random_state",42)),key="km_seed")
        active["K-Means"]={"k":int(k),"n_init":int(ninit),"random_state":int(seed)}
        st.session_state["active_model_parameters"]=active
        if st.button("Explorar K=2…10",key="run_k_search"):
            ks=tuple(range(2,max_k+1)); table=cached_search_k(X,ks,int(ninit),int(seed))
            st.session_state["k_search"]={"prep_id":prep["configuration_id"],"parameters":{"K":ks,"n_init":int(ninit),"seed":int(seed)},"table":table}
        search=st.session_state.get("k_search")
        search_valid=bool(search and search["prep_id"]==prep["configuration_id"])
        if search_valid:
            table=search["table"].copy(); table["seleccionado"]=table["k"]==int(k)
            st.caption(f"Exploración guardada: K de {search['parameters']['K'][0]} a {search['parameters']['K'][-1]}, n_init={search['parameters']['n_init']} y semilla={search['parameters']['seed']}.")
            figure=px.line(table,x="k",y="silhouette",markers=True,title="Silhouette por número de clusters")
            selected_row=table.loc[table["k"]==int(k)]
            if not selected_row.empty:
                figure.add_scatter(x=[int(k)],y=[float(selected_row.iloc[0]["silhouette"])],mode="markers",name=f"K seleccionado: {k}",marker={"size":16,"symbol":"star","color":"#F2C14E"})
            st.plotly_chart(_style_chart(figure,discrete_x=True),width="stretch")
            inertia=px.line(table,x="k",y="inertia",markers=True,title="Método del codo")
            st.plotly_chart(_style_chart(inertia,discrete_x=True),width="stretch")
            st.dataframe(table.round(4),width="stretch",hide_index=True)
            best=table.loc[table["silhouette"].idxmax()]
            st.caption(f"Mayor Silhouette de la exploración: K={int(best['k'])} ({best['silhouette']:.4f}); la selección no se cambia automáticamente. Revise también tamaños y perfiles.")
        elif search:
            st.warning("La exploración guardada corresponde a otro preprocesamiento. Ejecútela nuevamente.")

        if st.button("Ejecutar K-Means",key="fit_kmeans"):
            model,labels=fit_kmeans(X,k,int(ninit),int(seed))
            metrics=clustering_metrics(X,labels,noise_label=None)
            config={"k":int(k),"n_init":int(ninit),"random_state":int(seed)}
            _save_model("kmeans",model,labels,metrics,config,prep)
            st.session_state.pop("stability_kmeans",None)
        expected=_current_configs(prep)["K-Means"]; result=_results().get("kmeans")
        result_current=result_matches(result,expected,n) and result.get("preprocessing_id")==prep["configuration_id"]
        selected_score=None; score_source=None
        if search_valid:
            selected_search=search["table"].loc[search["table"]["k"]==int(k)]
            if not selected_search.empty:
                selected_row=selected_search.iloc[0]
                best=search["table"].loc[search["table"]["silhouette"].idxmax()]
                st.info(f"K={k} produce grupos entre {int(selected_row['grupo_menor_n'])} y {int(selected_row['grupo_mayor_n'])} días; K={int(best['k'])} alcanza la mayor Silhouette explorada ({best['silhouette']:.4f}) y su grupo menor tiene {int(best['grupo_menor_n'])} días. Un K menor simplifica la segmentación, pero puede combinar subperfiles; valore esta diferencia con medianas, perfiles y estabilidad.")
        if result_current:
            selected_score=result["metrics"].get("silhouette"); score_source="entrenamiento vigente"
        elif search_valid and not search["table"].loc[search["table"]["k"]==int(k)].empty:
            selected_score=float(search["table"].loc[search["table"]["k"]==int(k),"silhouette"].iloc[0]); score_source="exploración guardada"
        shown="—" if selected_score is None or pd.isna(selected_score) else f"{selected_score:.4f}"
        st.metric(f"Silhouette para K={k}",shown,help=f"Fuente: {score_source or 'entrene K-Means o explore varios valores de K'}.")
        if result_current:
            labels=result["labels"]
            st.markdown("#### Resultado del K-Means seleccionado")
            st.caption(f"Configuración independiente: K={result['configuration']['k']}, n_init={result['configuration']['n_init']}, semilla={result['configuration']['random_state']}. Métricas sobre todas las observaciones.")
            sizes=pd.Series(labels).value_counts().sort_index()
            size_table=pd.DataFrame({"cluster":sizes.index,"observaciones":sizes.values,"porcentaje":100*sizes.values/len(labels)})
            _metric_cards(result["metrics"],extra=[("Clusters",int(result["metrics"]["n_clusters"])),("Observaciones",int(result["metrics"]["n_evaluated"]))])
            st.dataframe(size_table,width="stretch",hide_index=True); st.bar_chart(sizes)
            if X.shape[1]>1:
                scatter=px.scatter(x=X[:,0],y=X[:,1],color=labels.astype(str),title="K-Means en las dos primeras variables escaladas",labels={"x":prep["features"][0],"y":prep["features"][1]})
                st.plotly_chart(_style_chart(scatter),width="stretch")
            centers=result["model"].cluster_centers_; raw_centers=prep["scaler"].inverse_transform(centers)
            st.subheader("Centroides en unidades originales"); st.dataframe(pd.DataFrame(raw_centers,columns=prep["features"]).assign(cluster=range(len(centers))),width="stretch")
            with st.expander("Parámetros y métricas técnicas"):
                st.json({"configuration_id":result["configuration_id"],"ejecutado":result["executed_at"],**result["configuration"],**result["metrics"]})
            if st.button("Evaluar estabilidad K-Means",key="stability_kmeans_button"):
                summary,ari=assess_kmeans_stability(X,k,seeds=(11,23,42,77,101),n_init=int(ninit))
                st.session_state["stability_kmeans"]={"configuration_id":expected,"summary":summary,"ari":ari}
                st.session_state.setdefault("stability_results",{})["K-Means"]={"configuration_id":expected,"summary":f"ARI medio={ari['ari_mean']:.3f}; desviación={ari['ari_std']:.3f}"}
            stability=st.session_state.get("stability_kmeans")
            if stability and stability["configuration_id"]==expected:
                st.info(f"Estabilidad por semillas: ARI medio={stability['ari']['ari_mean']:.3f} (DE={stability['ari']['ari_std']:.3f}).")
                st.dataframe(stability["summary"],width="stretch",hide_index=True)
        else:
            if result: _artifact_warning(prep,"K-Means")
            if selected_score is not None:
                st.info("Este Silhouette proviene de la exploración guardada. Entrene K-Means con los parámetros actuales para ver tamaños y métricas del modelo.")
        st.caption("Silhouette resume cohesión y separación, pero no elige K por sí sola. Considere sencillez, tamaños, perfiles y estabilidad.")
        with st.expander("Registrar justificación del analista"):
            rationale=st.text_area("¿Por qué se conserva este K frente a las alternativas?",key="km_selection_reason")
            if st.button("Guardar justificación de K",key="save_k_rationale"):
                st.session_state["km_rationale"]={"configuration_id":expected,"text":rationale}
            saved=st.session_state.get("km_rationale")
            if saved and saved.get("configuration_id")==expected and saved.get("text"):
                st.success(saved["text"])

    elif page=="DBSCAN":
        prep=_prepared(df,defaults); X=prep["scaled"].to_numpy(); n=len(X)
        ms_curve=st.slider("min_samples para curva k-distances",2,min(30,n),5,key="db_curve_min")
        kd=cached_k_distances(X,int(ms_curve))
        curve=px.line(x=np.arange(len(kd)),y=kd,title="Curva k-distances",labels={"x":"Observaciones ordenadas","y":"Distancia al k-ésimo vecino"})
        st.plotly_chart(_style_chart(curve),width="stretch")
        active=st.session_state.setdefault("active_model_parameters",{})
        db_defaults=active.get("DBSCAN",{})
        eps=st.number_input("eps",min_value=.01,max_value=20.,value=float(db_defaults.get("eps",.8)),step=.05,key="db_eps")
        min_samples=st.slider("min_samples",2,min(30,n),min(int(db_defaults.get("min_samples",5)),min(30,n)),key="db_min_samples")
        active["DBSCAN"]={"eps":float(eps),"min_samples":int(min_samples)}
        st.session_state["active_model_parameters"]=active
        if st.button("Explorar grilla DBSCAN",key="db_grid_button"):
            eps_values=tuple(sorted({round(float(v),4) for v in np.linspace(max(.05,eps*.5),eps*1.5,5)}|{round(float(eps),4)}))
            ms_values=tuple(sorted({max(2,min_samples-1),int(min_samples),min(30,min_samples+1)}))
            grid=cached_dbscan_grid(X,eps_values,ms_values)
            grid["seleccionada"]=(np.isclose(grid["eps"],float(eps)) & (grid["min_samples"]==int(min_samples)))
            st.session_state["db_grid"]={"prep_id":prep["configuration_id"],"parameters":{"eps":eps_values,"min_samples":ms_values},"table":grid}
        grid=st.session_state.get("db_grid")
        grid_valid=bool(grid and grid["prep_id"]==prep["configuration_id"])
        if grid_valid:
            st.caption(f"Grilla exploratoria. La combinación seleccionada está marcada; eps={grid['parameters']['eps']} y min_samples={grid['parameters']['min_samples']}.")
            st.dataframe(grid["table"].round(3),width="stretch",hide_index=True)
        elif grid:
            st.warning("La grilla guardada pertenece a otro preprocesamiento. Explórela de nuevo.")
        if st.button("Ejecutar DBSCAN",key="fit_dbscan"):
            model,labels,metrics,kinds=fit_dbscan(X,float(eps),int(min_samples))
            config={"eps":float(eps),"min_samples":int(min_samples)}
            _save_model("dbscan",model,labels,metrics,config,prep,kinds=kinds)
            st.session_state.pop("stability_dbscan",None)
        expected=_current_configs(prep)["DBSCAN"]; result=_results().get("dbscan")
        result_current=result_matches(result,expected,n) and result.get("preprocessing_id")==prep["configuration_id"]
        if result_current:
            labels=result["labels"]; metrics=result["metrics"]
            st.markdown("#### Configuración DBSCAN seleccionada")
            st.caption(f"eps={result['configuration']['eps']}, min_samples={result['configuration']['min_samples']}. Las métricas internas excluyen el ruido (-1); cobertura y ruido se calculan sobre las {n} observaciones.")
            _metric_cards(metrics,extra=[("Clusters",int(metrics["n_clusters"])),("Ruido",int(metrics["n_noise"])),("Cobertura",f"{100*metrics['coverage']:.1f}%")])
            sizes=pd.Series(labels).value_counts().sort_index()
            size_table=pd.DataFrame({"cluster":sizes.index,"observaciones":sizes.values,"porcentaje":100*sizes.values/n})
            st.dataframe(size_table,width="stretch",hide_index=True)
            if X.shape[1]>1:
                scatter=px.scatter(x=X[:,0],y=X[:,1],color=labels.astype(str),title="DBSCAN en las dos primeras variables escaladas",labels={"x":prep["features"][0],"y":prep["features"][1]})
                st.plotly_chart(_style_chart(scatter),width="stretch")
            with st.expander("Parámetros y métricas técnicas"):
                st.json({"configuration_id":result["configuration_id"],"ejecutado":result["executed_at"],**result["configuration"],**metrics})
            if st.button("Evaluar estabilidad DBSCAN",key="stability_dbscan_button"):
                selected_eps=float(result["configuration"]["eps"]); selected_ms=int(result["configuration"]["min_samples"])
                eps_values=sorted({max(.05,selected_eps*.75),selected_eps,selected_eps*1.25})
                ms_values=sorted({max(2,selected_ms-1),selected_ms,min(30,selected_ms+1)})
                configs,pairs=assess_dbscan_stability(X,eps_values,ms_values)
                st.session_state["stability_dbscan"]={"configuration_id":expected,"configs":configs,"pairs":pairs}
                ari_mean=pairs["ari_including_noise"].mean() if not pairs.empty else float("nan")
                st.session_state.setdefault("stability_results",{})["DBSCAN"]={"configuration_id":expected,"summary":f"ARI medio entre parámetros={ari_mean:.3f}"}
            stability=st.session_state.get("stability_dbscan")
            if stability and stability["configuration_id"]==expected:
                st.info("Estabilidad ante variaciones de eps y min_samples; este experimento mide sensibilidad a parámetros, no estabilidad por semillas.")
                st.dataframe(stability["configs"],width="stretch",hide_index=True); st.dataframe(stability["pairs"],width="stretch",hide_index=True)
            with st.expander("Registrar justificación de parámetros"):
                reason=st.text_area("¿Por qué esta cobertura, cantidad de grupos y proporción de ruido son útiles para el análisis?",key="db_selection_reason")
                if st.button("Guardar justificación DBSCAN",key="save_db_rationale"):
                    st.session_state["db_rationale"]={"configuration_id":expected,"text":reason}
                saved=st.session_state.get("db_rationale")
                if saved and saved.get("configuration_id")==expected and saved.get("text"):
                    st.success(saved["text"])
        else:
            if result: _artifact_warning(prep,"DBSCAN")
            st.caption("La combinación visible es la selección actual del analista, no un óptimo automático. Contraste la grilla, el número de clusters, la cobertura y el ruido antes de justificarla.")

    elif page=="Análisis PCA":
        prep=_prepared(df,defaults); X=prep["scaled"].to_numpy()
        active=st.session_state.setdefault("active_model_parameters",{})
        threshold=st.slider("Varianza objetivo",.5,.99,float(active.get("pca_threshold",.9)),.01,key="pca_threshold")
        active["pca_threshold"]=float(threshold); st.session_state["active_model_parameters"]=active
        expected=pca_fingerprint(prep["configuration_id"],threshold); pca_result=st.session_state.get("pca_result")
        if st.button("Ajustar PCA",key="fit_pca_button"):
            model,scores,info=cached_pca(X,float(threshold))
            pca_result={"model":model,"scores":scores,"info":info,"configuration_id":expected,"preprocessing_id":prep["configuration_id"],"threshold":float(threshold)}
            st.session_state["pca_result"]=pca_result
        if pca_result and pca_result["configuration_id"]==expected and pca_result["preprocessing_id"]==prep["configuration_id"]:
            model,scores,info=pca_result["model"],pca_result["scores"],pca_result["info"]
            c1,c2,c3=st.columns(3); c1.metric("Componentes",info["n_components"]); c2.metric("Varianza retenida",f"{info['variance_retained']:.1%}"); c3.metric("RMSE estandarizado",f"{info['reconstruction_rmse_scaled']:.3f}")
            scree=px.bar(x=np.arange(1,len(info["explained_variance_ratio"])+1),y=info["explained_variance_ratio"],labels={"x":"Componente","y":"Varianza explicada"},title="Scree plot")
            st.plotly_chart(_style_chart(scree,discrete_x=True),width="stretch")
            cumulative=px.line(x=np.arange(1,len(info["cumulative_variance"])+1),y=info["cumulative_variance"],markers=True,title="Varianza acumulada")
            st.plotly_chart(_style_chart(cumulative,discrete_x=True),width="stretch")
            st.dataframe(info["loadings"].set_axis(prep["features"],axis=0),width="stretch")
            if scores.shape[1]>=2: st.plotly_chart(px.scatter(x=scores[:,0],y=scores[:,1],labels={"x":"PC1","y":"PC2"},title="Observaciones en PC1 frente a PC2"),width="stretch")
        elif pca_result: st.warning("PCA guardado pertenece a otra configuración. Ajuste PCA para el umbral y preprocesamiento actuales.")
        else: st.info("Ajuste PCA para calcular componentes, varianza, cargas y reconstrucción.")

    elif page=="Modelos combinados":
        prep=_prepared(df,defaults); X=prep["scaled"].to_numpy()
        active=st.session_state.setdefault("active_model_parameters",{})
        threshold=st.slider("Varianza objetivo PCA compartido",.5,.99,float(active.get("pca_threshold",.9)),.01,key="pca_threshold")
        active["pca_threshold"]=float(threshold)
        pca_id=pca_fingerprint(prep["configuration_id"],threshold)
        combined_defaults=active.get("PCA + K-Means",{})
        dbp_defaults=active.get("PCA + DBSCAN",{})
        k=st.slider("K para PCA + K-Means",2,min(10,len(X)-1),min(int(combined_defaults.get("k",4)),min(10,len(X)-1)),key="combined_k")
        eps=st.number_input("eps para PCA + DBSCAN",.01,20.,float(dbp_defaults.get("eps",.8)),.05,key="combined_eps")
        ms=st.slider("min_samples para PCA + DBSCAN",2,min(30,len(X)),min(int(dbp_defaults.get("min_samples",5)),min(30,len(X))),key="combined_min_samples")
        active["PCA + K-Means"]={"k":int(k),"n_init":20,"random_state":42}
        active["PCA + DBSCAN"]={"eps":float(eps),"min_samples":int(ms)}
        st.session_state["active_model_parameters"]=active
        pca_result=st.session_state.get("pca_result")
        pca_valid=bool(pca_result and pca_result["configuration_id"]==pca_id and pca_result["preprocessing_id"]==prep["configuration_id"])
        if not pca_valid:
            st.info("Ejecute el botón para ajustar los dos modelos sobre la misma transformación PCA activa.")
        if st.button("Ejecutar ambos modelos combinados",key="fit_combined"):
            if not pca_valid:
                pca_model,scores,info=cached_pca(X,float(threshold))
                pca_result={"model":pca_model,"scores":scores,"info":info,"configuration_id":pca_id,"preprocessing_id":prep["configuration_id"],"threshold":float(threshold)}
                st.session_state["pca_result"]=pca_result
            pca_model,scores,info=pca_result["model"],pca_result["scores"],pca_result["info"]
            km,lk=fit_pca_kmeans(scores,k); km_metrics=clustering_metrics(X,lk,noise_label=None)
            db,ld,db_metrics,kinds=fit_pca_dbscan(scores,eps,ms); db_metrics_original=clustering_metrics(X,ld)
            _save_model("pca_kmeans",km,lk,km_metrics,{"k":int(k),"n_init":20,"random_state":42},prep,pca_id,pca_variance_threshold=float(threshold),pca_components=int(info["n_components"]))
            _save_model("pca_dbscan",db,ld,db_metrics_original,{"eps":float(eps),"min_samples":int(ms)},prep,pca_id,kinds=kinds,metrics_pca=db_metrics,pca_variance_threshold=float(threshold),pca_components=int(info["n_components"]))
            configs_now=_current_configs(prep)
            st.session_state["pca_diagnostic_metrics"]={
                "configuration_ids":{"PCA + K-Means":configs_now["PCA + K-Means"],"PCA + DBSCAN":configs_now["PCA + DBSCAN"]},
                "metrics":{"pca_kmeans":clustering_metrics(scores,lk,noise_label=None),"pca_dbscan":db_metrics},
            }
        current=_valid_results(prep)
        if "PCA + K-Means" in current:
            result=current["PCA + K-Means"]; scores=pca_result["scores"]
            st.markdown("#### PCA + K-Means")
            st.caption(f"Configuración independiente: K={result['configuration']['k']}, n_init={result['configuration']['n_init']}, semilla={result['configuration']['random_state']}. No hereda K del K-Means original.")
            scatter=px.scatter(x=scores[:,0],y=scores[:,1],color=result["labels"].astype(str),title="Clusters en las dos primeras componentes",labels={"x":"PC1","y":"PC2"})
            st.plotly_chart(_style_chart(scatter),width="stretch")
            centers=reconstruct_centroids(result["model"],pca_result["model"],prep["scaler"])
            st.subheader("Centroides reconstruidos en unidades originales")
            st.dataframe(pd.DataFrame(centers,columns=prep["features"]).assign(cluster=range(len(centers))),width="stretch")
            _metric_cards(result["metrics"],extra=[("Clusters",int(result["metrics"]["n_clusters"])),("Observaciones",int(result["metrics"]["n_evaluated"]))])
            with st.expander("Detalles técnicos PCA + K-Means"):
                st.json({"configuration_id":result["configuration_id"],"pca_id":result["pca_id"],"ejecutado":result["executed_at"],**result["configuration"],**result["metrics"]})
        else:
            _artifact_warning(prep,"PCA + K-Means")
        if "PCA + DBSCAN" in current:
            result=current["PCA + DBSCAN"]; scores=pca_result["scores"]; metrics=result["metrics"]
            st.markdown("#### PCA + DBSCAN")
            st.caption(f"Configuración independiente: eps={result['configuration']['eps']}, min_samples={result['configuration']['min_samples']}. Comparte PCA con PCA + K-Means, pero tiene sus propias etiquetas y parámetros.")
            scatter=px.scatter(x=scores[:,0],y=scores[:,1],color=result["labels"].astype(str),title="Clusters y ruido en las dos primeras componentes",labels={"x":"PC1","y":"PC2"})
            st.plotly_chart(_style_chart(scatter),width="stretch")
            _metric_cards(metrics,extra=[("Clusters",int(metrics["n_clusters"])),("Ruido",int(metrics["n_noise"])),("Cobertura",f"{metrics['coverage']:.1%}")])
            st.caption("Las métricas principales se calculan en el espacio original estandarizado; las métricas en PCA son diagnóstico secundario y excluyen ruido.")
            with st.expander("Detalles técnicos PCA + DBSCAN"):
                st.json({"configuration_id":result["configuration_id"],"pca_id":result["pca_id"],"ejecutado":result["executed_at"],**result["configuration"],"metricas_espacio_original":metrics,"metricas_espacio_pca":result.get("metrics_pca")})
        else:
            _artifact_warning(prep,"PCA + DBSCAN")
        if pca_result and pca_result.get("preprocessing_id")==prep["configuration_id"] and pca_result.get("configuration_id")==pca_id:
            st.info(f"PCA compartido: {pca_result['info']['n_components']} componentes, {pca_result['info']['variance_retained']:.1%} de varianza retenida; RMSE={pca_result['info']['reconstruction_rmse_scaled']:.3f}.")

    elif page=="Comparación y estabilidad":
        prep=_prepared(df,defaults); valid=_valid_results(prep)
        labels={name:result["labels"] for name,result in valid.items()}
        configs={name:{**result["configuration"],**({"umbral_varianza_PCA":result.get("pca_variance_threshold"),"componentes_PCA":result.get("pca_components")} if name.startswith("PCA +") else {})} for name,result in valid.items()}
        stability=_stability_labels(valid)
        if valid and st.button("Evaluar estabilidad de resultados vigentes",key="comparison_stability"):
            stability=_evaluate_stability(prep,valid)
        individual=individual_comparison(prep["scaled"].to_numpy(),labels,configs,stability)
        st.subheader("A. Resultados individuales de las cuatro alternativas")
        st.caption("K-Means se evalúa sobre todas las filas. En DBSCAN las métricas internas excluyen el ruido (-1); cobertura y porcentaje de ruido usan el total. Los soportes difieren, por lo que las métricas individuales no son comparables directamente.")
        individual_cols=["modelo","estado","configuracion","n_clusters","n_evaluated","n_noise","coverage_pct","noise_pct","silhouette","davies_bouldin","calinski_harabasz","estabilidad"]
        st.dataframe(individual[individual_cols].round(4),width="stretch",hide_index=True)
        st.download_button("Descargar resultados individuales",individual[individual_cols].to_csv(index=False).encode("utf-8-sig"),"comparacion_individual_cuatro_modelos.csv","text/csv")
        missing=[name for name in MODEL_KEYS if name not in valid]
        if missing:
            st.info("Para habilitar la comparación común, entrene estas alternativas con la configuración activa: "+", ".join(missing)+".")
        else:
            common=common_assigned_comparison(prep["scaled"].to_numpy(),labels)
            st.subheader("B. Comparación sobre observaciones comunes")
            st.write(f"Soporte compartido: **{common['n_common']} de {len(prep['scaled'])} observaciones** no ruidosas en las dos configuraciones DBSCAN.")
            st.caption("Las cuatro etiquetas se evalúan en las mismas filas y en la matriz original estandarizada. Si un modelo no tiene dos grupos válidos u observaciones suficientes, sus métricas quedan no disponibles.")
            if common["valid"]:
                common_table=common["table"].copy()
                common_table["configuracion"]=common_table["modelo"].map(lambda name:json.dumps(configs[name],ensure_ascii=False,sort_keys=True))
                common_cols=["modelo","configuracion","n_common","n_clusters","n_evaluated","silhouette","davies_bouldin","calinski_harabasz"]
                st.dataframe(common_table[common_cols].round(4),width="stretch",hide_index=True)
                st.download_button("Descargar comparación común",common_table[common_cols].to_csv(index=False).encode("utf-8-sig"),"comparacion_soporte_comun.csv","text/csv")
            else:
                st.warning(f"Comparación común no válida: {common['reason']}. Las métricas que incumplen las condiciones se mantienen vacías.")
            diagnostic=st.session_state.get("pca_diagnostic_metrics")
            diagnostic_current=bool(diagnostic and all(name in valid and diagnostic["configuration_ids"].get(name)==valid[name]["configuration_id"] for name in ("PCA + K-Means","PCA + DBSCAN")))
            if diagnostic_current:
                with st.expander("Métricas técnicas en espacio PCA"):
                    st.write("Diagnóstico secundario. La tabla común anterior usa siempre el espacio original estandarizado.")
                    st.json(diagnostic["metrics"])
        st.markdown("**Estabilidad:** ARI (Adjusted Rand Index) compara particiones sin depender del número de cada etiqueta; 1 indica concordancia perfecta, cerca de 0 concordancia similar al azar y puede ser negativo. K-Means se mide con distintas semillas; DBSCAN al variar eps y min_samples. Son experimentos diferentes.")
        st.caption("La columna Estabilidad indica el procedimiento cuando existe resultado para la configuración vigente; si no, señala el análisis pendiente.")
        stale=[name for name in MODEL_KEYS if name not in valid and _results().get(MODEL_KEYS[name])]
        if stale: st.warning("Resultados anteriores excluidos de la comparación: "+", ".join(stale))

    else:
        _render_profiles_and_decision(df,defaults,mapping)


def _render_profiles_and_decision(df,defaults,mapping):
    prep=_prepared(df,defaults); valid=_valid_results(prep)
    if not valid:
        st.info("Ejecute una alternativa con el preprocesamiento actual para revisar sus perfiles y evidencia.")
        return
    options=list(valid)
    selected=st.selectbox("Modelo cuyos clusters se van a perfilar",options,key="profile_model")
    result=valid[selected]
    st.markdown(f"#### Configuración activa: {selected}")
    if selected=="K-Means":
        st.caption(f"Espacio original estandarizado · K={result['configuration']['k']} · n_init={result['configuration']['n_init']} · semilla={result['configuration']['random_state']}.")
    elif selected=="DBSCAN":
        st.caption(f"Espacio original estandarizado · eps={result['configuration']['eps']} · min_samples={result['configuration']['min_samples']}.")
    elif selected=="PCA + K-Means":
        st.caption(f"Espacio PCA compartido · K={result['configuration']['k']} · umbral={result.get('pca_variance_threshold',.9):.0%} · componentes={result.get('pca_components','—')} · n_init={result['configuration']['n_init']} · semilla={result['configuration']['random_state']}.")
    else:
        st.caption(f"Espacio PCA compartido · eps={result['configuration']['eps']} · min_samples={result['configuration']['min_samples']} · umbral PCA={result.get('pca_variance_threshold',.9):.0%} · componentes={result.get('pca_components','—')}.")
    with st.expander("Huella y detalles de la ejecución"):
        st.json({"configuration_id":result["configuration_id"],"preprocessing_id":result["preprocessing_id"],"pca_id":result.get("pca_id"),"ejecutado":result["executed_at"],**result["configuration"]})

    summary,details,heatmap=profile_details(prep["raw"],prep["scaled"],result["labels"])
    st.subheader("Interpretación descriptiva por cluster")
    st.caption("Los nombres se generan con las medias estandarizadas observadas de cada grupo respecto a la media global; las estadísticas detalladas se expresan en las unidades de las variables.")
    st.dataframe(summary.round({"porcentaje":1}),width="stretch",hide_index=True)
    st.subheader("Mapa de calor de medias estandarizadas")
    heatmap_fig=px.imshow(heatmap,text_auto=".2f",aspect="auto",color_continuous_scale="RdBu_r",color_continuous_midpoint=0,
        labels={"x":"Variable","y":"Cluster","color":"Media estandarizada"},title="Desviación de cada grupo respecto al promedio")
    st.plotly_chart(_style_chart(heatmap_fig),width="stretch")
    with st.expander("Medias, medianas y desviaciones estándar por variable"):
        st.dataframe(details.round(2),width="stretch",hide_index=True)
    st.caption("Cada cluster agrupa días persona-día: una persona puede aparecer en distintos clusters. Los perfiles son descriptivos; no establecen causalidad ni diagnósticos médicos.")

    public_meta=pseudonymize_frame(prep["metadata"],mapping)
    st.subheader("Días potencialmente inusuales")
    if selected in {"K-Means","PCA + K-Means"}:
        percentile=st.slider("Percentil de distancia a centroides para K-Means",80,100,95,key="outlier_percentile")
        candidates,threshold=kmeans_outlier_candidates(prep["scaled"].to_numpy(),result["labels"],public_meta,percentile)
        st.caption(f"Umbral empírico: percentil {percentile} de distancia en espacio original estandarizado = {threshold:.4f}.")
        if candidates.empty: st.info("No hay observaciones por encima del umbral seleccionado.")
        else: st.dataframe(candidates,width="stretch",hide_index=True)
    else:
        noise=outlier_days(public_meta,result["labels"])
        if noise.empty: st.info("No se identificaron observaciones con etiqueta de ruido (-1).")
        else: st.dataframe(noise,width="stretch",hide_index=True)
    st.caption("Los candidatos a días inusuales no son errores confirmados ni diagnósticos.")

    st.divider(); st.header("Evidencia y decisión del equipo")
    labels={name:artifact["labels"] for name,artifact in valid.items()}
    if len(valid)==len(MODEL_KEYS):
        common=common_assigned_comparison(prep["scaled"].to_numpy(),labels)
        metrics_table=common["table"] if common["valid"] else compare_models(prep["scaled"].to_numpy(),labels)
    else:
        metrics_table=compare_models(prep["scaled"].to_numpy(),labels)
    stability=_stability_labels(valid)
    if st.button("Evaluar estabilidad de resultados vigentes",key="decision_stability"):
        stability=_evaluate_stability(prep,valid)
    pca_result=st.session_state.get("pca_result")
    pca_id=pca_fingerprint(prep["configuration_id"],st.session_state.get("active_model_parameters",{}).get("pca_threshold",.90))
    pca_variance=pca_result["info"]["variance_retained"] if pca_result and pca_result.get("configuration_id")==pca_id else None
    evidence=evidence_matrix(metrics_table,stability,pca_variance)
    display_cols=[c for c in ["modelo","n_clusters","n_evaluated","n_noise","coverage_pct","silhouette","davies_bouldin","calinski_harabasz","estabilidad","varianza_pca","ventajas","limitaciones"] if c in evidence]
    st.dataframe(evidence[display_cols],width="stretch",hide_index=True)
    st.markdown("**Lectura de alternativas:** K-Means asigna todos los días y facilita centroides, pero requiere K; DBSCAN identifica ruido sin fijar K y deja cobertura variable. PCA reduce dimensiones y conserva la proporción de varianza que se reporta.")
    st.markdown("**Límites del estudio:** muestra de conveniencia de 33 participantes, periodo cercano a un mes, mediciones estimadas por dispositivos, cobertura de sueño incompleta frente a actividad y generalización a otras poblaciones no demostrada.")
    st.caption("La decisión académica final requiere considerar soporte, cobertura, estabilidad, perfiles e interpretabilidad. Silhouette no elige el modelo por sí sola.")
    chosen=st.selectbox("Alternativa que el equipo considera más adecuada",["Sin definir",*MODEL_KEYS],key="decision_choice")
    arguments=st.text_area("Argumentos académicos de la elección",key="decision_arguments")
    limitations=st.text_area("Limitaciones específicas de la decisión",key="decision_limitations")
    if st.button("Guardar justificación del equipo",key="save_decision"):
        decision=decision_guidance(evidence,chosen,arguments,limitations)
        decision["configuration_id"]=fingerprint({name:artifact["configuration_id"] for name,artifact in valid.items()})
        st.session_state["decision_log"]=decision
    decision=st.session_state.get("decision_log")
    current_decision_id=fingerprint({name:artifact["configuration_id"] for name,artifact in valid.items()})
    if decision and decision.get("configuration_id")==current_decision_id:
        st.info(decision["note"]); st.write("Alternativa documentada:",decision["selected_model"])
        if decision.get("arguments"): st.write("Argumentos:",decision["arguments"])
        if decision.get("limitations"): st.write("Limitaciones registradas:",decision["limitations"])
        st.download_button("Descargar evidencia de decisión",evidence.to_csv(index=False).encode("utf-8-sig"),"matriz_evidencia.csv","text/csv")
    elif decision:
        st.warning("La justificación guardada corresponde a otra configuración de modelos. Revísela y guárdela nuevamente.")
