"""Interfaz multipágina en español para análisis interactivo."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from .config import ROOT, DATA_ORIGINAL, DEFAULT_FEATURES
from .data_loader import load_activity, load_sleep
from .integration import integrate
from .eda import missing_summary, descriptive_summary, correlation_matrix, findings
from .data_dictionary import dictionary_frame
from .preprocessing import prepare_features
from .kmeans_model import search_k, fit_kmeans
from .dbscan_model import fit_dbscan, k_distances, explore_dbscan
from .pca_model import fit_pca
from .combined_models import fit_pca_kmeans, fit_pca_dbscan, reconstruct_centroids
from .evaluation import clustering_metrics, compare_labels
from .model_comparison import compare_models
from .kmeans_stability import assess_kmeans_stability
from .dbscan_stability import assess_dbscan_stability
from .profiling import profile_clusters

NAV=["Inicio","Dataset y calidad","Exploración de datos","Preprocesamiento","K-Means","DBSCAN","Análisis PCA","Modelos combinados","Comparación y estabilidad","Perfiles y conclusiones"]

@st.cache_data(show_spinner=False)
def load_sources():
    ap=DATA_ORIGINAL/"dailyActivity_merged.csv"; sp=DATA_ORIGINAL/"sleepDay_merged.csv"
    return load_activity(ap),load_sleep(sp)

def run():
    st.set_page_config(page_title="FitSense AI",page_icon="🏃",layout="wide")
    st.sidebar.title("FitSense AI")
    page=st.sidebar.radio("Navegación",NAV)
    st.sidebar.caption("Perfiles de actividad física con dispositivos vestibles")
    try:
        activity,sleep=load_sources()
        df,a0,s0,audit,_,_=integrate(activity,sleep)
    except Exception as exc:
        st.error(f"No se pudieron cargar e integrar los datos: {exc}"); st.stop()
    numerical=[c for c in df.select_dtypes(include="number").columns if c not in {"Id"}]
    default=[c for c in DEFAULT_FEATURES if c in numerical]
    if not default: default=numerical[:min(6,len(numerical))]
    st.title(page)
    if page=="Inicio":
        st.subheader("Perfiles de actividad física con dispositivos vestibles")
        st.write("FitSense AI compara K-Means, DBSCAN, PCA + K-Means y PCA + DBSCAN para describir rutinas diarias de actividad y descanso. No realiza diagnósticos médicos.")
        st.markdown("**Pregunta:** ¿Qué rutinas de actividad física y descanso se pueden diferenciar y qué días presentan comportamientos atípicos?")
        c1,c2,c3,c4=st.columns(4); c1.metric("Días integrados",f"{len(df):,}"); c2.metric("Participantes",df.Id.nunique()); c3.metric("Actividad",f"{len(activity):,} filas"); c4.metric("Sueño",f"{len(sleep):,} filas")
        st.info("El aprendizaje no supervisado descubre agrupaciones sin etiquetas objetivo; no predice una clase conocida.")
        st.dataframe(audit,use_container_width=True,hide_index=True)
    elif page=="Dataset y calidad":
        st.write("Fuentes originales CSV. Los identificadores se leen como texto; las fechas se normalizan antes de unir.")
        m1,m2,m3=st.columns(3); m1.metric("Actividad",f"{activity.shape[0]} × {activity.shape[1]}"); m2.metric("Sueño",f"{sleep.shape[0]} × {sleep.shape[1]}"); m3.metric("Duplicados exactos sueño",int(sleep.duplicated().sum()))
        structure=pd.DataFrame({"variable":df.columns,"tipo":df.dtypes.astype(str).values,"faltantes":df.isna().sum().values,"faltantes (%)":(100*df.isna().mean()).round(2).values})
        st.subheader("Estructura y calidad persona-día"); st.dataframe(structure,use_container_width=True,hide_index=True)
        st.caption(f"Duplicados exactos de actividad: {activity.duplicated().sum()}. Duplicados exactos integrados: {df.duplicated().sum()}.")
        st.dataframe(audit,use_container_width=True,hide_index=True); st.subheader("Diccionario de datos"); st.dataframe(dictionary_frame(),use_container_width=True,hide_index=True)
        upload_a=st.file_uploader("CSV alternativo de actividad",type="csv",key="ua"); upload_s=st.file_uploader("CSV alternativo de sueño",type="csv",key="us")
        if upload_a and upload_s:
            try:
                aa=pd.read_csv(upload_a,dtype={"Id":"string"}); ss=pd.read_csv(upload_s,dtype={"Id":"string"}); merged,*_=integrate(aa,ss); st.success(f"Integración alternativa: {len(merged)} observaciones."); st.dataframe(merged.head(30))
            except Exception as exc: st.error(f"No se pudo integrar: {exc}")
        st.subheader("Vista integrada"); st.dataframe(df.head(50),use_container_width=True)
    elif page=="Exploración de datos":
        st.write("Hallazgos descriptivos; correlación no implica causalidad.")
        for item in findings(df): st.markdown(f"- {item}")
        st.subheader("Estadísticas descriptivas"); st.dataframe(descriptive_summary(df).round(2),use_container_width=True)
        cols=st.multiselect("Variables numéricas",numerical,default=default[:4]);
        if cols:
            tab1,tab2=st.tabs(["Distribuciones","Correlación"])
            with tab1:
                col=st.selectbox("Variable",cols); st.plotly_chart(px.histogram(df,x=col,marginal="box",title=f"Distribución de {col}"),use_container_width=True)
            with tab2: st.plotly_chart(px.imshow(correlation_matrix(df[cols]),text_auto=".2f",color_continuous_scale="RdBu_r",zmin=-1,zmax=1),use_container_width=True)
        st.subheader("Valores faltantes"); st.dataframe(missing_summary(df),use_container_width=True)
    elif page=="Preprocesamiento":
        features=st.multiselect("Variables para clustering (sin Id ni fecha)",numerical,default=default)
        missing=st.selectbox("Tratamiento de faltantes",["median","mean","drop","zero"],format_func=lambda x:{"median":"Imputar mediana","mean":"Imputar media","drop":"Excluir filas incompletas","zero":"Imputar cero (explícito)"}[x])
        outlier=st.selectbox("Valores extremos",["keep","iqr","winsorize"],format_func=lambda x:{"keep":"Conservar","iqr":"Limitar por regla IQR (1.5×IQR)","winsorize":"Winsorización en límites IQR"}[x])
        try:
            prep=prepare_features(df,features,missing,outlier); st.session_state["prep"]=prep
            st.metric("Observaciones preparadas",len(prep["scaled"])); st.write("Ceros se conservan como valores válidos. La matriz escalada se comparte entre los cuatro modelos.")
            st.dataframe(prep["raw"].head(30)); st.download_button("Descargar variables preparadas",prep["raw"].to_csv(index=False).encode(),"preprocessed_features.csv","text/csv")
            config={"variables":features,"faltantes":missing,"extremos":outlier,"escalado":"StandardScaler","semilla_modelos":42}
            st.json(config); st.download_button("Descargar registro de transformaciones",json.dumps(config,ensure_ascii=False,indent=2).encode(),"preprocessing_config.json","application/json")
            if prep["bounds"]: st.json(prep["bounds"])
        except Exception as exc: st.warning(f"Configure variables válidas: {exc}")
    elif page=="K-Means":
        prep=_get_prep(df,default); X=prep["scaled"].to_numpy(); max_k=min(10,len(X)-1)
        if max_k<2: st.warning("Se requieren al menos tres observaciones."); return
        search=search_k(X,range(2,max_k+1)); st.plotly_chart(px.line(search,x="k",y="inertia",markers=True,title="Método del codo"),use_container_width=True); st.plotly_chart(px.line(search,x="k",y="silhouette",markers=True,title="Silhouette por K"),use_container_width=True)
        k=st.slider("K elegido por el analista",2,max_k,min(3,max_k)); ninit=st.number_input("n_init",10,100,20); seed=st.number_input("Semilla",0,100000,42)
        if st.button("Ejecutar K-Means"):
            model,labels=fit_kmeans(X,k,ninit,seed); st.session_state["labels_kmeans"]=labels; st.session_state["kmeans_model"]=model
        if "labels_kmeans" in st.session_state:
            labels=st.session_state["labels_kmeans"]; st.json(clustering_metrics(X,labels,noise_label=None)); sizes=pd.Series(labels).value_counts().sort_index(); st.bar_chart(sizes)
            if X.shape[1] > 1: st.plotly_chart(px.scatter(x=X[:,0],y=X[:,1],color=labels.astype(str),title="K-Means · dos primeras variables escaladas",labels={"x":prep["features"][0],"y":prep["features"][1]}),use_container_width=True)
            centers=st.session_state["kmeans_model"].cluster_centers_; raw_centers=prep["scaler"].inverse_transform(centers); st.dataframe(pd.DataFrame(raw_centers,columns=prep["features"]).assign(cluster=range(len(centers))))
            summary,ari=assess_kmeans_stability(X,k); st.subheader("Estabilidad por semillas"); st.dataframe(summary); st.write(ari)
    elif page=="DBSCAN":
        prep=_get_prep(df,default); X=prep["scaled"].to_numpy(); ms=st.slider("min_samples para curva",2,min(30,len(X)),5); kd=k_distances(X,ms); st.plotly_chart(px.line(x=np.arange(len(kd)),y=kd,title="Curva k-distances",labels={"x":"Observaciones ordenadas","y":"Distancia k-ésimo vecino"}),use_container_width=True)
        eps=st.number_input("eps",min_value=.01,max_value=20.,value=.8,step=.05); min_samples=st.slider("min_samples",2,min(30,len(X)),5)
        if st.button("Ejecutar DBSCAN"):
            model,labels,metrics,kinds=fit_dbscan(X,eps,min_samples); st.session_state["labels_dbscan"]=labels; st.session_state["dbscan_metrics"]=metrics; st.session_state["dbscan_kinds"]=kinds
        if "labels_dbscan" in st.session_state:
            labels=st.session_state["labels_dbscan"]; st.json(st.session_state["dbscan_metrics"]); st.write("Ruido = -1; cobertura es fracción asignada a clusters."); st.bar_chart(pd.Series(labels).value_counts().sort_index())
            st.write("Observaciones por tipo:",pd.Series(st.session_state["dbscan_kinds"]).value_counts().to_dict())
            if X.shape[1] > 1: st.plotly_chart(px.scatter(x=X[:,0],y=X[:,1],color=labels.astype(str),title="DBSCAN · dos primeras variables escaladas",labels={"x":prep["features"][0],"y":prep["features"][1]}),use_container_width=True)
        grid=explore_dbscan(X,np.linspace(max(.1,eps*.5),eps*1.5,6),[max(2,min_samples-1),min_samples,min(30,min_samples+1)]); st.dataframe(grid.round(3),use_container_width=True)
    elif page=="Análisis PCA":
        prep=_get_prep(df,default); X=prep["scaled"].to_numpy(); threshold=st.slider("Varianza objetivo",.5,.99,.9,.01); model,scores,info=fit_pca(X,variance_threshold=threshold); st.metric("Componentes",info["n_components"]); st.metric("Varianza retenida",f"{info['variance_retained']:.1%}"); st.metric("RMSE de reconstrucción en escala estandarizada",f"{info['reconstruction_rmse_scaled']:.3f}")
        st.plotly_chart(px.bar(x=np.arange(1,len(info["explained_variance_ratio"])+1),y=info["explained_variance_ratio"],labels={"x":"Componente","y":"Varianza explicada"},title="Scree plot"),use_container_width=True); st.plotly_chart(px.line(x=np.arange(1,len(info["cumulative_variance"])+1),y=info["cumulative_variance"],markers=True,title="Varianza acumulada"),use_container_width=True); st.dataframe(info["loadings"].set_axis(prep["features"],axis=0)); st.session_state["pca"]=(model,scores,info)
    elif page=="Modelos combinados":
        prep=_get_prep(df,default); model_pca,scores,info=st.session_state.get("pca",fit_pca(prep["scaled"].to_numpy(),variance_threshold=.9)); st.caption(f"PCA compartido: {info['n_components']} componentes, {info['variance_retained']:.1%} varianza.")
        k=st.slider("K para PCA + K-Means",2,min(10,len(scores)-1),min(3,len(scores)-1)); eps=st.number_input("eps para PCA + DBSCAN",.01,20.,.8,.05); ms=st.slider("min_samples combinado",2,min(30,len(scores)),5)
        if st.button("Ejecutar modelos combinados"):
            km,lk=fit_pca_kmeans(scores,k); db,ld,md,_=fit_pca_dbscan(scores,eps,ms); st.session_state["labels_pca_kmeans"]=lk; st.session_state["labels_pca_dbscan"]=ld; st.session_state["pca_kmeans_model"]=km; st.session_state["pca_dbscan_metrics"]=md
        for key,title in [("labels_pca_kmeans","PCA + K-Means"),("labels_pca_dbscan","PCA + DBSCAN")]:
            if key in st.session_state:
                labels=st.session_state[key]
                st.plotly_chart(px.scatter(x=scores[:,0],y=scores[:,1],color=labels.astype(str),title=title,labels={"x":"PC1","y":"PC2"}),use_container_width=True)
                st.json(clustering_metrics(scores,labels))
        if "labels_pca_kmeans" in st.session_state:
            centers=reconstruct_centroids(st.session_state["pca_kmeans_model"],model_pca,prep["scaler"])
            st.subheader("Centroides PCA + K-Means reconstruidos a unidades originales"); st.dataframe(pd.DataFrame(centers,columns=prep["features"]).assign(cluster=range(len(centers))))
        st.info("La evaluación comparativa principal se calcula en el espacio escalado original; estas métricas en PCA son diagnósticas.")
    elif page=="Comparación y estabilidad":
        prep=_get_prep(df,default); labels={}
        for key,name in [("labels_kmeans","K-Means"),("labels_dbscan","DBSCAN"),("labels_pca_kmeans","PCA + K-Means"),("labels_pca_dbscan","PCA + DBSCAN")]:
            if key in st.session_state: labels[name]=st.session_state[key]
        if not labels: st.info("Ejecute al menos un modelo para comparar.")
        else:
            table=compare_models(prep["scaled"].to_numpy(),labels)
            stability_values=[]
            for name in table["modelo"]:
                if name in {"K-Means","PCA + K-Means"}:
                    input_matrix=prep["scaled"].to_numpy()
                    if name=="PCA + K-Means": input_matrix=st.session_state.get("pca",fit_pca(input_matrix,variance_threshold=.9))[1]
                    stab,ari=assess_kmeans_stability(input_matrix,k=int(table.loc[table.modelo==name,"n_clusters"].iloc[0]),seeds=(11,23,42,77,101))
                    stability_values.append(f"ARI semillas={ari['ari_mean']:.3f}")
                else:
                    input_matrix=prep["scaled"].to_numpy()
                    if name=="PCA + DBSCAN": input_matrix=st.session_state.get("pca",fit_pca(input_matrix,variance_threshold=.9))[1]
                    _,pairs=assess_dbscan_stability(input_matrix,[.6,.8,1.0],[4,5])
                    stability_values.append(f"ARI config media={pairs.ari_including_noise.mean():.3f}" if not pairs.empty else "No disponible")
            table["estabilidad (ARI)"]=stability_values
            st.dataframe(table.round(4),use_container_width=True)
            st.download_button("Descargar comparación CSV",table.to_csv(index=False).encode(),"comparacion_modelos.csv","text/csv")
        st.markdown("Las métricas internas de DBSCAN excluyen ruido y reportan observaciones evaluadas y cobertura. K-Means se evalúa en toda la muestra. La comparación conjunta de DBSCAN solo es válida en el subconjunto asignado por ambas configuraciones.")
        if "labels_dbscan" in st.session_state and "labels_pca_dbscan" in st.session_state:
            cfg,stability=assess_dbscan_stability(prep["scaled"].to_numpy(),[.6,.8,1.0],[4,5]); st.dataframe(cfg); st.dataframe(stability)
            common=compare_labels(prep["scaled"].to_numpy(),st.session_state["labels_dbscan"],st.session_state["labels_pca_dbscan"],"DBSCAN original","DBSCAN PCA")
            st.subheader("DBSCAN: evaluación en las mismas observaciones asignadas")
            st.write(common)
    else:
        prep=_get_prep(df,default); key=st.selectbox("Modelo para perfiles",[k for k in ["labels_kmeans","labels_dbscan","labels_pca_kmeans","labels_pca_dbscan"] if k in st.session_state] or ["ninguno"])
        if key=="ninguno": st.info("Ejecute un modelo primero.")
        else:
            labels=st.session_state[key]; profiles=profile_clusters(prep["raw"],labels); st.dataframe(profiles,use_container_width=True)
            for cluster in sorted(np.unique(labels)):
                if cluster == -1: continue
                means=prep["scaled"].loc[np.asarray(labels)==cluster].mean().sort_values()
                above=means[means>=.25].sort_values(ascending=False).head(3).index.tolist()
                below=means[means<=-.25].sort_values().head(3).index.tolist()
                st.markdown(f"**Grupo {cluster}:** sobre la media global en {', '.join(above) if above else 'ninguna variable supera el umbral descriptivo'}; bajo la media global en {', '.join(below) if below else 'ninguna variable supera el umbral descriptivo'}. Revise unidades y tamaños antes de asignar un nombre de perfil.")
            if (labels==-1).any():
                meta=prep["metadata"].copy(); meta["etiqueta"]=labels; meta=meta[meta.etiqueta==-1].drop(columns="Id",errors="ignore"); st.subheader("Días marcados como ruido"); st.dataframe(meta)
            st.warning("Una observación inusual no implica necesariamente un error ni un problema de salud. Los perfiles son descriptivos, no clínicos.")

def _get_prep(df, default):
    if "prep" not in st.session_state:
        st.session_state["prep"]=prepare_features(df,default)
    return st.session_state["prep"]

