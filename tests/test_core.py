import numpy as np
import pandas as pd
import pytest
from io import StringIO
from pathlib import Path
from src.integration import integrate, DuplicateKeyError
from src.data_loader import load_sleep
from src.preprocessing import prepare_features
from src.kmeans_model import fit_kmeans
from src.dbscan_model import fit_dbscan
from src.pca_model import fit_pca
from src.evaluation import clustering_metrics, compare_labels
from src.profiling import profile_clusters, profile_details, kmeans_outlier_candidates
from src.kmeans_stability import assess_kmeans_stability
from src.dbscan_stability import assess_dbscan_stability
from src.model_comparison import compare_models, common_assigned_comparison, individual_comparison
from src.combined_models import fit_pca_kmeans, fit_pca_dbscan
from src.result_tracking import (participant_mapping, pseudonymize_frame, preprocessing_fingerprint,
    result_fingerprint, register_result, result_matches, feature_export_frames)

def sample_frames():
    activity=pd.DataFrame({"Id":["01","01","02"],"ActivityDate":["1/1/2024","1/2/2024","1/1/2024"],"steps":[100,200,300]})
    sleep=pd.DataFrame({"Id":["01","01","01","02"],"SleepDay":["1/1/2024 12:00:00 AM","1/1/2024 12:00:00 AM","1/2/2024","1/1/2024"],"minutes":[400,400,420,350]})
    return activity,sleep

def test_integration_exact_dedup_and_unique():
    a,s=sample_frames(); merged,a0,s0,audit,_,_,discarded=integrate(a,s)
    assert len(merged)==3 and audit.loc[audit.Indicador=="Duplicados exactos sueño","Valor"].iloc[0]==1
    assert not merged.duplicated(["Id","Date"]).any()
    assert len(a0)==0 and len(s0)==0 and merged.Id.dtype.name=="string"
    assert len(discarded)==1 and "SleepDay_source" in merged.columns

def test_integration_conflicting_key_raises():
    a,s=sample_frames(); s.loc[1,"minutes"]=410
    with pytest.raises(DuplicateKeyError): integrate(a,s)

def test_distinct_source_times_collide_only_after_normalization():
    activity=pd.DataFrame({"Id":["01"],"ActivityDate":["2024-01-01 00:00:00"],"steps":[1]})
    sleep=pd.DataFrame({"Id":["01","01"],"SleepDay":["2024-01-01 00:00:00","2024-01-01 08:00:00"],"minutes":[400,400]})
    with pytest.raises(DuplicateKeyError) as error:
        integrate(activity,sleep)
    assert len(error.value.sleep_conflicts)==2
    assert set(error.value.sleep_conflicts.SleepDay_source)=={"2024-01-01 00:00:00","2024-01-01 08:00:00"}
    assert error.value.sleep_duplicates.empty

def test_loader_preserves_source_date():
    source=StringIO("Id,SleepDay,TotalSleepRecords,TotalMinutesAsleep,TotalTimeInBed\n001,4/12/2016 08:30:00 AM,1,400,420\n")
    loaded=load_sleep(source)
    assert str(loaded.loc[0,"SleepDay_source"])=="4/12/2016 08:30:00 AM"
    assert loaded.loc[0,"SleepDay"]==pd.Timestamp("2016-04-12")

def test_preprocessing_scaling_and_zero_preservation():
    data=pd.DataFrame({"Id":["1","2","3","4"],"Date":pd.date_range("2024-01-01",periods=4),"x":[0.,1.,None,3.],"y":[1.,2.,3.,4.]})
    prep=prepare_features(data,["x","y"])
    assert prep["scaled"].shape==(4,2) and prep["raw"].iloc[0,0]==0 and np.allclose(prep["scaled"].mean(),0)

def test_models_metrics_and_profiles():
    X=np.array([[0.,0.],[.1,.2],[4.,4.],[4.2,4.1],[10.,10.]])
    km,labels=fit_kmeans(X,2); assert len(labels)==5
    _,db,metrics,_=fit_dbscan(X,.5,2); assert -1 in db and metrics["n_clusters"]==2
    assert clustering_metrics(X,np.zeros(5))['silhouette']!=clustering_metrics(X,np.zeros(5))['silhouette']
    pca,scores,info=fit_pca(X,n_components=2); assert scores.shape==(5,2) and info["variance_retained"]<=1.00001
    fit_pca_kmeans(scores,2)
    fit_pca_dbscan(scores,.5,2)
    prof=profile_clusters(pd.DataFrame(X,columns=["a","b"]),labels); assert "mean" in prof.columns.get_level_values(1)

def test_comparison_and_stability_contracts():
    X=np.array([[0.,0.],[.1,.2],[4.,4.],[4.2,4.1],[8.,8.],[8.1,8.2]])
    km,labels=fit_kmeans(X,2,n_init=10)
    table=compare_models(X,{"kmeans":labels}); assert table.loc[0,"n_evaluated"]==len(X)
    runs,ari=assess_kmeans_stability(X,k=2,seeds=(1,2),n_init=10); assert len(runs)==2 and 0<=ari["ari_mean"]<=1
    configs,pairs=assess_dbscan_stability(X,[.3,.5],[2]); assert len(configs)==2 and "ari_including_noise" in pairs

def test_metric_math_and_label_alignment_guards():
    X=np.array([[0.,0.],[1.,1.],[2.,2.]])
    single=clustering_metrics(X,[0,0,0]); assert not single["valid"] and np.isnan(single["silhouette"])
    singleton_clusters=clustering_metrics(X,[0,1,2]); assert not singleton_clusters["valid"] and "superar" in singleton_clusters["reason"]
    missing=clustering_metrics(X,[0,None,1]); assert not missing["valid"] and "faltantes" in missing["reason"]
    mismatch=clustering_metrics(X,[0,1]); assert not mismatch["valid"] and "alineado" in mismatch["reason"]
    comparison=compare_labels(X,[0,1,2],[0,1,1])
    assert not comparison["valid"] and comparison["common_n"]==3

def test_common_support_uses_same_rows_for_all_models():
    X=np.arange(16,dtype=float).reshape(8,2)
    labels={"K-Means":np.array([0,0,0,0,1,1,1,1]),"DBSCAN":np.array([0,0,0,0,1,1,-1,-1]),"PCA + K-Means":np.array([1,1,0,0,1,1,0,0]),"PCA + DBSCAN":np.array([0,0,0,0,1,1,1,-1])}
    result=common_assigned_comparison(X,labels)
    assert result["valid"] and result["n_common"]==6
    assert set(result["table"].n_evaluated)=={6} and set(result["table"].n_common)=={6}
    labels["PCA + DBSCAN"]=np.array([-1,-1,-1,-1,1,1,1,1])
    invalid=common_assigned_comparison(X,labels)
    assert not invalid["valid"] and "clusters válidos" in invalid["reason"]

def test_model_results_are_bound_to_configuration_and_pseudonyms():
    frame=pd.DataFrame({"Id":["100","200"],"Date":pd.date_range("2024-01-01",periods=2),"x":[1.,2.]})
    mapping=participant_mapping(frame.Id)
    public=pseudonymize_frame(frame,mapping)
    assert public.Participante.tolist()==["Participante_001","Participante_002"] and "Id" not in public
    prep_id=preprocessing_fingerprint(frame,["x"],"median","keep")
    altered=preprocessing_fingerprint(frame.iloc[::-1],["x"],"median","keep")
    model_id=result_fingerprint(prep_id,"K-Means",{"k":2})
    result=register_result("model",[0,1],{"silhouette":.4},{"k":2},model_id)
    changed_k=result_fingerprint(prep_id,"K-Means",{"k":3})
    assert result_matches(result,model_id,2) and not result_matches(result,altered,2) and not result_matches(result,changed_k,2)
    assert result["configuration"]=={"k":2} and result["executed_at"]

def test_scaled_export_has_pseudonyms_and_no_raw_ids():
    metadata=pd.DataFrame({"Id":["A","B"],"Date":pd.date_range("2024-01-01",periods=2)})
    raw=pd.DataFrame({"x":[1.,2.]}); scaled=pd.DataFrame({"x":[-1.,1.]})
    exports=feature_export_frames(metadata,raw,scaled,participant_mapping(metadata.Id))
    assert exports["estandarizada"].x.tolist()==[-1.,1.]
    assert "Id" not in exports["estandarizada"] and "Participante" in exports["estandarizada"]

def test_kmeans_outlier_candidates_return_aligned_pseudonymous_rows():
    X=np.array([[0.,0.],[.1,.1],[0.,.2],[5.,5.],[5.1,5.],[7.,7.]])
    labels=np.array([0,0,0,1,1,1]); meta=pd.DataFrame({"Participante":[f"Participante_{i:03}" for i in range(6)],"Date":pd.date_range("2024-01-01",periods=6)})
    candidates,threshold=kmeans_outlier_candidates(X,labels,meta,percentile=80)
    assert threshold>0 and not candidates.empty and "distancia_centroide" in candidates
    assert "Participante" in candidates and "Date" in candidates and "Id" not in candidates

def test_preprocessing_outlier_strategies_have_distinct_bounds():
    df=pd.DataFrame({"x":[0.,1.,2.,3.,4.,5.,6.,7.,8.,100.]})
    iqr=prepare_features(df,["x"],outliers="iqr")
    win=prepare_features(df,["x"],outliers="winsorize")
    assert iqr["bounds"]["x"]!=win["bounds"]["x"]

def test_profile_details_are_data_driven_and_include_noise():
    raw=pd.DataFrame({"TotalSteps":[100.,110.,900.,1000.],"VeryActiveMinutes":[0.,1.,30.,40.],
        "FairlyActiveMinutes":[0.,0.,15.,20.],"LightlyActiveMinutes":[100.,110.,250.,260.],
        "SedentaryMinutes":[1000.,950.,500.,450.],"TotalMinutesAsleep":[300.,320.,450.,470.]})
    scaled=(raw-raw.mean())/raw.std(ddof=0)
    summary,details,heatmap=profile_details(raw,scaled,[0,0,1,-1])
    assert summary.observaciones.tolist()==[1,2,1]
    assert summary.porcentaje.sum()==100 and summary.perfil_sugerido.nunique()==3
    assert "ruido DBSCAN" in summary.loc[summary.cluster==-1,"perfil_sugerido"].iloc[0]
    assert len(details)==len(summary)*raw.shape[1] and heatmap.shape==(3,raw.shape[1])

def test_individual_comparison_always_lists_four_and_marks_pending():
    X=np.array([[0.,0.],[.1,.1],[4.,4.],[4.1,4.2]])
    _,labels=fit_kmeans(X,2)
    table=individual_comparison(X,{"K-Means":labels},{"K-Means":{"k":2}})
    assert table.modelo.tolist()==["K-Means","DBSCAN","PCA + K-Means","PCA + DBSCAN"]
    assert table.estado.tolist()==["Ejecutado","Pendiente","Pendiente","Pendiente"]
    assert "semillas" in table.loc[table.modelo=="PCA + K-Means","estabilidad"].iloc[0]

def test_streamlit_views_model_actions_and_scaled_download():
    from streamlit.testing.v1 import AppTest
    app_path=Path(__file__).resolve().parents[1]/"app.py"
    app=AppTest.from_file(str(app_path),default_timeout=120).run()
    options=list(app.radio[0].options)
    assert len(options)==10 and not app.exception
    for option in options:
        app.radio[0].set_value(option).run()
        assert not app.exception, (option,app.exception[0].message if app.exception else "")
    for option in (options[4],options[5]):
        app.radio[0].set_value(option).run()
        app.button[0].click().run()
        app.button[1].click().run()
        assert not app.exception, (option,app.exception[0].message if app.exception else "")
    for option in (options[6],options[7]):
        app.radio[0].set_value(option).run()
        app.button[0].click().run()
        assert not app.exception, (option,app.exception[0].message if app.exception else "")
    for option in (options[8],options[9]):
        app.radio[0].set_value(option).run()
        assert not app.exception, (option,app.exception[0].message if app.exception else "")
    app.radio[0].set_value(options[4]).run()
    app.slider(key="km_k").set_value(4).run()
    app.radio[0].set_value(options[8]).run()
    current=next(frame.value for frame in app.dataframe if "estado" in frame.value.columns)
    states=current.set_index("modelo")["estado"].to_dict()
    assert states["K-Means"]=="Pendiente"
    assert states["DBSCAN"]=="Ejecutado" and states["PCA + DBSCAN"]=="Ejecutado"
    assert not app.exception
    app.radio[0].set_value(options[3]).run()
    assert any(button.label=="Descargar matriz estandarizada" for button in app.get("download_button"))
    assert not app.exception

