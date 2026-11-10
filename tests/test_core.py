import numpy as np
import pandas as pd
import pytest
from src.integration import integrate, DuplicateKeyError
from src.preprocessing import prepare_features
from src.kmeans_model import fit_kmeans
from src.dbscan_model import fit_dbscan
from src.pca_model import fit_pca
from src.evaluation import clustering_metrics
from src.profiling import profile_clusters
from src.kmeans_stability import assess_kmeans_stability
from src.dbscan_stability import assess_dbscan_stability
from src.model_comparison import compare_models

def sample_frames():
    activity=pd.DataFrame({"Id":["01","01","02"],"ActivityDate":["1/1/2024","1/2/2024","1/1/2024"],"steps":[100,200,300]})
    sleep=pd.DataFrame({"Id":["01","01","01","02"],"SleepDay":["1/1/2024 12:00:00 AM","1/1/2024 12:00:00 AM","1/2/2024","1/1/2024"],"minutes":[400,400,420,350]})
    return activity,sleep

def test_integration_exact_dedup_and_unique():
    a,s=sample_frames(); merged,a0,s0,audit,_,_=integrate(a,s)
    assert len(merged)==3 and audit.loc[audit.Indicador=="Duplicados exactos sueño","Valor"].iloc[0]==1
    assert not merged.duplicated(["Id","Date"]).any()
    assert len(a0)==0 and len(s0)==0 and merged.Id.dtype.name=="string"

def test_integration_conflicting_key_raises():
    a,s=sample_frames(); s.loc[1,"minutes"]=410
    with pytest.raises(DuplicateKeyError): integrate(a,s)

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
    prof=profile_clusters(pd.DataFrame(X,columns=["a","b"]),labels); assert "mean" in prof.columns.get_level_values(1)

def test_comparison_and_stability_contracts():
    X=np.array([[0.,0.],[.1,.2],[4.,4.],[4.2,4.1],[8.,8.],[8.1,8.2]])
    km,labels=fit_kmeans(X,2,n_init=10)
    table=compare_models(X,{"kmeans":labels}); assert table.loc[0,"n_evaluated"]==len(X)
    runs,ari=assess_kmeans_stability(X,k=2,seeds=(1,2),n_init=10); assert len(runs)==2 and 0<=ari["ari_mean"]<=1
    configs,pairs=assess_dbscan_stability(X,[.3,.5],[2]); assert len(configs)==2 and "ari_including_noise" in pairs

