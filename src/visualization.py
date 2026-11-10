"""Figuras Plotly reutilizables."""
import plotly.express as px

def scatter_scores(scores, labels, title="Clusters"):
    data=scores.iloc[:,:2].copy(); data["grupo"]=labels
    return px.scatter(data,x=data.columns[0],y=data.columns[1],color=data["grupo"].astype(str),title=title)

