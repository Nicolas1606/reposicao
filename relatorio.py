import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def gerar(pasta):
    modelo,alvo=(pasta/"politica.txt").read_text().split()
    d=pd.read_csv(pasta/f"auditoria_{modelo}_{alvo}.csv.gz")
    def indicadores(g):
        total=g.vendas.sum()
        f=g.dropna(subset=["real_h"])
        den=f.real_h.sum()
        return pd.Series({"atendimento":1-g.perda.sum()/total if total else float("nan"),
                          "estoque_medio":g.saldo.mean(),"custo":(g.saldo+8*g.perda).sum(),
                          "wape":(f.real_h-f.previsto).abs().sum()/den if den else float("nan"),
                          "cobertura":(f.real_h<=f.alvo).mean()})
    for keys,name in [(["loja","produto"],"itens"),(["semana"],"semanas")]:
        d.groupby(keys).apply(indicadores).to_csv(pasta/f"monitor_{name}.csv")
    r=pd.read_csv(pasta/"comparacao.csv").query("prazo==1 and periodo=='teste'").copy()
    fig,ax=plt.subplots(figsize=(8,4),layout="constrained")
    ax.barh(r.modelo+" · "+r.alvo,r.custo,color=["#82979c","#82979c","#248177","#14584e"])
    ax.set(xlabel="Unidades de custo simulado",title="Custo de estoque e falta nas dez semanas de teste")
    fig.savefig(pasta/"painel.png",dpi=160)
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("pasta",type=Path)
    gerar(p.parse_args().pasta)

