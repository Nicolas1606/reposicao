"""Previsão e reposição periódica. Rode a partir da raiz do projeto."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

KEY = ["loja","produto"]


def carregar(path):
    d = pd.read_csv(path)
    if d[KEY+["semana","vendas"]].isna().any().any() or d.duplicated(KEY+["semana"]).any():
        raise ValueError("Chaves duplicadas ou campos ausentes.")
    v = d.pivot(index=KEY,columns="semana",values="vendas").sort_index(axis=1)
    if list(v.columns)!=list(range(v.shape[1])) or v.shape[1]<52:
        raise ValueError("Forneça semanas consecutivas desde zero e pelo menos 52 semanas.")
    a=v.to_numpy(dtype=float)
    if not np.isfinite(a).all() or (a<0).any() or (a!=np.floor(a)).any():
        raise ValueError("Vendas devem ser inteiras, completas e não negativas.")
    return v.index,a


def atributos(a,t):
    hist=a[:,t-8:t]
    return np.column_stack([hist[:,-1],hist[:,-2],hist[:,-4],hist[:,-4:].mean(1),
                            hist.mean(1),hist.std(1),hist[:,-4:].mean(1)-hist[:,:4].mean(1)])


def prever(a,t,h,metodo):
    if metodo=="ciclo4":
        return sum(a[:,t-4+i%4] for i in range(h))
    # Para cada origem histórica, todos os h alvos já terminaram antes de t.
    origins=range(8,t-h+1)
    x=np.vstack([atributos(a,i) for i in origins])
    y=np.concatenate([a[:,i:i+h].sum(1) for i in origins])
    m=make_pipeline(StandardScaler(),Ridge(alpha=20)).fit(x,y)
    return m.predict(atributos(a,t)).clip(0)


def ajustar(a,h,metodo):
    errors=[]
    for t in range(25,36-h+1):
        p=prever(a,t,h,metodo)
        errors.extend((a[:,t:t+h].sum(1)-p)/np.sqrt(1+p))
    return np.quantile(errors,.90)


def comprar(alvo,saldo,transito):
    return np.maximum(0,np.ceil(alvo)-saldo-transito).astype(int)


def simular(a,index,inicio,fim,prazo,metodo,q):
    h=prazo+1
    saldo=np.ceil(a[:,inicio-4:inicio].mean(1)*h)
    fila=[np.zeros(len(a)) for _ in range(prazo)]
    rows=[]
    for t in range(inicio,fim):
        saldo=saldo+fila.pop(0)
        p=prever(a,t,h,metodo)
        alvo=(p+q*np.sqrt(1+p)).clip(0)
        pedido=comprar(alvo,saldo,sum(fila))
        fila.append(pedido)
        vendas=a[:,t]
        perda=(vendas-saldo).clip(0)
        saldo=(saldo-vendas).clip(0)
        real=a[:,t:t+h].sum(1) if t+h<=fim else np.full(len(a),np.nan)
        rows.append(pd.DataFrame(dict(semana=t,vendas=vendas,perda=perda,saldo=saldo,pedido=pedido,
                     real_h=real,previsto=p,alvo=alvo),index=index).reset_index())
    return pd.concat(rows,ignore_index=True)


def resumo(d):
    f=d.dropna(subset=["real_h"])
    return dict(wape=(f.real_h-f.previsto).abs().sum()/f.real_h.sum(),
                cobertura=(f.real_h<=f.alvo).mean(),atendimento=1-d.perda.sum()/d.vendas.sum(),
                estoque_medio=d.saldo.mean(),faltas=d.perda.sum(),custo=(d.saldo+8*d.perda).sum(),
                custo_falta4=(d.saldo+4*d.perda).sum(),custo_falta16=(d.saldo+16*d.perda).sum())


def estudar(idx,a,out):
    out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for prazo in [1,2]:
        for metodo in ["ciclo4","ridge"]:
            q=ajustar(a,prazo+1,metodo)
            for buffer,k in [("pontual",0),("q90",q)]:
                for split,start,end in [("validacao",36,42),("teste",42,52)]:
                    d=simular(a,idx,start,end,prazo,metodo,k)
                    rows.append(dict(prazo=prazo,modelo=metodo,alvo=buffer,periodo=split,**resumo(d)))
                    if prazo==1 and split=="teste":
                        d.to_csv(out/f"auditoria_{metodo}_{buffer}.csv.gz",index=False)
    r=pd.DataFrame(rows)
    r.to_csv(out/"comparacao.csv",index=False)
    best=r.query("periodo=='validacao' and prazo==1").sort_values("custo").iloc[0]
    print(r.query("periodo=='teste' and prazo==1").to_string(index=False))
    print("Política selecionada em validação:",best.modelo,best.alvo)
    (out/"politica.txt").write_text(best.modelo+" "+best.alvo,encoding="utf8")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("acao",choices=["estudar","comprar"])
    p.add_argument("--historico",default="dados/vendas.csv")
    p.add_argument("--saida",type=Path,required=True)
    p.add_argument("--estado",type=Path)
    p.add_argument("--politica",type=Path,default=Path("resultados/politica.txt"))
    args=p.parse_args()
    idx,a=carregar(args.historico)
    if args.acao=="estudar":
        estudar(idx,a,args.saida)
    else:
        if not args.estado: p.error("Informe --estado com saldo após recebimentos e trânsito.")
        s=pd.read_csv(args.estado).set_index(KEY,verify_integrity=True)
        if set(s.index)!=set(idx): p.error("Estado e histórico precisam cobrir os mesmos itens.")
        s=s.reindex(idx)
        for c in ["saldo","transito"]:
            if not np.isfinite(s[c]).all() or (s[c]<0).any() or (s[c]%1!=0).any(): p.error("Estado inválido.")
        m,b=args.politica.read_text(encoding="utf8").split()
        if m not in ["ciclo4","ridge"] or b not in ["q90","pontual"]: p.error("Política inválida.")
        point=prever(a,a.shape[1],2,m)
        q=ajustar(a,2,m) if b=="q90" else 0
        s["alvo"]=(point+q*np.sqrt(1+point)).clip(0)
        s["pedido"]=comprar(s.alvo,s.saldo,s.transito)
        s.to_csv(args.saida,mode="x")

