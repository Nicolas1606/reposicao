import unittest
import numpy as np
import pandas as pd
from estoque import comprar,prever,simular


class TestEstoque(unittest.TestCase):
    def test_transito(self):
        np.testing.assert_array_equal(comprar(np.array([8.2,2]),3,2),[4,0])

    def test_futuro_invisivel(self):
        a=np.arange(80.).reshape(2,40)
        b=a.copy()
        b[:,25:]=10000
        for modelo in ["ridge","ciclo4"]:
            np.testing.assert_allclose(prever(a,25,2,modelo),prever(b,25,2,modelo))

    def test_chegada_e_conservacao(self):
        a=np.ones((1,40))*5
        a[0,30]=20
        idx=pd.MultiIndex.from_tuples([("A","X")],names=["loja","produto"])
        for prazo in [1,2]:
            d=simular(a,idx,30,37,prazo,"ciclo4",0)
            anterior=np.r_[5*(prazo+1),d.saldo.values[:-1]]
            chegadas=d.pedido.shift(prazo,fill_value=0)
            np.testing.assert_allclose(anterior+chegadas-d.vendas+d.perda,d.saldo)
            self.assertTrue((d.saldo>=0).all())

