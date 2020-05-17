from __future__ import print_function

#import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np

from sporco.dictlrn import cbpdndl
from sporco import util
from sporco import plot
from sporco.admm import cbpdn
#import sporco.admm.cbpdn as cbpdn
#import sporco.dictlrn.cbpdndl as cbpdndl
### Classe dizionario
class myCDL:
    
    
    def __init__(self, imgs, dimFilter,nFilters):
        self.imgs=imgs
        self.dimFilter=dimFilter
        self.nFilters=nFilters
        
        
        
    def findDictionary(self,epochs):

#        #Filtro di tikhonov 
#        npd = 16
#        fltlmbd = 5
#        sl, sh = util.tikhonov_filter(self.imgs, fltlmbd, npd)
#        #plot.imview(sl[:,:,1],title='IMG low');
        
        #Creazione di un dizionario random
        
        np.random.seed(12345)
        D0 = np.random.randn(self.dimFilter, self.dimFilter,self.nFilters)
        
        #Opzioni per l'algoritmo DL e Soluzione del problema
#        lmbda = 0.1
        lmbda =0.1
        opt = cbpdndl.ConvBPDNDictLearn.Options({'Verbose': True, 'MaxMainIter': epochs,
                                    'CBPDN': {'rho': 50.0*lmbda + 0.5},
                                    'CCMOD': {'rho': 10.0, 'ZeroMean': True}},
                                    dmethod='cns')
        
        d = cbpdndl.ConvBPDNDictLearn(D0,self.imgs, lmbda, opt, dmethod='cns')
        D1 = d.solve()
        print("ConvBPDNDictLearn solve time: %.2fs" % d.timer.elapsed('solve'))
        
        D1=D1.squeeze()
        self.D=D1
        self.coef=d.getcoef()
        self.rec=d.reconstruct().squeeze();
        self.d=d #riferimento alla libreria 



    def findDictionary1(self,epochs):
        
        #Creazione di un dizionario random
        
        np.random.seed(12345)
        D0 = np.random.randn(self.dimFilter, self.dimFilter,self.nFilters)
        
        
        lmbda = 0.2
        L_sc = 360.0
        L_du = 50.0
        opt = cbpdndl.ConvBPDNDictLearn.Options({
                        'Verbose': True, 'MaxMainIter': epochs, 'DictSize': D0.shape,
                        'CBPDN': {'BackTrack': {'Enabled': True }, 'L': L_sc},
                        'CCMOD': {'BackTrack': {'Enabled': True }, 'L': L_du}},
                        xmethod='fista', dmethod='fista')
        d = cbpdndl.ConvBPDNDictLearn(D0, self.imgs, lmbda, opt, xmethod='fista',dmethod='fista')
        D1 = d.solve()
        print("ConvBPDNDictLearn solve time: %.2fs" % d.timer.elapsed('solve'))
        D1=D1.squeeze()
        self.D=D1
        self.coef=d.getcoef()
        self.rec=d.reconstruct().squeeze();
        self.d=d #riferimento alla libreria
    
    def findHeatmaps(self,img):
        #Opzioni e soluzione dell'algoritmo di CSC
        lmbda = 5e-2
        opt = cbpdn.ConvBPDN.Options({'Verbose': True, 'MaxMainIter': 100, 
                                  'RelStopTol': 5e-3, 'AuxVarObj': False})
            
        csc = cbpdn.ConvBPDN(self.D,img, lmbda, opt, dimK=0)
        X = csc.solve()
        print(X.shape)
        print("ConvBPDN solve time: %.2fs" % csc.timer.elapsed('solve'))
        #rec = csc.reconstruct().squeeze()
        return X