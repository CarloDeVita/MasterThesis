# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 04:35:02 2020

@author: cdevi
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 02:50:45 2020

@author: cdevi
"""

#from __future__ import absolute_import, division, print_function, unicode_literals

# Install TensorFlow

import tensorflow as tf
import util
from cdl import myCDL
import grafici as gr
import numpy as np


x_train, b_train, y_train, x_test, b_test, y_test = util.loadFashionMNIST()
xl0,xl1,xl2,xl3,xl4,xl5,xl6,xl7,xl8,xl9=util.groupMNIST(x_test[:2000],y_test[:2000])
l0,l1,l2,l3,l4,l5,l6,l7,l8,l9=util.groupMNIST(b_test[:2000],y_test[:2000])
imgs = [l0,l1,l2,l3,l4,l5,l6,l7,l8,l9]
#imgs= [xl0,xl1,xl2,xl3,xl4,xl5,xl6,xl7,xl8,xl9]
#numero test
print("---trovo dizionario--")
dicts=[]

tnumber=100
nfilter=16
dimfilter=5

#for i in range(10):
#    im_tr = imgs[i]
#    diz = myCDL(im_tr[:,:,:50],dimfilter,nfilter)
#    diz.findDictionary(1000)
#    util.saveFile(diz,f"dizFashionMNIST_{i} 16x9.dump")
#    dicts.append(diz)
#    gr.print_filters(diz.D,nfilter,f'filtri_{i}');

dicts=util.takeDictEachNumber('FashionMNIST','16x5')
for i in range(10):
    diz=dicts[i]
    gr.print_filters(diz.D,nfilter,f'filtri_{i}');
    
print("--creo Neural network--")
#model = tf.keras.models.Sequential([
#  tf.keras.layers.Flatten(input_shape=(28, 28)),
#  tf.keras.layers.Dense(128, activation='relu'),
#  tf.keras.layers.Dropout(0.2),
#  tf.keras.layers.Dense(10, activation='softmax')
#])
#
#model.compile(optimizer='adam',
#              loss='sparse_categorical_crossentropy',
#              metrics=['accuracy'])
#
#
#model.fit(x_train, y_train, epochs=10)
#tf.keras.models.save_model(model,'FashionMNIST.h5')
model=tf.keras.models.load_model('FashionMNIST.h5')
#img da dare come training
    
prove=25
    
    
    
for i in range(500,500+prove): 
    
    dr = f"info {i}"
    util.mkdir_p(dr)  
    img=x_test[i]
    imgHF=b_test[i]
    imgLF=x_test[i]-b_test[i]
    gr.print_save_img(img,f"{dr}/img originale.png")
    gr.print_save_img(imgHF,f"{dr}/img processata.png")
    diz=dicts[y_test[i]]
    #diz=dicts[0]
    c=diz.findHeatmaps(imgHF)
    cnv=gr.print_conv(diz.D,dimfilter,c,nfilter,diz.d,f"{dr}/convoluzioni.png")
    
    values1=np.zeros(nfilter) #APPROCCIO SINGOLARE
    values2=np.zeros(nfilter) #APPROCCIO SOTTRAZIONE
    
    classificazione=model.predict(img.reshape((1,28,28)));
    valoreclasse=np.max(classificazione,1)[0]   
    classe=np.argmax(classificazione,1)[0]
    gr.saveSomeInfo(f"{dr}",classe,valoreclasse) 
    for f in range(nfilter):
        curr=cnv[f]
        exc=imgLF+curr
        p = model.predict(exc.reshape((1,28,28)))
        values1[f]= p[0,y_test[i]]
        rec=cnv.sum(axis=0);
        exc=imgLF+rec-curr
#        exc=rec-cnv[f]
        p = model.predict(exc.reshape((1,28,28)))
        values2[f]= p[0,y_test[i]]
    
#    ind=np.argsort(values1)[::-1]
#    imp_cnv1 = np.dstack((cnv[ind[0]],cnv[ind[1]],cnv[ind[2]],cnv[ind[3]],cnv[ind[4]]))
#    imp_cnv1=imp_cnv1.transpose(2,0,1)
#    gr.print_n_imgs(imp_cnv1,5,f"{dr}/convoluzioni importanti approccio sing.png")
#    tot=imp_cnv1.sum(axis=0)
#    gr.print_save_img(tot,f"{dr}/somma conv imp approccio sing.png")
#    
#    
#    ind=np.argsort(values2)
#    imp_cnv2 = np.dstack((cnv[ind[0]],cnv[ind[1]],cnv[ind[2]],cnv[ind[3]],cnv[ind[4]]))
#    imp_cnv2=imp_cnv2.transpose(2,0,1)
#    gr.print_n_imgs(imp_cnv2,5,f"{dr}/convoluzioni importanti approccio sottr.png")
#    tot=imp_cnv2.sum(axis=0)
#    gr.print_save_img(tot,f"{dr}/somma conv imp approccio sottr.png")
    
    ind=np.argsort(values1)[::-1]
    imp_cnv1 = np.dstack((cnv[ind[0]],cnv[ind[1]],cnv[ind[2]],cnv[ind[3]],cnv[ind[4]]))
    imp_cnv1=imp_cnv1.transpose(2,0,1)
    gr.print_n_imgs(imp_cnv1,5,f"{dr}/convoluzioni importanti approccio sing.png")
    tot=imp_cnv1.sum(axis=0)
    gr.print_save_img(tot,f"{dr}/somma conv imp approccio sing.png")
    
    
    ind=np.argsort(values2)
    #imp_cnv2 = np.dstack((cnv[ind[0]],cnv[ind[1]],cnv[ind[2]],cnv[ind[3]],cnv[ind[4]]))
    imp_cnv2=util.relConvs(5,cnv,ind)
    imp_cnv2=imp_cnv2.transpose(2,0,1)
    gr.print_n_imgs(imp_cnv2,5,f"{dr}/convoluzioni importanti approccio sottr.png")
    tot=imp_cnv2.sum(axis=0)
    gr.print_save_img(tot,f"{dr}/somma conv imp approccio sottr.png")
    sumRelevantCoeffs=util.relCoeffs(5,c,ind)
    gr.print_save_img(sumRelevantCoeffs,f"{dr}/somma coefficienti imp approccio sottr.png",color='Blues')



