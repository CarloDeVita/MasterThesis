import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig

def print_img(img):
    fig=plt.figure()
    plt.imshow(img,cmap='gray')
    fig.show()
    
def print_save_img(img,path,color='gray'):
    fig=plt.figure()
    plt.imshow(img,cmap=color)
    fig.show()
    fig.savefig(path)
    
    
def print_n_imgs(imgs,n_imgs,path):
    fig = plt.figure(figsize=(10,10))
    cols = 5;
    rows = np.ceil(n_imgs/cols)
    for i in range(n_imgs):
        a=fig.add_subplot(rows,cols,i+1)
        a.set_title(f'imp conv {i+1}')
        a.text(0.5,-0.1, "(b) my other label", size=12, ha="center",transform=a.transAxes)
        plt.imshow(imgs[i],cmap='gray');
        plt.axis('off')
    
    fig.show()
    fig.savefig(path)

def print_n_imgs_with_relevance(imgs,relevance,n_imgs,path):
    fig = plt.figure(figsize=(10,10))
    cols = 5;
    rows = np.ceil(n_imgs/cols)
    for i in range(n_imgs):
        a=fig.add_subplot(rows,cols,i+1)
        a.set_title(f'imp conv {i+1}')
        a.text(0.5,-0.1, relevance[i], size=12, ha="center",transform=a.transAxes)
        plt.imshow(imgs[i],cmap='gray');
        plt.axis('off')
    
    fig.show()
    fig.savefig(path)

def print_imgs(img1,img2,path):
    fig = plt.figure(figsize=(14, 7))
    s1=fig.add_subplot(1, 2, 1)
    s1.set_title("Original")
    plt.imshow(img1,cmap='gray')
    s2=fig.add_subplot(1, 2, 2)
    rec = img2
    s2.set_title('Reconstructed')
    plt.imshow(rec,cmap='gray')
    fig.show()
    
    mse = np.square(np.subtract(rec, img1)).mean()
    print("MSE=",mse)
    
    fig.savefig(path)
    
    
def print_imgs_col(img1,img2,path):
    fig = plt.figure(figsize=(14, 7))
    s1=fig.add_subplot(1, 2, 1)
    s1.set_title("Original")
    plt.imshow(img1)
    s2=fig.add_subplot(1, 2, 2)
    rec = img2
    s2.set_title('Reconstructed')
    plt.imshow(rec)
    fig.show()
    
    mse = np.square(np.subtract(rec, img1)).mean()
    print("MSE=",mse)
    
    fig.savefig(path)

def print_filters(D,nfilter,path):
    fig=plt.figure(figsize=(10,10))
    c=np.ceil(np.sqrt(nfilter))
    D = D - D.min()
    D = D / D.max()
    for f in range(0,nfilter):
        a = fig.add_subplot(c,c,f+1)
        a.set_title(f'filtro {f+1}')
        plt.imshow((D[:,:,f]),cmap='gray')   
        plt.axis('off')

    plt.subplots_adjust(hspace=1)
    fig.show()
    fig.savefig(path)
    
def print_filters_col(D,nfilter):
    fig=plt.figure(figsize=(10,10))
    c=np.ceil(np.sqrt(nfilter))
    D = D - D.min()
    D = D / D.max()
    for f in range(0,nfilter):
        a = fig.add_subplot(c,c,f+1)
        a.set_title(f'filtro {f+1}')
        plt.imshow((D[:,:,:,f] * 255).astype(np.uint8))
        plt.axis('off')

    plt.subplots_adjust(hspace=1)
    fig.show()
    fig.savefig("filtri.png")
    
def print_coeff_noAbs(X,nfilter,path1,path2):
    fig=plt.figure(figsize=(10,10))
    c=np.ceil(np.sqrt(nfilter))
    sum = np.zeros((28,28))
    for f in range(0,nfilter):
        a = fig.add_subplot(c,c,f+1)
        a.set_title(f'coeff {f+1}')
        z = X[:,:,0,0,f]
        plt.imshow(z,cmap='Blues')   
        plt.axis('off')
        sum = sum + z
    
    plt.subplots_adjust(hspace=1)
    fig.show()
    fig.savefig(path1)
    
    fig = plt.figure()
    plt.imshow(sum, cmap='gray',vmin=sum.min(),vmax=sum.max())
    fig.suptitle("Sparse representation")
    fig.show()
    fig.savefig(path2)

def print_coeff(X,nfilter,path1,path2):
    fig=plt.figure(figsize=(10,10))
    c=np.ceil(np.sqrt(nfilter))
    sum=sum = np.zeros((28,28))
    for f in range(0,nfilter):
        a = fig.add_subplot(c,c,f+1)
        a.set_title(f'coeff {f+1}')
        z = X[:,:,0,0,f]
        plt.imshow(abs(z),cmap='gray')   
        plt.axis('off')
        sum = sum + abs(z)
    
    plt.subplots_adjust(hspace=1)
    fig.show()
    fig.savefig(path1)
    
    fig = plt.figure()
    plt.imshow(sum, cmap='gray',vmin=sum.min(),vmax=sum.max())
    fig.suptitle("Sparse representation")
    fig.show()
    fig.savefig(path2)
    return sum;


def print_prod_coeff_img(img,sparse,path):
    fig=plt.figure(figsize=(10,10))
    z = np.multiply(img,sparse)
    plt.imshow(z,cmap='gray',vmin=z.min(),vmax=z.max())
    fig.show()
    fig.savefig(path)
    
    
    
def print_coeff_norm(X,nfilter,path1,path2):
    fig=plt.figure(figsize=(10,10))
    c=np.ceil(np.sqrt(nfilter))
    sum=sum = np.zeros((28,28))
    for f in range(0,nfilter):
        a = fig.add_subplot(c,c,f+1)
        a.set_title(f'coeff {f+1}')
        
        z = X[:,:,0,0,f]
        z1 = z
        z1 = z1 - z1.min()
        z1 = z1 / z1.max()
        plt.imshow(z1,cmap='gray',vmin=z1.min(),vmax=z1.max())   
        plt.axis('off')
        sum = sum + z1
    
    plt.subplots_adjust(hspace=1)
    fig.show()
    fig.savefig(path1)
    
    fig = plt.figure()
    plt.imshow(sum, cmap='gray',vmin=sum.min(),vmax=sum.max())
    fig.suptitle("Sparse representation")
    fig.show()
    fig.savefig(path2)
    

def print_coeff_noNegative(X,nfilter,path1,path2):
    fig=plt.figure(figsize=(10,10))
    c=np.ceil(np.sqrt(nfilter))
    sum=sum = np.zeros((28,28))
    for f in range(0,nfilter):
        a = fig.add_subplot(c,c,f+1)
        a.set_title(f'coeff {f+1}')
        z = X[:,:,0,0,f]
        z1 = np.maximum(z,0)
        plt.imshow(z1,cmap= 'Blues')  
        plt.axis('off')
        sum = sum + z1
    
    plt.subplots_adjust(hspace=1)
    fig.show()
    fig.savefig(path1)
    
    fig = plt.figure()
    plt.imshow(sum, cmap='gray',vmin=sum.min(),vmax=sum.max())
    fig.suptitle("Sparse representation")
    fig.show()
    fig.savefig(path2)
    



def print_conv(D,dim,X,nfilter,d,path):
    fig=plt.figure(figsize=(10,10))
    c=np.ceil(np.sqrt(nfilter))
    #print("DIM:", c)
    #cols = 2;
    #rows = np.ceil(nfilter/2)
    sum = np.zeros((28,28))
    convolutions = np.zeros((nfilter,28,28))
    for f in range (0,nfilter):
        Xn=np.reshape(X[:,:,0,0,f],(28,28,1,1,1))
        Dn=np.reshape(D[:,:,f],(dim,dim,1,1,1))
        #    print(Xn.shape, Dn.shape)
        z = d.reconstruct(D=Dn,X=Xn).squeeze()   
        a = fig.add_subplot(c,c,f+1)
        a.set_title(f'convolution {f+1}')
        plt.imshow((z),cmap='gray') 
        convolutions[f]=z
        
        
        plt.axis('off')
        sum=sum+z
    
    plt.subplots_adjust(hspace=1)
    fig.show()
    fig.savefig(path) 
    return convolutions


def print_conv_resized(D,dim,X,nfilter,d,path):
    fig=plt.figure(figsize=(10,10))
    c=np.ceil(np.sqrt(nfilter))
    #print("DIM:", c)
    #cols = 2;
    #rows = np.ceil(nfilter/2)
    dimImm=64
    sum = np.zeros((dimImm,dimImm))
    convolutions = np.zeros((nfilter,dimImm,dimImm))
    for f in range (0,nfilter):
        Xn=np.reshape(X[:,:,0,0,f],(dimImm,dimImm,1,1,1))
        Dn=np.reshape(D[:,:,f],(dim,dim,1,1,1))
        #    print(Xn.shape, Dn.shape)
        z = d.reconstruct(D=Dn,X=Xn).squeeze()   
        a = fig.add_subplot(c,c,f+1)
        a.set_title(f'convolution {f+1}')
        plt.imshow((z),cmap='gray') 
        convolutions[f]=z
        
        plt.axis('off')
        sum=sum+z
    
    plt.subplots_adjust(hspace=1)
    fig.show()
    fig.savefig(path) 
    return convolutions

def print_conv2(D,dim,X,nfilter,d,path):
    fig=plt.figure(figsize=(10,10))
    c=np.ceil(np.sqrt(nfilter))
    #print("DIM:", c)
    #cols = 2;
    #rows = np.ceil(nfilter/2)
    sum = np.zeros((28,28))
    
    for f in range (0,nfilter):
        r1=sig.convolve2d(X[:,:,0,0,f],D[:,:,f],mode='same')
         
        a = fig.add_subplot(c,c,f+1)
        a.set_title(f'convolution {f+1}')
        plt.imshow(r1,cmap='gray') 
        plt.axis('off')
        sum=sum+r1
    
    plt.subplots_adjust(hspace=1)
    fig.show()
    fig.savefig(path) 

def print_conv_norm(D,dim,X,nfilter,d,path):
    fig=plt.figure(figsize=(10,10))
    c=np.ceil(np.sqrt(nfilter))
    #print("DIM:", c)
    #cols = 2;
    #rows = np.ceil(nfilter/2)
    sum = np.zeros((28,28))
    for f in range (0,nfilter):
        Xn=np.reshape(X[:,:,0,0,f],(28,28,1,1,1))
        Dn=np.reshape(D[:,:,f],(dim,dim,1,1,1))
        #    print(Xn.shape, Dn.shape)
        z = d.reconstruct(D=Dn,X=Xn).squeeze()   
        a = fig.add_subplot(c,c,f+1)
        a.set_title(f'convolution {f+1}')
        z1 = z
        z1 = z1 - z1.min()
        z1 = z1 / z1.max() 
        plt.imshow(z1,cmap='gray',vmin=z1.min(),vmax=z1.max()) 
    
        
        plt.axis('off')
        sum=sum+z
    
    plt.subplots_adjust(hspace=1)
    fig.show()
    fig.savefig(path) 
    
def print_conv_col(D,dim,X,nfilter,d,path):
    fig=plt.figure(figsize=(10,10))
    c=np.ceil(np.sqrt(nfilter))
    #print("DIM:", c)
    #cols = 2;
    #rows = np.ceil(nfilter/2)
    sum = np.zeros((32,32,3))
    for f in range (0,nfilter):
        Xn=np.reshape(X[:,:,:,0,f],(28,28,1,1,1))
        Dn=np.reshape(D[:,:,:,f],(dim,dim,3,1,1))
        #    print(Xn.shape, Dn.shape)
        z = d.reconstruct(D=Dn,X=Xn)
        z=z.squeeze() 
        a = fig.add_subplot(c,c,f+1)
        a.set_title(f'convolution {f+1}')
        z = z - z.min()
        z = z / z.max()
        plt.imshow(z)
        
        plt.axis('off')
        sum=sum+z
        
    plt.subplots_adjust(hspace=1)
    fig.show()
    fig.savefig(path)
    return sum

def saveSomeInfo(path,classe,valoreclasse):
    file = open(f"{path}/sample.txt","w")
    file.write(f"risultato rete è {classe}, con valore {valoreclasse}")