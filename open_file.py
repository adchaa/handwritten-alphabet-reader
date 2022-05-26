import pandas as pd
import numpy as np
class open_file:
    def __init__(self,path):
        self.path = path
    def reshape_image(self,imgs,max):
        tmp = np.zeros((max,28,28),dtype=np.ubyte)
        for i in range(max):
            tmp[i] = np.reshape(imgs[i],(28,28))
        return tmp
    #return array of number of occurances of each character
    def desc(self,labels):
        chara = np.zeros((26),dtype=np.ushort)
        for x in labels:
            chara[x] += 1
        return chara
    def split_train_test(self,images,labels,array_char):
        x=0
        x_train = np.empty((0,28,28),dtype=np.ubyte)
        x_test = np.empty((0,28,28),dtype=np.ubyte)
        y_train = np.array([],dtype=np.ubyte)
        y_test = np.array([],dtype=np.ubyte)
        for i in array_char:
            x_train =np.concatenate((x_train,images[x:x+round(i*0.9),:,:]),dtype=np.ubyte)
            x_test =np.concatenate((x_test,images[x+round(i*0.9):x+i,:,:]),dtype=np.ubyte)
            y_train =np.concatenate((y_train,labels[x:x+round(i*0.9)]),dtype=np.ubyte)
            y_test =np.concatenate((y_test,labels[x+round(i*0.9):x+i]),dtype=np.ubyte)
            x +=i
        
        return (x_train,y_train),(x_test,y_test)






    def load_data(self):
        file = pd.read_csv(self.path,dtype=np.ubyte)
        label =file.pop("0")
        max = file.shape[0]
        images = pd.DataFrame.to_numpy(file,dtype=np.ubyte)
        labels = pd.Series.to_numpy(label,dtype=np.ubyte)
        # for optimization
        del file 
        del label
        images = self.reshape_image(images,max)
        array_char =self.desc(labels)
        return self.split_train_test(images,labels,array_char)

        

        





        

        

