import numpy as np
import matplotlib.pyplot as plt
import glob
import sklearn.preprocessing
import sklearn.datasets
import sklearn.neural_network
import sklearn.model_selection
#Se utiliza nuestro dataset de las imagenes de frutas, sacadas de  https://www.kaggle.com/moltean/fruits
#Por la rapidez del algoritmo solo se tomo del folder test, de una sola fruta, 13 frutas. 
#Organizamos las frutas en un dataset en el que les cambiamos la dimensionalidad y le asignamos a cada una de las frutas un numero
numeros = sklearn.datasets.load_digits()
imagenes = numeros['images']  # Hay 1797 digitos representados en imagenes 8x8
n_imagenes = len(imagenes)
X = imagenes.reshape((n_imagenes, -1)) # para volver a tener los datos como imagen basta hacer data.reshape((n_imagenes, 8, 8))
Y = numeros['target']
print(np.shape(X), np.shape(Y))

#Separamos y preaparamos los datos en 50% train y 50% test
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.5)
scaler = sklearn.preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Realizamos las redes de neuronas, de una capa escondida, de 1 a 20 neuronas. Guardando el loss y el F1
loss=[]
F1_train = []
F1_test = []
for i in range(20):
    n = i+1
    mlp = sklearn.neural_network.MLPClassifier(activation='logistic', hidden_layer_sizes=(n,),max_iter=2500)
    mlp.fit(X_train, Y_train)
    loss.append(mlp.loss_)
    F1_train.append(sklearn.metrics.f1_score(Y_train, mlp.predict(X_train), average='macro'))
    F1_test.append(sklearn.metrics.f1_score(Y_test, mlp.predict(X_test), average='macro'))
    print('Se completaron %i neuronas'% n)

#Graficamos la evolucion de la funcion de perdida y de F1
x = np.linspace(0,20,20)
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.plot(x,loss)
plt.title('Evolucion de la funcion de perdida')
plt.xlabel('Numero de neuronas')
plt.ylabel('Loss_')
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(x,F1_test, label='Test')
plt.plot(x,F1_train, label='Train')
plt.grid(True)
plt.legend()
plt.title('Evolucion de la funcion de F1 sobre el test')
plt.xlabel('Numero de neuronas')
plt.ylabel('F1')

plt.savefig('loss_f1.png')

#Realizamos el fit con el mejor numero de neuronas
mlp_bueno = sklearn.neural_network.MLPClassifier(activation='logistic', hidden_layer_sizes=(6,),max_iter=2500)
mlp_bueno.fit(X_train, Y_train)
#Vamos ahora a construir las neuronas 
print(np.shape(mlp_bueno.coefs_[0]))
plt.figure(figsize=(10,10))
num =np.shape(mlp_bueno.coefs_[0])[1]
scale = np.max(mlp_bueno.coefs_[0])
for i in range(num):
    lineas = int(num/3)+1
    n = i+1
    plt.subplot(lineas,3,i+1)
    plt.imshow(mlp_bueno.coefs_[0][:,i].reshape(8,8),cmap=plt.cm.RdBu, vmin=-scale, vmax=scale)
    plt.title('Neurona %i'% n)

plt.savefig('neuronas.png')