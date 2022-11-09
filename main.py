import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras import Input, Model
from keras.layers import Dense
from keras.utils.vis_utils import plot_model

pod = np.loadtxt('podaciCas02.txt')

ulaz = pod[:, :2]
izlaz = pod[:, 2]

K0 = ulaz[izlaz == 0, :]
K1 = ulaz[izlaz == 1, :]

plt.figure()
plt.plot(K0[:, 0], K0[:, 1], 'o')
plt.plot(K1[:, 0], K1[:, 1], 'x')
plt.show()

print(izlaz.shape)
print(K0.shape)

izlaz = np.reshape(izlaz, (izlaz.size, 1))
print(izlaz.shape)

ulaz_trening, ulaz_test, izlaz_trening, izlaz_test = train_test_split(ulaz, izlaz,
                                                                      test_size=0.2,
                                                                      shuffle=True,
                                                                      random_state=20)
print(ulaz_trening.shape)

# Metoda 1
ulazni_sloj = Input(shape=(2, )) # drugi broj je broj podataka i ne mora da se napise
dense1 = Dense(10, activation='relu')(ulazni_sloj) # prvi broj je broj cvorova
dense2 = Dense(5, activation='relu')(dense1)
izlazni_sloj = Dense(1, activation='sigmoid')(dense2)

model = Model(ulazni_sloj, izlazni_sloj)
print(model.summary())

model.compile('adam', loss='binary_crossentropy', metrics='accuracy')

history = model.fit(x=ulaz_trening, y=izlaz_trening,
          epochs=100, shuffle=True, verbose=2)

pred = model.predict(ulaz_test)
pred = np.round(pred) # pred = pred > 0.5

acc = np.sum(pred == izlaz_test)/izlaz_test.shape[0]
print(acc)
