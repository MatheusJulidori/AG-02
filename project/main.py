import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from perceptron import Perceptron


df = pd.read_csv('tic-tac-toe.csv', delimiter=',')

p = Perceptron(0.1, 100)

df = df.replace('o', -1)
df = df.replace('b', 0)
df = df.replace('x', 1)
df = df.replace('positive', 1)
df = df.replace('negative', -1)

train, test = train_test_split(df, test_size=0.2)

x_train = train.iloc[:, :-1].values
d_train = train.iloc[:, -1].values

p.train(x_train, d_train)

x_test = test.iloc[:, :-1].values
d_test = test.iloc[:, -1].values

results = p.test(x_test)

print('Accuracy: ', accuracy_score(d_test, results))
print('Confusion Matrix: \n', confusion_matrix(d_test, results))
print('Classification Report: \n', classification_report(d_test, results))

plt.plot(np.arange(0, len(p.total_error)), p.total_error)
plt.title('Erro por época')
plt.xlabel('Época')
plt.ylabel('Erro')
plt.show()

flag = False

while not flag:
    # inserção de novos dados pelo usuário
    print('Insira os dados para classificação: ')
    print('top-left-square: ')
    top_left_square = input()
    print('top-middle-square: ')
    top_middle_square = input()
    print('top-right-square: ')
    top_right_square = input()
    print('middle-left-square: ')
    middle_left_square = input()
    print('middle-middle-square: ')
    middle_middle_square = input()
    print('middle-right-square: ')
    middle_right_square = input()
    print('bottom-left-square: ')
    bottom_left_square = input()
    print('bottom-middle-square: ')
    bottom_middle_square = input()
    print('bottom-right-square: ')
    bottom_right_square = input()

    new_data = np.array([top_left_square, top_middle_square, top_right_square, middle_left_square, middle_middle_square, middle_right_square, bottom_left_square, bottom_middle_square, bottom_right_square])
    new_data = new_data.astype(int)

    result = p.predict(new_data)
    result = p.activation(result)


    print("Vitória de X?")
    if result == 1:
        print('Sim')
    else:
        print('Não')

    print('Deseja continuar? (s/n)')
    flag = True if input() == 'n' else False
