import time
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from perceptron import Perceptron

# TREINAMENTO
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

start_time = time.time()
p.train(x_train, d_train)
end_time = time.time()
training_time = (end_time - start_time) * 1000
training_time = "{:.2f}".format(training_time)

# TESTE
x_test = test.iloc[:, :-1].values
d_test = test.iloc[:, -1].values

results = p.test(x_test)

accuracy = accuracy_score(d_test, results)
confusion_matrix_results = confusion_matrix(d_test, results)
print('Classification Report: \n', classification_report(d_test, results))
print('Tempo de treinamento: ', training_time, 'ms')

# erros por epoca
plt.plot(np.arange(0, len(p.total_error)), p.total_error)
plt.title('Erro por época')
plt.xlabel('Época')
plt.ylabel('Erro')
plt.show()

# Gráfico de acuracia
labels = 'Acertos', 'Erros'
sizes = [accuracy, 1 - accuracy]
explode = (0, 0.1)
title = 'Acurácia do modelo'
fig1, ax1 = plt.subplots()
ax1.set_title(title)
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.2f%%', shadow=True, startangle=90)
ax1.axis('equal')
plt.show()

# Grafico de matriz de confusão
true_negatives, false_positives, false_negatives, true_positives = confusion_matrix_results.ravel()
labels = 'True Negatives', 'False Positives', 'False Negatives', 'True Positives'
fig2, ax2 = plt.subplots()
bar_width = 0.35
indices = np.arange(4)
bar1 = ax2.bar(indices[0], true_negatives, bar_width, label='True Negatives')
bar2 = ax2.bar(indices[1], false_positives, bar_width, label='False Positives')
bar3 = ax2.bar(indices[2], false_negatives, bar_width, label='False Negatives')
bar4 = ax2.bar(indices[3], true_positives, bar_width, label='True Positives')
ax2.set_xlabel('Classificação')
ax2.set_ylabel('Número de ocorrências')
ax2.set_title('Matriz de confusão')
ax2.set_xticks(indices)
ax2.set_xticklabels(['True Negatives', 'False Positives', 'False Negatives', 'True Positives'])


def add_values(rects):
    for rect in rects:
        height = rect.get_height()
        ax2.annotate('{}'.format(height),
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')


add_values(bar1)
add_values(bar2)
add_values(bar3)
add_values(bar4)
ax2.legend()
plt.show()

flag = False

# INSERÇÃO DE DADOS PARA CLASSIFICAÇÃO
while not flag:
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

    new_data = np.array([top_left_square, top_middle_square, top_right_square, middle_left_square, middle_middle_square,
                         middle_right_square, bottom_left_square, bottom_middle_square, bottom_right_square])
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
