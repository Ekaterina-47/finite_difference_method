import numpy as np
import matplotlib.pyplot as plt

# Параметры:
L = 1   # Длинна стержня
N = 10  # Количество делений

# Массивы для хранения значений, включая граничные
X = np.zeros(N+2)      # узлы
T = np.zeros(N+2)      # температура

# Граничные условия:
T[0] = - 10    # Температура на одном конце
T[N+1] = 10    # Температура на другом конце

h = L/N       # шаг для внутренних узлов
h2 = h / 2    # шаг для граничных узлов


# Создание отрезков разбиений
for i in range(1, N + 2):
    if i == 1 or i == N+1:
        X[i] = X[i - 1] + h2
    else:
        X[i] = X[i - 1] + h

# Матрица коэффициентов A
A = np.zeros((N, N))

for i in range(N):
    if i > 0:
        A[i, i - 1] = -10
    A[i, i] = 20
    if i < N - 1:
        A[i, i + 1] = -10

# Вектор правых частей B
B = np.zeros(N)
B[0] = 10 * T[0]
B[-1] = 10 * T[N + 1]


# Решение системы уравнений
T_result = np.linalg.solve(A, B)

# Добавление внутренних узлов в общий массив температур
T[1:N + 1] = T_result


# Аналитическое решение:
T_an = np.zeros(N+2)      # температура для аналитического решения
X_an = np.linspace(1, L, N+2) # узлы для аналитического решения

# Граничные условия:
T_an[0] = -10
T_an[-1] = 10


# Функция для аналитического вычисления температуры
def analytical_solution(x, t1, t2, L):
    return (t2 - t1) / L * x + t1


T_an = analytical_solution(X, T[0], T[N+1], L)


# Вычисление ошибки погрешности
error = np.abs(T[1:N + 1] - T_an[1:N + 1])
max_error = np.max(error)
mean_error = np.mean(error)

print("Температура (численное решение):", T)
print("Температура (аналитическое решение):", T_an)
print("Максимальная ошибка:", max_error)
print("Средняя ошибка:", mean_error)

# Построение графика зависимости X и T
plt.plot(X, T, marker='o', linestyle='-', color='b', label='Численное решение')
plt.plot(X, T_an, linestyle='--', color='r', label='Аналитическое решение')
plt.xlabel('Узлы')
plt.ylabel('Температура')
plt.title('Распределение температуры вдоль стержня')
plt.legend()
plt.grid(True)
plt.show()