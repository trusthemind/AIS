import numpy as np
import matplotlib.pyplot as plt

x = np.array([16, 27, 38, 19, 100,72])
y = np.array([12, 35,39,41,60,55])

degree = 2
coefficients = np.polyfit(x, y, degree)  # Знаходження коефіцієнтів полінома
polynomial = np.poly1d(coefficients)    # Створення поліноміальної функції

x_fit = np.linspace(min(x), max(x), 100)
y_fit = polynomial(x_fit)

print("Коефіцієнти полінома:", coefficients)
print("Поліном: ", polynomial)

plt.scatter(x, y, color='red', label='Експериментальні точки')  # Точки даних
plt.plot(x_fit, y_fit, label=f"Апроксимація поліномом ступеня {degree}", color='blue')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Метод найменших квадратів - Поліноміальна апроксимація")
plt.legend()
plt.grid()
plt.show()
