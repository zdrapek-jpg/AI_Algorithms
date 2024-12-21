import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def znajdz_asymptoty_z_stringu(func_str):
    """
    Wylicza asymptoty pionowe i poziome funkcji wymiernej, logarytmicznej oraz wykładniczej podanej jako string.

    Parametry:
        func_str (str): Wyrażenie funkcji w postaci stringa, np. "(x**2 - 1) / (x - 2)", "log(x)" lub "2**x".

    Zwraca:
        dict: Słownik zawierający asymptoty pionowe i poziome.
    """
    x = sp.Symbol('x')
    func = sp.sympify(func_str)

    asymptoty = {"pionowe": [], "pozioma": None}

    # Obsługa funkcji logarytmicznej
    if isinstance(func, sp.log):
        pionowe = sp.solve(sp.denom(func), x)
        asymptoty["pionowe"] = pionowe

    # Obsługa funkcji wykładniczej (a^x)
    elif isinstance(func, sp.Pow):
        base = func.args[0]
        if base != 0 and base != 1:
            # Funkcje wykładnicze nie mają asymptoty poziomej
            asymptoty["pozioma"] = None

    # Obsługa funkcji wymiernej (P(x) / Q(x))
    else:
        P, Q = sp.fraction(func)
        pionowe = sp.solve(Q, x)
        pionowe = [a for a in pionowe if sp.simplify(P.subs(x, a)) != 0]
        asymptoty["pionowe"] = pionowe

        stopien_P = sp.degree(P, x)
        stopien_Q = sp.degree(Q, x)

        if stopien_P < stopien_Q:
            asymptoty["pozioma"] = 0  # y = 0
        elif stopien_P == stopien_Q:
            wspolczynnik_P = sp.LC(P, x)  # Współczynnik wiodący licznika
            wspolczynnik_Q = sp.LC(Q, x)  # Współczynnik wiodący mianownika
            asymptoty["pozioma"] = wspolczynnik_P / wspolczynnik_Q  # y = wspolczynnik_P / wspolczynnik_Q

    return asymptoty

# Przykład użycia
func_str = "(x)/(x-4)"  # Funkcja wykładnicza
asymptoty = znajdz_asymptoty_z_stringu(func_str)

print("Asymptoty pionowe:", asymptoty["pionowe"])
print("Asymptota pozioma:", asymptoty["pozioma"])

# Rysowanie wykresu funkcji
x = sp.Symbol('x')
func = sp.lambdify(x, sp.sympify(func_str), 'numpy')
x_vals = np.linspace(-16, 16, 1000)

# Omijamy wartości wokół pionowych asymptot
y_vals = func(x_vals)
for pionowa in asymptoty["pionowe"]:
    index = np.abs(x_vals - float(pionowa)) < 0.1
    y_vals[index] = np.nan  # NaN do uniknięcia rysowania wokół asymptoty

plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, label=func_str)

if asymptoty["pozioma"] is not None:
    plt.axhline(y=asymptoty["pozioma"], color='red', linestyle='--', label='Asymptota pozioma')

plt.axhline(y=0, color='black')
plt.axvline(x=0, color='black')

for pionowa in asymptoty["pionowe"]:
    plt.axvline(x=float(pionowa), color='green', linestyle='--', label=f'Asymptota pionowa x={pionowa}')

plt.xlabel('x')
plt.ylabel('y')
plt.ylim(-10, 10)  # Ustawienie zakresu dla osi y
plt.xlim(-15,15)
plt.title('Wykres funkcji z asymptotami')
plt.legend()
plt.grid()
plt.show()
