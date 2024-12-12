def logic_and(x1, x2):
    """
    Реалізація логічної функції AND.
    Повертає 1, якщо x1 і x2 дорівнюють 1, інакше 0.
    """
    return int(x1 == 1 and x2 == 1)

def logic_or(x1, x2):
    """
    Реалізація логічної функції OR.
    Повертає 1, якщо хоча б одне з x1 або x2 дорівнює 1, інакше 0.
    """
    return int(x1 == 1 or x2 == 1)

def logic_xor(x1, x2):
    """
    Реалізація логічної функції XOR через функції OR та AND.
    XOR працює за формулою: (x1 OR x2) AND NOT (x1 AND x2).
    """
    or_result = logic_or(x1, x2)
    and_result = logic_and(x1, x2)
    return logic_and(or_result, not and_result)

def test_logic_functions():
    """
    Тестування функцій AND, OR, XOR на всіх можливих комбінаціях.
    """
    inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    print("x1 | x2 | AND | OR  | XOR")
    print("---|----|-----|-----|-----")
    for x1, x2 in inputs:
        print(f" {x1} |  {x2} |  {logic_and(x1, x2)}  |  {logic_or(x1, x2)}  |  {logic_xor(x1, x2)}")

if __name__ == "__main__":
    test_logic_functions()
