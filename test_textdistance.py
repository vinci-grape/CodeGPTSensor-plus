import textdistance

# 测试下textdistance计算是否正确

code1 = """def add(a, b):
    return a + b
"""

code2 = """def add_numbers(a, c):
    return a + c
"""

distance = textdistance.levenshtein.distance(code1, code2)
print("编辑距离:", distance)
print("编辑距离(归一化):", distance / max(len(code1), len(code2)))
