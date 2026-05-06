numbers = []

with open("results.txt", "r") as f:
    for line in f:
        line = line.strip()
        if line:  # skip empty lines
            number = int(line.split(",")[-1].strip())
            numbers.append(number)

one = numbers.count(1)
minusone = numbers.count(-1)
zero = numbers.count(0)
print(f"Ones: {one}; Minus Ones: {minusone}, Zeros: {zero}")
