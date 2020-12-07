n        = 4
elements = [i for i in range(2**n)]

def oracle(x, search_element):
    if x==search_element:
        return 1
    return 0


if __name__ == "__main__":
    print("Элементүүд")
    print(elements)
    search_element = 3
    print("Хайх утга", search_element)
    for element in elements:
        print(element, oracle(element, search_element))


