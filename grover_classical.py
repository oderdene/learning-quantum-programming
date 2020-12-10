n        = 4
elements = [i for i in range(2**n)]

def oracle(x, search_element):
    if x==search_element:
        return 1
    return 0

def solve(search_element):
    for element in elements:
        if oracle(element, search_element):
            return element
    return None

if __name__ == "__main__":
    print("Элементүүд")
    print(elements)

    search_element = 3
    print("Хайх утга", search_element)
    result = solve(search_element=search_element)
    if result:
        print(result, "элемент олдлоо")
    else:
        print("олдсонгүй")

    search_element = 40
    print("Хайх утга", search_element)
    result = solve(search_element=search_element)
    if result:
        print(result, "элемент олдлоо")
    else:
        print("олдсонгүй")


