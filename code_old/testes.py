import numpy as np
import ast


def main():
    #chaveSujeitoTeste = np.random.randint(0, 2)
    #print(chaveSujeitoTeste)


    string = '[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]'
    print(string)
    lst = ast.literal_eval(string)
    print(lst[2])







if __name__ == '__main__':
	main()