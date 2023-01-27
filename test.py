import numpy as np

def main ():
    a1 = [0, 0, 10, 10] 
    a2 = [10, 0, 0, 10] 
    a3 = [0, 10, 10, 00] 
    a4 = [10, 0, 10, 0] 

    # print(np.vstack((a1,a2)))

    arr = []
    arr = np.append(arr, np.array([[1,2,3]]), axis=0)
    arr = np.append(arr, np.array([[4,5,6]]), axis=0)
    
    # arr = np.append(arr, a3)
    print(arr)

    print('TBD!')

if __name__ == '__main__':
    main()