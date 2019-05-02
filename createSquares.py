def createSquares(size):

    if(size < 3):
        print("size is too small")

    with open("squares.txt", "w+") as f:
        #for all matrix sizes
        for matrixSize in range(3, size+1):
            #for all positions
            for x in range(0, size-matrixSize+1):
                for y in range(0, size-matrixSize+1):
                    #initalize matrix to 0
                    matrix = [[0 for x in range(size)] for y in range(size)] 

                    #make square
                    for j in range(x,x+matrixSize):
                        matrix[j][y] = 1

                    for j in range(x,x+matrixSize):
                        matrix[j][y+matrixSize-1] = 1     

                    for j in range(y,y+matrixSize):
                        matrix[x][j] = 1

                    for j in range(y,y+matrixSize):
                        matrix[x+matrixSize-1][j] = 1

                    # print("x/y: ",x,y)
                    
                    f.write("\n#########\n")
                    for row in matrix:
                        f.write(str(row) + "\n")

                    



if __name__ == '__main__':
    size = 5

    createSquares(size)