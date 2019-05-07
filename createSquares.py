import pickle
from lines import gen_square

def createSquares(dimen):

    square_sizes = [8]

    matrices = []
    answers = []

    if(dimen < 3):
        print("size is too small")

    with open("squares.txt", "wb+") as f:
        #for all matrix sizes
            #for all positions
        cumul = 0
        for sq_i, sq_size in enumerate(square_sizes):
            track = 0
            for x in range(dimen - sq_size + 1):
                for y in range(dimen - sq_size + 1):
                    matrix = gen_square([x, y], sq_size, dimen)
                    ans = cumul + track
                    answer = [ans, dimen * x + y, sq_i]
                    answers.append(answer)
                    matrices.append(matrix)
                    track += 1
            cumul += (dimen - sq_size + 1) ** 2

        pickle.dump([matrices, answers], f)
                    



if __name__ == '__main__':
    size = 10

    createSquares(size)