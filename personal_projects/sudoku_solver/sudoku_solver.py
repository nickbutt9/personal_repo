# logic is to use the 'backtrack' algorithm to solve any solvable sudoku puzzle
# steps: 1. find the first (rows then cols) unsolved space (denoted by 0 in the board)
# 2. try valid numbers [1-9] 3. pick first one that works (checking against row, col, and square).
# 4. repeat. 5. If location has no possible valid solution, backtrack

# backtrack in this case means to go to the previously solved location and go to the next valid number and continue.
# if the previous location has no valid number, go to one before that. Keep repeating until continuing results in
# a correctly solved board

# # easy board from sudoku app
# board = [
#     [0, 4, 8, 2, 0, 0, 0, 0, 1],
#     [1, 0, 0, 3, 8, 4, 7, 2, 6],
#     [3, 0, 0, 7, 0, 1, 9, 4, 8],
#     [0, 7, 2, 6, 4, 5, 1, 8, 0],
#     [8, 0, 0, 0, 0, 2, 4, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 7],
#     [0, 8, 4, 0, 0, 0, 3, 0, 0],
#     [6, 0, 0, 4, 1, 0, 0, 0, 2],
#     [0, 0, 3, 0, 0, 0, 0, 7, 4]
# ]

# # nightmare board from sudoku app
# board = [
#     [4, 0, 0, 0, 0, 9, 0, 3, 0],
#     [0, 0, 0, 6, 0, 0, 0, 2, 9],
#     [0, 0, 1, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 7, 0, 0, 2, 0, 0],
#     [0, 6, 0, 5, 8, 3, 0, 7, 0],
#     [0, 0, 0, 0, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 8, 0, 0],
#     [9, 3, 0, 0, 0, 6, 0, 0, 0],
#     [0, 5, 0, 2, 0, 0, 0, 0, 6]
# ]

# importing and formatting randomized boards from sudoku package
from sudoku import Sudoku
sudoku_board = Sudoku(3).difficulty(0.8).board
for z in range(len(sudoku_board)):
    sudoku_board[z] = [0 if x is None else x for x in sudoku_board[z]]


def solve(board):  # recursive function that uses backtracking to find the correct soln if it exists

    empty = find_empty(board)
    if not empty:  # if there are no empty locations, board is solved
        return True
    else:
        row, col = empty

    for i in range(1, 10):  # loops through the 9 possible values for a position
        if check_valid(board, i, (row, col)):
            board[row][col] = i  # changes value at empty to position to i

            # recursively calls itself to check if rest of board is possible to be valid with board[row][col] = i
            if solve(board):
                return True

            board[row][col] = 0  # if board[row][col] was not valid, reset value to 0 and try the values after i


def check_valid(board, value, position):  # value and position of inserted number
    # checking if value is valid in its row
    for i in range(len(board[0])):
        if board[position[0]][i] == value and position[1] != i:
            return False

    # checking if value is valid in its column
    for i in range(len(board)):
        if board[i][position[1]] == value and position[0] != i:
            return False

    # checking if value is valid in its 3x3 box
    box_x = position[1] // 3
    box_y = position[0] // 3

    for i in range(box_y*3, box_y*3 + 3):  # (box_y*3, box_x*3) puts you in the upper left corner of your desired box
        for j in range(box_x*3, box_x*3 + 3):
            if board[i][j] == value and (i, j) != position:
                return False

    # passed all checks so valid
    return True


def print_board(board):
    for i in range(len(board)):
        if i % 3 == 0 and i != 0:
            print('- - - - - - - - - - - -')
        for j in range(len(board[0])):
            if j % 3 == 0 and j != 0:
                print(' | ', end='')

            if j < 8:
                print(f'{board[i][j]} ', end='')
            else:
                print(board[i][j])


def find_empty(board):
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 0:
                return i, j  # row and col of empty location

    return None


def run_solver(board):
    print('The input board is:')
    print_board(board)

    solve(board)
    print('\nThe solved board is:')
    print_board(board)


# run_solver(sudoku_board)
