# GUI for sudoku
# press space for visualization of auto-solving

import pygame
import time
from sudoku import Sudoku
pygame.font.init()


class SudokuBoard:
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

    # this produces a random board with difficulty ranging from 0 to 1 (easy to hard)
    board = Sudoku(3).difficulty(0.7).board
    for i in range(len(board)):
        board[i] = [0 if x is None else x for x in board[i]]

    def __init__(self, rows, cols, width, height, win):
        self.rows = rows
        self.cols = cols
        self.tiles = [[Tiles(self.board[i][j], i, j, width, height) for j in range(cols)] for i in range(rows)]
        self.width = width
        self.height = height
        self.model = None
        self.update_model()
        self.selected = None
        self.win = win

    def update_model(self):
        # refreshes the model every time a value gets updated
        self.model = [[self.tiles[i][j].value for j in range(self.cols)] for i in range(self.rows)]

    def click(self, pos):
        # returns the row and column location of a click
        if pos[0] < self.width and pos[1] < self.height:
            gap = self.width / 9
            x = pos[0] // gap
            y = pos[1] // gap
            return (int(y),int(x))
        else:
            return None

    def clear(self):
        row, col = self.selected
        if self.tiles[row][col].value == 0:
            self.tiles[row][col].set_temp(0)

    def solve(self):  # recursive function that uses backtracking to find the correct soln if it exists
        empty = find_empty(self.model)
        if not empty:  # if there are no empty locations, board is solved
            return True
        else:
            row, col = empty

        for i in range(1, 10):  # loops through the 9 possible values for a position
            if check_valid(self.model, i, (row, col)):
                self.model[row][col] = i  # changes value at empty to position to i

                # recursively calls itself to check if rest of board is possible to be valid with board[row][col] = i
                if self.solve():
                    return True

                self.model[row][col] = 0  # if board[row][col] was not valid, reset value to 0 and try the values after i

        return False

    def solve_gui(self):
        self.update_model()
        find = find_empty(self.model)
        if not find:
            return True
        else:
            row, col = find

        for i in range(1, 10):
            if check_valid(self.model, i, (row, col)):
                self.model[row][col] = i
                self.tiles[row][col].set(i)
                self.tiles[row][col].draw_change(self.win, True)
                self.update_model()
                pygame.display.update()
                pygame.time.delay(35)

                if self.solve_gui():
                    return True

                self.model[row][col] = 0
                self.tiles[row][col].set(0)
                self.update_model()
                self.tiles[row][col].draw_change(self.win, False)
                pygame.display.update()
                pygame.time.delay(100)

        return False

    def is_finished(self):
        # checks if all tiles are filled to end the game
        for i in range(self.rows):
            for j in range(self.cols):
                if self.tiles[i][j].value == 0:
                    return False
        return True

    def select(self, row, col):
        # First makes sure all other tiles are unselected, and then selects the clicked location
        for i in range(self.rows):
            for j in range(self.cols):
                self.tiles[i][j].selected = False

        self.tiles[row][col].selected = True
        self.selected = (row, col)

    def sketch(self, val):
        # allows a preview in the selected tile
        row, col = self.selected
        self.tiles[row][col].set_temp(val)

    def place(self, val, board):
        # places an answer value in the selected tile
        row, col = self.selected
        if self.tiles[row][col].value == 0:
            self.tiles[row][col].set(val)
            self.update_model()

            if check_valid(self.model, val, (row,col)) and self.solve():
                return True
            else:
                self.tiles[row][col].set(0)
                self.tiles[row][col].set_temp(0)
                self.update_model()
                return False

    def draw(self):
        # Drawing the tile lines
        gap = self.width / 9
        for i in range(self.rows+1):
            # making the
            if i % 3 == 0 and i != 0:
                thick = 4
            else:
                thick = 1
            pygame.draw.line(self.win, (0,0,0), (0, i*gap), (self.width, i*gap), thick)
            pygame.draw.line(self.win, (0, 0, 0), (i * gap, 0), (i * gap, self.height), thick)

        # Drawing the tile values
        for i in range(self.rows):
            for j in range(self.cols):
                self.tiles[i][j].draw(self.win)

    # TODO: add a function that resets the board to a new one


class Tiles:
    rows = 9
    cols = 9

    def __init__(self, value, row, col, width, height):
        self.value = value
        self.temp = 0
        self.row = row
        self.col = col
        self.width = width
        self.height = height
        self.selected = False

    def draw(self, win):
        # TODO: change font type later
        fnt = pygame.font.SysFont("comicsans", 40)
        fnt_temp = pygame.font.SysFont("comicsans", 18)

        gap = self.width / 9
        x = self.col * gap
        y = self.row * gap

        # setting the number value in the chosen tile
        if self.temp != 0 and self.value == 0:
            # TODO: add ability to have more than one temp value
            text = fnt_temp.render(str(self.temp), 1, (128,128,128))
            win.blit(text, (x+5, y+5))
        elif not(self.value == 0):
            text = fnt.render(str(self.value), 1, (0, 0, 0))
            win.blit(text, (x + (gap/2 - text.get_width()/2), y + (gap/2 - text.get_height()/2)))

        if self.selected:
            pygame.draw.rect(win, (255,0,0), (x,y, gap ,gap), 3)

    def draw_change(self, win, g=True):
        fnt = pygame.font.SysFont("comicsans", 40)

        gap = self.width / 9
        x = self.col * gap
        y = self.row * gap

        pygame.draw.rect(win, (255, 255, 255), (x, y, gap, gap), 0)

        text = fnt.render(str(self.value), 1, (0, 0, 0))
        win.blit(text, (x + (gap / 2 - text.get_width() / 2), y + (gap / 2 - text.get_height() / 2)))
        if g:
            pygame.draw.rect(win, (0, 255, 0), (x, y, gap, gap), 3)
        else:
            pygame.draw.rect(win, (255, 0, 0), (x, y, gap, gap), 3)

    def set(self, val):
        self.value = val

    def set_temp(self, val):
        self.temp = val


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


def find_empty(board):
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 0:
                return i, j  # row and col of empty location
    return None


def format_time(secs):
    sec = secs % 60
    minutes = secs//60
    formatted = f'{minutes}: {sec}'
    return formatted


def redraw_window(win, board, time, strikes):
    win.fill((255,255,255))
    # Draw time
    fnt = pygame.font.SysFont("comicsans", 40)
    text = fnt.render("Time: " + format_time(time), 1, (0,0,0))
    win.blit(text, (325, 540))
    # Draw Strikes
    text = fnt.render("X " * strikes, 1, (255, 0, 0))
    win.blit(text, (20, 540))
    # Draw grid and board
    board.draw()


def main():
    win = pygame.display.set_mode((540,600))
    pygame.display.set_caption("Sudoku")
    board = SudokuBoard(9, 9, 540, 540, win)
    key = None
    run = True
    start = time.time()
    strikes = 0
    while run:

        play_time = round(time.time() - start)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    key = 1
                if event.key == pygame.K_2:
                    key = 2
                if event.key == pygame.K_3:
                    key = 3
                if event.key == pygame.K_4:
                    key = 4
                if event.key == pygame.K_5:
                    key = 5
                if event.key == pygame.K_6:
                    key = 6
                if event.key == pygame.K_7:
                    key = 7
                if event.key == pygame.K_8:
                    key = 8
                if event.key == pygame.K_9:
                    key = 9
                if event.key == pygame.K_KP1:
                    key = 1
                if event.key == pygame.K_KP2:
                    key = 2
                if event.key == pygame.K_KP3:
                    key = 3
                if event.key == pygame.K_KP4:
                    key = 4
                if event.key == pygame.K_KP5:
                    key = 5
                if event.key == pygame.K_KP6:
                    key = 6
                if event.key == pygame.K_KP7:
                    key = 7
                if event.key == pygame.K_KP8:
                    key = 8
                if event.key == pygame.K_KP9:
                    key = 9
                if event.key == pygame.K_DELETE:
                    board.clear()
                    key = None

                if event.key == pygame.K_SPACE:
                    board.solve_gui()

                if event.key == pygame.K_RETURN:
                    i, j = board.selected
                    if board.tiles[i][j].temp != 0:
                        if board.place(board.tiles[i][j].temp, board):
                            print("Success")
                        else:
                            print("Wrong")
                            strikes += 1
                        key = None

                        if board.is_finished():
                            print("Game over")

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                clicked = board.click(pos)
                if clicked:
                    board.select(clicked[0], clicked[1])
                    key = None

        if board.selected and key != None:
            board.sketch(key)

        redraw_window(win, board, play_time, strikes)
        pygame.display.update()


main()
pygame.quit()
