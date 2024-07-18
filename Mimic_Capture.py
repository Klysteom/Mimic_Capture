import random
import time
import os

COLS = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}


def convert_indexes(block):
    block = block.strip()
    if len(block) != 2 or block[0].upper() not in COLS or block[1].isdigit() is False or not (0 < int(block[1]) < 8):
        return None, None
    return int(block[1]) - 1, COLS[block[0].upper()]


def check_move(board, from_i, from_j, to_i, to_j):
    for index in [from_i, from_j, to_i, to_j]:
        if index < 0 or index > 6:
            return False
    if board.matrix[to_i][to_j] is False or board.matrix[from_i][from_j] is False:
        return False
    if from_j - 1 <= to_j <= from_j + 1:
        if from_j % 2 == 0:
            if from_i - 1 <= to_i < from_i + 1 or (to_i == from_i + 1 and to_j == from_j):
                return True
        else:
            if from_i - 1 < to_i <= from_i + 1 or (to_i == from_i - 1 and to_j == from_j):
                return True
    return False


class Board:
    def __init__(self, frog='d4'):
        self.matrix = [[False for _ in range(7)] for _ in range(7)]
        self.frog = list(convert_indexes(frog))
        if None in self.frog:
            print('The frog is in an undefined block.')
            exit()

    def update_board(self, blocks):
        while True:
            blocks = blocks.replace('\n', ' ').replace('\t', ' ').replace('  ', ' ').strip()
            if '  ' not in blocks:
                break
        blocks_list = blocks.split(' ')
        for block in blocks_list:
            block_i, block_j = convert_indexes(block)
            if None in [block_i, block_j]:
                print(f'The board cannot be loaded.\nBlock {block} is not defined.')
                exit()
            self.matrix[block_i][block_j] = True

    def move(self):
        border_i, border_j = self.calculate_best_move(self.frog[0], self.frog[1])
        if border_i is None or border_j is None:
            clear_and_show_board(self)
            print('The frog lost the game.')
            exit()
        i, j = self.calculate_best_move(border_i, border_j, to_frog=True)
        self.frog = [i, j]
        if i == border_i and j == border_j:
            clear_and_show_board(self)
            print('The frog wins.')
            exit()

    def calculate_best_move(self, start_i, start_j, to_frog=False):
        borders = []
        if to_frog is False:
            for i, row in enumerate(self.matrix):
                for j, value in enumerate(row):
                    if value is True and (i in [0, 6] or j in [0, 6]):
                        borders.append([i, j])
        # calculate the shortest path from start to destination
        visited = []
        queue = [[start_i, start_j]]
        while queue:
            i, j = queue[0]
            visited.append(queue.pop(0))
            for offset_i in range(-1, 2):
                for offset_j in range(-1, 2):
                    dest_i = i + offset_i
                    dest_j = j + offset_j
                    if ([dest_i, dest_j] not in visited
                            and [dest_i, dest_j] not in queue
                            and check_move(self, i, j, dest_i, dest_j)):
                        if to_frog:
                            if [dest_i, dest_j] == self.frog:
                                return i, j
                        else:
                            if [dest_i, dest_j] in borders:
                                return dest_i, dest_j
                        queue.append([dest_i, dest_j])
        return None, None

    def show_board(self):
        for i, row in enumerate(self.matrix):
            for t in range(2):
                for j in range(len(row)):
                    if j % 2 == t and row[j]:
                        if i == self.frog[0] and j == self.frog[1]:
                            print('🐸', end='\t')
                        else:
                            print('⬣', end='\t')
                    else:
                        print(' ', end='\t')
                print()


def clear_and_show_board(board):
    os.system('cls' if os.name == 'nt' else 'clear')
    board.show_board()


def play_game(board):
    while True:
        clear_and_show_board(board)
        time.sleep(1)
        remove_i, remove_j = convert_indexes(input('Select a block to remove: '))
        clear_and_show_board(board)
        if (remove_i is None or remove_j is None or board.matrix[remove_i][remove_j] is False
                or [remove_i, remove_j] == board.frog):
            print('Invalid block indexes.\nPlease try again.\n')
            time.sleep(1)
            continue
        board.matrix[remove_i][remove_j] = False
        clear_and_show_board(board)
        time.sleep(1)
        board.move()


def main():
    board = Board()
    string_board = """
                    a1    c1 d1    f1 g1
                    a2    c2    e2
                    a3 b3 c3 d3 e3 f3 g3
                    a4 b4 c4 d4    f4
                       b5 c5    e5 f5 g5
                    a6 b6 c6 d6 e6    g6
                       b7 c7 d7    f7
                    """
    board.update_board(string_board)
    play_game(board)


if __name__ == '__main__':
    main()
