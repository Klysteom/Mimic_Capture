import itertools
import time
import os
import cv2
from PIL import Image
import numpy as np

COLS = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
REV_COL = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G'}


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

    def update_board(self, points):
        for point in points:
            _, _, i, j, is_block = point
            if is_block is True:
                self.matrix[i][j] = True

    def remove_pointless_blocks(self):
        borders = get_borders(self)
        for border in borders:
            break_flag = False
            for i, row in enumerate(self.matrix):
                for j in range(len(row)):
                    if [i, j] not in borders and check_move(self, border[0], border[1], i, j):
                        break_flag = True
                        break
                if break_flag:
                    break
            if break_flag is False:
                self.matrix[border[0]][border[1]] = False

    def remove_unreachable_blocks(self):
        for i, row in enumerate(self.matrix):
            for j, value in enumerate(row):
                if value is True and [i, j] != self.frog:
                    if self.calculate_best_move(i, j, to_frog=True)[0] is None:
                        self.matrix[i][j] = False

    def move(self):
        border_i, border_j = self.calculate_best_move(self.frog[0], self.frog[1])
        if border_i is None or border_j is None:
            # The player won the game
            return True
        i, j = self.calculate_best_move(border_i, border_j, to_frog=True)
        self.frog = [i, j]
        if i == border_i and j == border_j:
            # The player lost the game
            return False
        return None

    def calculate_best_move(self, start_i, start_j, to_frog=False):
        borders = get_borders(self)
        # calculate the shortest path from start to destination
        visited = []
        queue = [[start_i, start_j]]
        while queue:
            i, j = queue[0]
            visited.append(queue.pop(0))
            for offset_i, offset_j in [[1, 0], [1, -1], [0, -1], [1, 1], [-1, 0], [-1, -1], [0, 1], [-1, 1]]:
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
                    if [dest_i, dest_j] not in borders:
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


def play_game(points):
    board = Board()
    board.update_board(points)
    player_moves = ''
    while True:
        clear_and_show_board(board)
        time.sleep(1)
        player_move = input('Select a block to remove: ')
        remove_i, remove_j = convert_indexes(player_move)
        clear_and_show_board(board)
        if (remove_i is None or remove_j is None or board.matrix[remove_i][remove_j] is False
                or [remove_i, remove_j] == board.frog):
            print('Invalid block indexes.\nPlease try again.\n')
            time.sleep(1)
            continue
        player_moves += player_move.upper() + ' -> '
        board.matrix[remove_i][remove_j] = False
        clear_and_show_board(board)
        time.sleep(1)
        is_win = board.move()
        if is_win in [True, False]:
            clear_and_show_board(board)
            if is_win is True:
                print('You win!')
            else:
                print('You lose!')
            print(f'Your moves: {player_moves[:-4]}')
            exit()


def get_borders(board):
    borders = []
    for i, row in enumerate(board.matrix):
        for j, value in enumerate(row):
            if value is True and (i in [0, 6] or j in [0, 6]):
                borders.append([i, j])
    return borders


def solve(points, screenshot_path):
    maximum_benefit = 0
    benefit_list = []
    blocks_to_remove = []
    available_blocks = []
    board = Board()
    board.update_board(points)
    board.remove_pointless_blocks()
    board.remove_unreachable_blocks()
    borders = get_borders(board)

    for i in range(7):
        for j in range(7):
            if board.matrix[i][j] is True and [i, j] != board.frog and [i, j] not in borders:
                available_blocks.append([i, j])

    # combination of one block from available blocks
    for i in range(1, 11):  # add all combinations of true blocks that not in the borders and not the frog block
        blocks_to_remove += list(itertools.combinations(available_blocks, i))
    for block_to_remove in blocks_to_remove:
        board = Board()
        board.update_board(points)
        for block in block_to_remove:
            i, j = block
            board.matrix[i][j] = False
        board.remove_pointless_blocks()
        board.remove_unreachable_blocks()
        borders = get_borders(board)
        amount_of_blocks = len(borders) + len(block_to_remove)
        if amount_of_blocks <= 10:
            true_blocks = len(block_to_remove)
            for i in range(7):
                for j in range(7):
                    if board.matrix[i][j] is True:
                        true_blocks += 1
            benefit = true_blocks - amount_of_blocks
            if benefit >= maximum_benefit:
                maximum_benefit = benefit
                blocks_to_remove_with_maximum_benefit = list(block_to_remove) + borders
                benefit_list.append([benefit, blocks_to_remove_with_maximum_benefit])

    counter = 1
    if len(benefit_list) != 0:
        print(f'The maximum benefit is {maximum_benefit}.\n')
        for item in benefit_list:
            if item[0] == maximum_benefit:
                print(f'Blocks to remove: {' '.join([REV_COL[i[1]]+str(i[0]+1) for i in item[1]])}')
                save_solution_as_image(counter, points, screenshot_path, item[1])
                counter += 1


def save_solution_as_image(solution_number, points, screenshot_path, blocks_to_remove):
    img_rgb = cv2.imread(screenshot_path)

    for point in points:
        pixel_i, pixel_j, matrix_i, matrix_j, is_block = point
        if [matrix_i, matrix_j] in blocks_to_remove:
            cv2.circle(img_rgb, (pixel_j, pixel_i), 10, (255, 0, 0), -1)
    cv2.imwrite(f'solution-{solution_number}.png', img_rgb)


def main():
    screenshot_path = '1.png'
    points = get_blocks_from_image(screenshot_path)
    play_game(points)
    # solve(points, screenshot_path)


def convert_pic(screenshot_path):
    im = Image.open(screenshot_path).convert('RGB')
    if screenshot_path.endswith('.jpg'):
        new_path = screenshot_path.replace('.jpg', '.png')
        im.save(new_path, "png")
        return new_path
    if screenshot_path.endswith('.webp'):
        new_path = screenshot_path.replace('.webp', '.png')
        im.save(new_path, "png")
        return new_path
    return screenshot_path


def get_blocks_from_image(screenshot_path):
    mimic_path = 'mimic_treasure.png'
    screenshot_path = convert_pic(screenshot_path)
    img_rgb = cv2.imread(screenshot_path)
    screenshot_h, screenshot_w = img_rgb.shape[:-1]

    # Finds the mimic treasure in the picture
    template = cv2.imread(mimic_path)
    template_w, template_h = template.shape[:-1]
    res = cv2.matchTemplate(img_rgb, template, cv2.TM_CCOEFF_NORMED)
    threshold = .5
    loc = np.where(res >= threshold)
    pts = list(zip(*loc[::-1]))
    if len(pts) > 0:
        pt = pts[0]
        mimic_center = int(pt[1] + template_h/2 + 12), int(screenshot_w/2)
    else:
        print("Error: Can't find Mimic Treasure.")
        mimic_center = int(screenshot_h * 0.61924), int(screenshot_w/2)
    mimic_center_i, mimic_center_j = mimic_center
    # Places points in the center of the blocks relative to the mimic treasure
    vertical_offset = screenshot_w * 0.0545
    horizontal_offset = screenshot_w * 0.113
    offsets = []
    for i in range(-7, 7):
        if i % 2 != 0:
            for j in [-3, -1, 1, 3]:
                offsets.append((i, j))
        else:
            for j in [-2, 0, 2]:
                offsets.append((i, j))
    # remove mimic treasure point because it's dark like non-block
    offsets.remove((0, 0))
    # finds non-blocks by color
    points = []
    for offset in offsets:
        i, j = offset
        square = 10  # the size from the center of the sample
        point_i, point_j = (int(mimic_center_i + i*vertical_offset), int(mimic_center_j + j*horizontal_offset))
        cropped_image = img_rgb[point_i-square:point_i+square, point_j-square:point_j+square]  # take block sample
        # calculates dominant color of sample
        pixels = np.float32(cropped_image.reshape(-1, 3))
        n_colors = 5
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)
        dominant = palette[np.argmax(counts)]
        point_matrix_i, point_matrix_j = (int((i+7)/2), j+3)
        # point by pixels, point by matrix index, is there a block at the point
        points.append((point_i, point_j, point_matrix_i, point_matrix_j, bool(dominant[2] > 140)))
    points.append((mimic_center_i, mimic_center_j, 3, 3, True))  # adds mimic treasure block
    return points


if __name__ == '__main__':
    main()
# priorities: down/down-right -> down-left -> up/down-right -> up/up-left -> up-right
