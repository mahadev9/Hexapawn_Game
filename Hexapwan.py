
ADVANCE = 'advance'
CAPTURE_LEFT = 'capture_left'
CAPTURE_RIGHT = 'capture_right'


class Hexapawn:
    def __init__(self, state):
        self.board = list(state)

    def to_move(self):
        return self.board[0]

    def actions(self):
        actions = []
        if self.board[0] == 1:
            for i in range(1, 10):
                if self.board[i] == 1:
                    if i-3 >= 0 and self.board[i-3] == 0:
                        actions.append((ADVANCE, (i-4)//3, (i-1) % 3))
                    if (i-1) % 3 != 0 and i-4 >= 0 and self.board[i-4] == -1:
                        actions.append((CAPTURE_LEFT, (i-4)//3, (i-5) % 3))
                    if (i-1) % 3 != 2 and i-2 >= 0 and self.board[i-2] == -1:
                        actions.append((CAPTURE_RIGHT, (i-3)//3, (i-3) % 3))
        elif self.board[0] == -1:
            for i in range(1, 10):
                if self.board[i] == -1:
                    if i+3 <= 9 and self.board[i+3] == 0:
                        actions.append((ADVANCE, (i+2)//3, (i-1) % 3))
                    if (i-1) % 3 != 0 and i+2 <= 9 and self.board[i+2] == 1:
                        actions.append((CAPTURE_LEFT, (i+2)//3, (i+1) % 3))
                    if (i-1) % 3 != 2 and i+4 <= 9 and self.board[i+4] == 1:
                        actions.append((CAPTURE_RIGHT, (i+3)//3, (i+3) % 3))
        return actions

    def result(self, action):
        new_board = self.board.copy()
        if action[0] == ADVANCE and self.board[0] == 1:
            new_board[action[1]*3 + action[2] + 1] = 1
            new_board[action[1]*3 + action[2] + 4] = 0
        elif action[0] == ADVANCE and self.board[0] == -1:
            new_board[action[1]*3 + action[2] + 1] = -1
            new_board[action[1]*3 + action[2] - 2] = 0
        elif action[0] == CAPTURE_LEFT and self.board[0] == 1:
            new_board[action[1]*3 + action[2] + 1] = 1
            new_board[action[1]*3 + action[2] + 5] = 0
        elif action[0] == CAPTURE_LEFT and self.board[0] == -1:
            new_board[action[1]*3 + action[2] + 1] = -1
            new_board[action[1]*3 + action[2] - 1] = 0
        elif action[0] == CAPTURE_RIGHT and self.board[0] == 1:
            new_board[action[1]*3 + action[2] + 1] = 1
            new_board[action[1]*3 + action[2] + 3] = 0
        elif action[0] == CAPTURE_RIGHT and self.board[0] == -1:
            new_board[action[1]*3 + action[2] + 1] = -1
            new_board[action[1]*3 + action[2] - 3] = 0
        
        if self.board[0] == 1:
            new_board[0] = -1
        elif self.board[0] == -1:
            new_board[0] = 1
        return Hexapawn(new_board)

    def is_terminal(self):
        if self.utility():
            return True
        return False

    def utility(self):
        for i in range(7, 10):
            if self.board[i] == -1:
                return -1
        for i in range(1, 4):
            if self.board[i] == 1:
                return 1
        if not self.actions():
            return -1 * self.board[0]
        return 0
    
    def copy(self):
        return Hexapawn(self.board.copy())
