import sys, os
import argparse
import copy  
import glob
import math

import base64
import bcrypt 
import curses 
from enum import IntEnum
import hashlib
from queue import Queue

from cipher import AESCipher
from constants import KEY_BINDINGS_BASE, KEY_BINDINGS, ALL_KEYS, JellyState, Direction, Justification, Orientation, Square, ENABLE_ANIMATION
from renderer import Renderer
from util import log, get_window_size, out_of_bounds, Position, Grid, get_bit

class Jelly(object):
    def __init__(self, positions):
        self.positions = positions

class Pane(object):
    def __init__(self, position, orientation, rows, cols):
        self.position = position 
        self.orientation = orientation 
        self.direction_gravity = Direction(self.orientation).opposite()
        self.rows = rows 
        self.cols = cols 
        self.jelly_state = None # Board coords, not frame coords!

    def to_coord(self, position): 
        if self.orientation % 2 == 1:
            x = position.x
            y = position.y
        else:
            x = position.y 
            y = position.x
        if self.orientation >= 2 and self.orientation <= 3:
            y = self.cols - 1 - y 
        if self.orientation > 2:
            x = self.rows - 1 - x 
        return Position(x,y)

    def from_frame(self, position): 
        x, y = position.x - self.position.x, position.y - self.position.y
        if self.orientation >= 2 and self.orientation <= 3:
            y = self.cols - 1 - y 
        if self.orientation > 2:
            x = self.rows - 1 - x 
        if self.orientation % 2 == 0:
            x, y = y, x
        return Position(x,y)

    def is_floating(self, jelly):
        return self.jelly_state[jelly.positions[0]] == JellyState.FLOATING

class JelliesGame(object):
    def __init__(self, stdscr, replay_file):
        self.frame_rows = None
        self.frame_cols = None
        self.board      = None
        self.component  = None
        self.stdscr = stdscr
        self.current_level = 0
        self.level_subdir = ''
        self.exited = False
        self.renderer = Renderer(stdscr)
        self.selected_jelly = -1
        self.selected_pane = -1
        self.future_selected_jelly = None
        self.num_levels = len(glob.glob(os.path.join('levels',self.level_subdir,'*[0-9].txt')))
        self.solved_levels = [False for i in range(self.num_levels)]
        self.replay_string = ''
        self.replay_idx = 0
        self.should_animate = True
        self.enable_animation = ENABLE_ANIMATION
        self.should_log_key_event = True
        self.replay_file = 'replay.log'
        if replay_file:
            self.replay_file = replay_file
            self.should_animate = False
            self.should_log_key_event = False
            with open(replay_file) as f:
                self.replay_string = ''.join(f.readlines()).replace('\n','')
        elif os.path.isfile('replay.log'):
            os.remove('replay.log')
        self.queued_key_events = ''
        self.show_welcome_message = True
        self.welcome_message = """
Hi! Welcome to jellies: electric bugaloo. All the rules are unchanged. Press (%s) or (%s) to select jellies and arrow keys to move them. 
Other commands: (%s): undo, (%s): redo, (%s): restart level, (%s): go to previous level, (%s): go to next level, (%s): toggle animation, and (%s): quit game. 
All keybindings are listed in `constants.py`. Clicking on a jelly will move it in the direction of the side it was clicked on, unless it is floating. 
Press (%s) to toggle this message.
""" % (
        KEY_BINDINGS_BASE['select_next'][0], 
        KEY_BINDINGS_BASE['select_prev'][0], 
        KEY_BINDINGS_BASE['undo'][0], 
        KEY_BINDINGS_BASE['redo'][0], 
        KEY_BINDINGS_BASE['level_restart'][0], 
        KEY_BINDINGS_BASE['level_prev'][0], 
        KEY_BINDINGS_BASE['level_next'][0], 
        KEY_BINDINGS_BASE['toggle_animation'][0],
        KEY_BINDINGS_BASE['quit'][0], 
        KEY_BINDINGS_BASE['toggle_welcome_message'][0])
        def render_messages():
            self.renderer.print('Level %d of %d' % (self.current_level+1, self.num_levels),[0,1],justification=Justification.LEFT,pause=False)
            if self.show_welcome_message:
                self.renderer.print(self.welcome_message.replace('\n',''),[0,30],True,pause=False,pad=True)
        self.render_messages = render_messages

    def load_from_file(self, filename):
        self.board_temp = []
        with open(filename) as f:
            orientations = list(map(Orientation, map(int, f.readline().split())))
            for row, line in enumerate(f.readlines()):
                line = line.rstrip()
                if len(self.board_temp) == 0:
                    self.board_temp += [['X'] * (2 + len(line))]
                self.board_temp += [['X'] + list(line) + ['X']]
            self.board_temp += [['X'] * len(self.board_temp[0])]

        # Set dimensions 
        self.board_rows = len(self.board_temp)
        self.board_cols = len(self.board_temp[0])

        self.board = Grid(self.board_rows, self.board_cols, '')
        self.board.array = self.board_temp

        # Create panes 
        padding = 1
        widths = [self.board_rows if o % 2 == 0 else self.board_cols for o in orientations]
        cumulative_widths = [0]
        for w in widths:
            cumulative_widths += [cumulative_widths[-1] + w + 2*padding]
        self.panes = [Pane(
            Position(padding,padding+w),  
            o,
            [self.board_rows, self.board_cols][(o+1) % 2], 
            [self.board_rows, self.board_cols][o % 2]
        ) for i, (w, o) in enumerate(zip(cumulative_widths, orientations))]

        # Set frame dimensions
        total_rows = max([self.board_rows if o % 2 == 1 else self.board_cols for o in orientations])
        total_cols = cumulative_widths[-1]

        self.frame_rows = total_rows + padding * 2
        self.frame_cols = total_cols 

    def find_jellies(self, board=None, component=None, moving_jellies=None):
        """Find all connected jellies."""
        if not board:
            board = self.board
            component = self.component

        jellies = []
        jelly_positions = [[]]
        def dfs(pos, jelly_id, visited):
            visited[pos] = len(jelly_positions) 
            jelly_positions[-1] += [pos] 
            for direction in Direction:
                delta = Position(direction=direction)
                if out_of_bounds(pos+delta,self.board_rows,self.board_cols):
                    continue
                if visited[pos+delta] != -1 or board[pos+delta] != jelly_id:
                    continue
                if not moving_jellies \
                        or ((pos not in moving_jellies) \
                        and (pos+delta not in moving_jellies)) \
                        or component[pos] == component[pos+delta]:
                    dfs(pos+delta,jelly_id,visited)

        visited = Grid(self.board_rows, self.board_cols, -1)
        for r in range(self.board_rows):
            for c in range(self.board_cols):
                pos = Position(r,c)
                if visited[pos] == -1 and board[pos].isdigit():
                        dfs(pos,board[pos],visited)
                        jelly_positions += [[]]

        for lst in jelly_positions[:-1]:
            jellies += [Jelly(lst)]
        return jellies, visited

    def determine_free_jellies(self, starting_positions, current_direction, board=None, component=None):
        """Returns a list of Positions."""
        if not board:
            board = self.board
            component = self.component

        def dfs(pos, target_direction, visited):
            visited[pos] = True
            for direction in Direction:
                delta = Position(direction=direction)
                if out_of_bounds(pos+delta,self.board_rows,self.board_cols):
                    continue
                if visited[pos+delta]: 
                    continue
                if board[pos+delta].isdigit() and (component[pos+delta] == component[pos] or direction == target_direction):
                    dfs(pos+delta,target_direction,visited)

        opposite_direction = Direction(current_direction).opposite()
        visited = Grid(self.board_rows, self.board_cols, False)
        for r in range(self.board_rows):
            for c in range(self.board_cols):
                pos = Position(r,c)
                delta = Position(direction=opposite_direction)
                if out_of_bounds(pos+delta,self.board_rows,self.board_cols):
                    continue
                if board[pos] == Square.WALL and not visited[pos+delta] and board[pos+delta].isdigit():
                    dfs(pos+delta,opposite_direction,visited)

        blocked_jellies_temp = [Position(i,j) for j in range(self.board_cols) for i in range(self.board_rows) if visited[Position(i,j)]]
        free_jellies = [pos for pos in starting_positions if pos not in blocked_jellies_temp]
        visited = Grid(self.board_rows, self.board_cols, False)
        for pos in starting_positions:
            if pos in blocked_jellies_temp or visited[pos]:
                continue 
            dfs(pos,current_direction,visited)
        free_jellies = [Position(i,j) for j in range(self.board_cols) for i in range(self.board_rows) if visited[Position(i,j)]]

        blocked_jellies = Grid(self.board_rows, self.board_cols, None)
        blocked_jellies.array = blocked_jellies_temp

        return free_jellies, blocked_jellies

    def next_board_and_positions(self, jelly_positions, floating_jellies, sitting_jellies, direction, is_gravity):
        # Update board 
        delta = Position(direction=direction)
        board = copy.deepcopy(self.board)
        component = copy.deepcopy(self.component)
        ids = [board[p] for p in jelly_positions]
        comps = [component[p] for p in jelly_positions]
        for p in jelly_positions:
            board[p] = Square.BLANK
            component[p] = -1
        for jelly_id, comp_id, p in zip(ids, comps, jelly_positions):
            board[p+delta] = jelly_id
            component[p+delta] = comp_id
        if self.future_selected_jelly.positions[0] in jelly_positions:
            self.future_selected_jelly = Jelly([p+delta for p in self.future_selected_jelly.positions])
        non_moving_jellies = [pos for pos in sitting_jellies if pos not in jelly_positions] # jellies whose floor may have been pulled from under them
        moving_jellies = [pos for pos in jelly_positions if pos in sitting_jellies or is_gravity] 
        next_free_jellies = [p+delta for p in moving_jellies] + non_moving_jellies
        next_floating_jellies = [p+delta if p in moving_jellies else p for p in floating_jellies]
        return board, component, next_free_jellies, next_floating_jellies

    def get_all_jellies(self, board=None):
        if not board:
            board = self.board
        return [Position(i,j) for j in range(self.board_cols) for i in range(self.board_rows) if board[i][j].isdigit()]

    def animate_jellies(self, jelly_positions, falling_jellies, direction, future_board, component):
        all_jellies = self.get_all_jellies()

        frame = Grid(self.frame_rows, self.frame_cols, ' ')
        for pane in self.panes:
            for i in range(self.board_rows):
                for j in range(self.board_cols):
                    orig_pos = Position(i,j)
                    pos = pane.to_coord(orig_pos)
                    frame[pane.position+pos] = self.board[orig_pos]

        jelly_state = Grid(self.frame_rows, self.frame_cols, -1)
        for pane in self.panes:
            free_jellies, _ = self.determine_free_jellies(all_jellies, pane.direction_gravity)
            for pos in free_jellies:
                pos = pane.to_coord(pos)
                jelly_state[pane.position+pos] = JellyState.FLOATING

        jelly_component = Grid(self.frame_rows, self.frame_cols, -1)
        for i, jelly in enumerate(self.jellies):
            for j, pane in enumerate(self.panes):
                component_id = len(self.jellies) * j + i
                for pos in jelly.positions:
                    pos = pane.to_coord(pos)
                    jelly_component[pane.position+pos] = component_id

        all_jellies = self.get_all_jellies(future_board)
        future_state = Grid(self.frame_rows, self.frame_cols, -1)
        for pane in self.panes:
            free_jellies, _ = self.determine_free_jellies(all_jellies, pane.direction_gravity, future_board, component)
            for pos in free_jellies:
                pos = pane.to_coord(pos)
                future_state[pane.position+pos] = JellyState.FLOATING

        future_jellies, component = self.find_jellies(board=future_board, component=component, moving_jellies=falling_jellies)
        future_component = Grid(self.frame_rows, self.frame_cols, -1)
        for i, jelly in enumerate(future_jellies):
            for j, pane in enumerate(self.panes):
                component_id = len(future_jellies) * j + i
                for pos in jelly.positions:
                    pos = pane.to_coord(pos)
                    future_component[pane.position+pos] = component_id

        moving_squares = Grid(self.frame_rows, self.frame_cols, 0)
        for pane in self.panes:
            orientation_direction_map = [
                [],
                [-1, 1, 2, 3, 4],
                [-1, 4, 1, 2, 3],
                [-1, 3, 4, 1, 2],
                [-1, 2, 3, 4, 1],
            ]
            pane_direction = orientation_direction_map[pane.orientation][direction]
            for i in range(self.board_rows):
                for j in range(self.board_cols):
                    pos = pane.to_coord(Position(i,j))
                    moving_squares[pane.position+pos] = -1 * abs(pane_direction)
            for pos in jelly_positions:
                pos = pane.to_coord(pos)
                moving_squares[pane.position+pos] = abs(pane_direction)

        animate_partial_func = None
        while animate_partial_func is None:
            animate_partial_func, num_frames = self.renderer.animate_partial(
                frame, [jelly_state, future_state], [jelly_component, future_component], moving_squares, self.render_messages
            )
        for k in range(num_frames):
            animate_partial_func(k)
            key = self.stdscr.getch()
            for dict_key, dict_val in KEY_BINDINGS.items():
                if dict_key in ['arrow_left', 'arrow_right'] and key in dict_val:
                    self.queued_key_events += str(KEY_BINDINGS_BASE[dict_key][0])
                    continue 
                if dict_key in ['select_prev', 'select_next', 'redo', 'toggle_welcome_message'] and key in dict_val:
                    continue
                if key in dict_val:
                    return False, {'pressed_key': str(KEY_BINDINGS_BASE[dict_key][0])} # representative should be string
        return True, {}

    def try_move(self, jelly, current_direction, pane):
        # Check if way is blocked 
        # The way is blocked iff there exists a path from a jelly square to the side of a gray square in the target direction
        free_jellies, blocked_jellies = self.determine_free_jellies(jelly.positions, current_direction)
        all_jellies = self.get_all_jellies()
        floating_jellies, sitting_jellies = self.determine_free_jellies(all_jellies, pane.direction_gravity)
        execute_first_move = True
        self.future_selected_jelly = copy.deepcopy(self.jellies[self.selected_jelly % len(self.jellies)])
        animate = self.should_animate and self.enable_animation
        draw = True
        while free_jellies:
            if execute_first_move:
                self.board_stack_prev.append([self.board, self.selected_jelly])
                self.board_stack_next = []
            # Apply update 
            # next_free_jellies stores jellies that were moving/can start moving
            board, component, next_free_jellies, next_floating_jellies = self.next_board_and_positions(
                free_jellies, 
                floating_jellies,
                sitting_jellies, 
                current_direction, 
                not execute_first_move
            )
            if execute_first_move: 
                falling_jellies, _ = self.determine_free_jellies(next_free_jellies, pane.direction_gravity, board=board, component=component)
            else:
                falling_jellies = [p for p in next_free_jellies if p in next_floating_jellies] # these ones are in continuous motion
            if animate:
                success, result = self.animate_jellies(free_jellies, falling_jellies, current_direction, board, component)
                if not success:
                    if result['pressed_key'] in KEY_BINDINGS['undo']:
                        draw = False # Don't show the end result of the move
                    self.push_key_events(result['pressed_key'])
                    self.queued_key_events = ''
                    animate = False # Skip the rest of animation for this move
            self.board = board 
            self.component = component 
            if execute_first_move:
                self.jellies, self.component = self.find_jellies(moving_jellies=falling_jellies)
            else:
                # Any jelly that is moving after the first frame is floating+falling
                falling_jellies = [p for p in next_free_jellies if p in next_floating_jellies]
                self.jellies, self.component = self.find_jellies(moving_jellies=falling_jellies) 
            current_direction = pane.direction_gravity
            free_jellies, blocked_jellies = self.determine_free_jellies(next_free_jellies, current_direction)
            all_jellies = self.get_all_jellies()
            floating_jellies, sitting_jellies = self.determine_free_jellies(all_jellies, pane.direction_gravity)
            execute_first_move = False
        # Updates
        self.update_members()
        if draw:
            self.draw() # Indicate selected jelly
        if execute_first_move:
            # Produce error message
            self.renderer.print('That direction is blocked.')
            curses.flushinp()

    def push_key_events(self, s):
        """s: str"""
        self.replay_string = self.replay_string[:self.replay_idx] + s + self.replay_string[self.replay_idx:]
        for c in s:
            log(c, self.replay_file)

    def update_members(self):
        if self.selected_jelly != -1 and self.future_selected_jelly:
            selected_jelly_pos = self.future_selected_jelly.positions[0]
        self.jellies, self.component = self.find_jellies()
        all_jellies = [jelly.positions[0] for jelly in self.jellies]
        for pane in self.panes:
            pane.jelly_state = Grid(self.board_rows, self.board_cols, -1)
            free_jellies, _ = self.determine_free_jellies(all_jellies, pane.direction_gravity)
            for pos in all_jellies:
                pane.jelly_state[pos] = 'X'
            for pos in free_jellies:
                pane.jelly_state[pos] = JellyState.FLOATING
        if self.selected_jelly != -1 and self.future_selected_jelly:
            while selected_jelly_pos not in self.jellies[self.selected_jelly % len(self.jellies)].positions:
                self.selected_jelly = (self.selected_jelly + 1) % (len(self.jellies) * len(self.panes))
            # Correct the pane 
            self.selected_jelly = self.selected_jelly % len(self.jellies) + len(self.jellies) * self.selected_pane
        elif self.selected_jelly == -1:
            self.selected_jelly = 0
        while self.panes[self.selected_jelly // len(self.jellies)].is_floating(
                self.jellies[self.selected_jelly % len(self.jellies)]
            ):
            self.selected_jelly = (self.selected_jelly + 1) % (len(self.jellies) * len(self.panes))
        self.selected_pane = self.selected_jelly // len(self.jellies)
        self.future_selected_jelly = None

    def load_level(self):
        self.load_from_file('./levels/%s/%d.txt' % (self.level_subdir, self.current_level))
        self.selected_jelly = -1
        self.selected_pane = -1
        self.update_members()
        self.draw()
        self.board_stack_prev = []
        self.board_stack_next = []

    def run(self):
        self.renderer.resize()
        self.load_level()
        while not self.exited:
            if len(self.replay_string) > 0 and self.replay_idx == len(self.replay_string):
                self.draw()
            self.should_log_key_event = self.replay_idx >= len(self.replay_string)
            key = self.read_char()
            if self.should_log_key_event:
                for dict_key, dict_val in KEY_BINDINGS.items():
                    if dict_key != 'quit' and key in dict_val:
                        log(KEY_BINDINGS_BASE[dict_key][0], self.replay_file) # the representative
            if key in KEY_BINDINGS['select_next']: 
                self.selected_jelly = (self.selected_jelly + 1) % (len(self.jellies) * len(self.panes))
                while self.panes[self.selected_jelly // len(self.jellies)].is_floating(
                        self.jellies[self.selected_jelly % len(self.jellies)]
                    ):
                    self.selected_jelly = (self.selected_jelly + 1) % (len(self.jellies) * len(self.panes))
                self.selected_pane = self.selected_jelly // len(self.jellies)
                if self.should_animate and self.should_log_key_event:
                    self.draw()
            elif key in KEY_BINDINGS['select_prev']: 
                self.selected_jelly = (self.selected_jelly - 1) % (len(self.jellies) * len(self.panes))
                while self.panes[self.selected_jelly // len(self.jellies)].is_floating(
                        self.jellies[self.selected_jelly % len(self.jellies)]
                    ):
                    self.selected_jelly = (self.selected_jelly - 1) % (len(self.jellies) * len(self.panes))
                self.selected_pane = self.selected_jelly // len(self.jellies)
                if self.should_animate and self.should_log_key_event:
                    self.draw()
            elif key in KEY_BINDINGS['arrow_left'] + KEY_BINDINGS['arrow_right'] + [curses.KEY_MOUSE]:
                if self.selected_jelly == -1:
                    continue
                if key in KEY_BINDINGS['arrow_left']:
                    direction = Direction.LEFT 
                elif key in KEY_BINDINGS['arrow_right']:
                    direction = Direction.RIGHT 
                else:
                    id, y, x, z, bstate = curses.getmouse()
                    if not get_bit(bstate, curses.BUTTON1_CLICKED):
                        continue
                    window_pos = Position(x,y)
                    frame_pos = self.renderer.to_frame(window_pos)
                    in_pane = False
                    for pane_idx, pane in enumerate(self.panes):
                        pos = pane.from_frame(frame_pos)
                        if not out_of_bounds(pos, self.board_rows, self.board_cols):
                            in_pane = True
                            break 
                    if not in_pane:
                        continue 
                    pos = Position(round(pos.x), round(pos.y))
                    jelly_idx = [i for i in range(len(self.jellies)) if pos in self.jellies[i].positions]
                    if not jelly_idx:
                        continue 
                    jelly_idx = jelly_idx[0]
                    jelly = self.jellies[jelly_idx]
                    if pane.is_floating(jelly):
                        self.renderer.print('That jelly is floating.')
                        continue 
                    # Get side clicked 
                    xs = [p.x for p in jelly.positions]
                    ys = [p.y for p in jelly.positions]
                    mid = Position(
                        (max(xs) + min(xs)) * 1. / 2,
                        (max(ys) + min(ys)) * 1. / 2
                    )
                    frame_mid = pane.to_coord(mid) + pane.position
                    window_mid = self.renderer.to_window(frame_mid)
                    direction = Direction.RIGHT if y-window_mid.y > 0 else Direction.LEFT
                    self.selected_pane = pane_idx
                    self.selected_jelly = self.selected_pane * len(self.jellies) + jelly_idx
                    dict_key = 'arrow_left' if direction == Direction.LEFT else 'arrow_right'
                    ch = KEY_BINDINGS_BASE[dict_key][0].upper()
                    log('%s %d' % (ch, self.selected_jelly), self.replay_file)  
                if not self.should_log_key_event:
                    self.selected_jelly = self.try_read_int() or self.selected_jelly
                    self.selected_pane = self.selected_jelly // len(self.jellies)
                self.stdscr.nodelay(True)
                self.try_move(
                    self.jellies[self.selected_jelly % len(self.jellies)], 
                    (self.panes[self.selected_jelly // len(self.jellies)].orientation + direction - 1 - 1) % 4 + 1, 
                    self.panes[self.selected_jelly // len(self.jellies)]
                )
                self.stdscr.nodelay(False)
                # Check if player has completed the level
                distinct_colors = set([self.board[jelly.positions[0]] for jelly in self.jellies]) 
                if len(self.jellies) == len(distinct_colors):
                    # Level complete!
                    self.queued_key_events = ''
                    self.solved_levels[self.current_level] = True
                    self.show_welcome_message = False
                    if all(self.solved_levels):
                        self.renderer.print('All levels complete!')
                        curses.napms(1000)
                        self.play_basic_closing_screen()
                    else:
                        self.renderer.print('Level complete!')
                        curses.napms(1000)
                        while self.current_level < self.num_levels and self.solved_levels[self.current_level]:
                            self.current_level += 1
                        if self.current_level == self.num_levels:
                            self.current_level = [i for i, b in enumerate(self.solved_levels) if not b][0]
                        self.load_level()
                        self.renderer.print('Level complete!')
                if self.queued_key_events:
                    self.push_key_events(self.queued_key_events[0])
                    self.queued_key_events = self.queued_key_events[1:]
            elif key in KEY_BINDINGS['undo']:
                if not self.board_stack_prev:
                    self.renderer.print('No moves to undo.')
                    continue
                self.board_stack_next.append([self.board, self.selected_jelly])
                self.board, self.selected_jelly = self.board_stack_prev.pop()
                self.selected_pane = self.selected_jelly // len(self.jellies)
                self.future_selected_jelly = None
                self.update_members()
                self.draw()
            elif key in KEY_BINDINGS['redo']:
                if not self.board_stack_next:
                    self.renderer.print('No moves to redo.')
                    continue
                self.board_stack_prev.append([self.board, self.selected_jelly])
                self.board, self.selected_jelly = self.board_stack_next.pop()
                self.selected_pane = self.selected_jelly // len(self.jellies)
                self.future_selected_jelly = None
                self.update_members()
                self.draw()
            elif key in KEY_BINDINGS['level_restart']:
                if self.board_stack_prev and not self.confirm_action('Restart level?'):
                    continue
                self.load_level()
            elif key in KEY_BINDINGS['level_next']:
                if self.board_stack_prev and not self.confirm_action('Skip level?'):
                    continue
                if self.current_level == self.num_levels-1:
                    self.renderer.print('Cannot go past last level.')
                    continue 
                self.current_level += 1
                self.show_welcome_message = False
                self.load_level()
            elif key in KEY_BINDINGS['level_prev']:
                if self.board_stack_prev and not self.confirm_action('Go back to the previous level?'):
                    continue
                if self.current_level == 0:
                    self.renderer.print('Cannot go past first level.')
                    continue 
                self.current_level -= 1
                self.show_welcome_message = False
                self.load_level()
            elif key in KEY_BINDINGS['quit']: 
                if self.board_stack_prev and not self.confirm_action('Quit the game?',log_response=False):
                    log('n', self.replay_file)
                    continue
                self.exited = True
                curses.napms(100)
            elif key in KEY_BINDINGS['toggle_welcome_message']:
                self.show_welcome_message = not self.show_welcome_message
                self.draw()
            elif key in KEY_BINDINGS['toggle_animation']:
                self.enable_animation = not self.enable_animation
                self.draw()
            elif key in KEY_BINDINGS['resize_event']:
                self.renderer.resize()
                self.draw()
            if self.replay_idx == len(self.replay_string):
                self.should_animate = True

    def read_char(self):
        if self.replay_idx < len(self.replay_string):
            key = self.replay_string[self.replay_idx]
            self.replay_idx += 1
        else:
            key = self.stdscr.getch()
        return key

    def try_read_int(self):
        """Returns int from previous command if present"""
        if self.replay_idx == 0:
            return None 
        ch = self.replay_string[self.replay_idx-1]
        if ch >= 'A' and ch <= 'Z':
            self.replay_idx += 1
            s = ''
            while self.replay_idx < len(self.replay_string) and self.replay_string[self.replay_idx].isdigit():
                s += self.replay_string[self.replay_idx]
                self.replay_idx += 1
            if self.replay_idx == len(self.replay_string):
                self.should_animate = True
            return int(s)
        return None

    def confirm_action(self, msg='Are you sure?',log_response=True):
        self.renderer.print('%s (y/n)' % msg)
        key = self.read_char()
        while key not in ['y', 'n', ord('y'), ord('n'), ord('Y'), ord('N')]:
            key = self.read_char()
        self.renderer.print(' ' * (len(msg) + 6))
        if self.replay_idx >= len(self.replay_string) and log_response:
            log((chr(key) if not isinstance(key, str) else key).lower(), self.replay_file)
        return key in ['y', ord('y'), ord('Y')]

    def prompt_for_response(self, message, expected_hash):
        answer = ''
        max_answer_width = 40
        self.stdscr.clear()
        self.renderer.print(message,0,True)
        self.renderer.print('Your answer: ' + answer.rjust(max_answer_width)) # hardcoded max width because not bothering with string formatting 
        num_tries = 0
        while bcrypt.hashpw(answer.encode('utf-8'), expected_hash) != expected_hash and num_tries < 15:
            answer = ''
            key = ''
            enter_key_vals = [curses.KEY_ENTER, 10, 13]
            if num_tries == 0:
                self.renderer.print('Your answer: ' + ('*' * len(answer)).rjust(max_answer_width))
            else:
                self.renderer.print(' ' * (len('Your answer: ') + max_answer_width))
                self.renderer.print('Sorry, try again.')
            while key not in enter_key_vals:
                key = self.stdscr.getch()
                while (key < ord('a') or key > ord('z')) and key not in enter_key_vals and len(answer) < max_answer_width:
                    key = self.stdscr.getch()
                if key not in enter_key_vals:
                    answer += chr(key)
                self.renderer.print('Your answer: ' + ('*' * len(answer)).rjust(max_answer_width)) 
            num_tries += 1
        return answer

    def play_fancy_closing_screen(self):
        key = ''
        while key not in [ord('q'), ord('Q')]: 
            nobody_inspects_the_spammish_repetition = b'5I1UPwyNKzMvKOYNl48XoAzVJzE8h8/oifMIV7JXV939nMtn6xneWrWlbjt09oFOJ7p9Hd/pIDiqvEeROn0NOWLEmgP6WqfBF0KMSFoz6njJ/s3EoL1fttZXezpeSz5dsHP82EfJ0n1jUj0sdR3Va2WKtC3rGNeD+iqNpM8e86c6Or9qencu0rCwJjfVQjy39oxOtio8nH7crXOujDPkU2VXnwSj3SxmxKjx4r7aOF17iXj0o4eMw8c4fAA9Z/LCwU3MLN2np6TtsnsBdpyrXzmYa6HYyyLbuofQ9sxMVRoGVHHvuAZQywrqnkhB8likFmfDt9wE/sHfuya6APEx3KEKe4MPf4etB4ZX/9xIN1CfxC+pNmKHE7G41TPxX0vMzC38TQ0XTLcurYMTpU2MUDVFeKQIuQnzqpW4MOojen9t3pbom9Fbh1CuJXLf/tBAqavesRocF7IAt6HsiMT2hNJKNLYFUT4tW7OPsYyH+bpwxeyNpwlQ6d+0DjZptxFAqGpSVTv6p4Y2eaCn2PmNeLjjL3gxQPpeilYfjuDxXdYDORJPFC4ONtK6w32I2/U0ozOOeGMENeYvJzI20zwPG+4Czl4q52ij5YW0QEOp+rCGnl18di+FvS6CPqbdFNHyweb8y0dQkpgKVhWPhIpu8Z8qqWhAEDvOXl5/78t4fE5IfFg7Uu2vdvv/rygvbdK30Tsmms3XrYiZfM4oRWHr1B4YxaZk034vvu/K9gklmgK7Ien8wpvggVNCM8BTiVbcj7uMcE2716ZgzRAqVR6Or+wm5mt7+uFtMKIzztyYMcmNck6OcZgrOziJ90/1IifqeW394s4Kp6pn1Vtmd8l1woJJQIoA+ZBOBDVtl07GY8ol0fQY3sgjFWud8kIsiJNOl0verMtnKg9LYiQ2rH+BU+qU5Sabvy03AZe9Xc76Hsbd1QRCOwVGwFQHbOrFoIGRVzeL9qi6TXPemD9Nvd9sJXSJR+V3D7obIM+zW07j0hI='
            message_enc = b'WW91IGhhdmUgdW5sb2NrZWQgVG90YWxseSBUcnVlIFRlcm1pbmFsIFRyaXZpYSEgX19fXyBpcyB0aGUgbW90aGVyIG9mIGFsbCBhbGdvcml0aG1zLCBhbmQgdGhlIGVhcmxpZXN0IGV4YW1wbGUgb2YgaXQgYXBwZWFycyBpbiB0aGUgcG9wdWxhciB3b3JrLCBfX19fLiBUaGUgX19fX18gaXMgYSBjZXJ0YWluIGZydWl0IGVuam95ZWQgd2l0aCBwaW5rIG9yIGJsYWNrIHNhbHQgb24gYSBzdW1tZXIncyBkYXksIGFuZCBfX19fXyBpcy9hcmUgc29tZXRpbWVzIHRocm93biBhY3Jvc3MgdGhlIHN0cmVldCBpbiBhIGNvbXBldGl0aXZlIGdhbWUgb2Ygc2tpbGwgb24gbmlnaHRzIGluIFNlcHRlbWJlci4gV2hhdCBpcyB0aGUgMzYtY2hhcmFjdGVyIHN0cmluZyByZXN1bHRpbmcgZnJvbSB0aGUgY29uY2F0ZW5hdGlvbiBvZiBhbGwgNiBibGFua2VkIHdvcmRzLCBhbGwgbG93ZXJjYXNlPw=='
            message = base64.b64decode(message_enc).decode()
            expected_hash = '$2b$12$H35iBuT/vx3emKP3wZ9wnOo150VMxOL6vJD9QXGDGPrJjxFHIubCG'.encode('utf-8')
            secret_key = self.prompt_for_response(message.replace('\n','').replace('\t',''), expected_hash)
            secret_key_enc = hashlib.sha256(base64.b64encode(secret_key.encode('utf-8'))).digest()[:16]
            cipher = AESCipher(secret_key_enc)
            plain_text = cipher.decrypt(nobody_inspects_the_spammish_repetition).decode()
            line_width = 40
            row = 1
            self.stdscr.clear()
            for line in plain_text.split('\n'):
                window_rows, window_cols = get_window_size()
                col = window_cols//2 - line_width//2
                for idx in range(math.ceil(len(line)/line_width)):
                    self.stdscr.addstr(row,col,line[idx*line_width:idx*line_width+line_width].ljust(line_width))
                    row += 1
            self.stdscr.refresh()
            key = self.stdscr.getch()
            while key not in [ord('q'), ord('Q')]: 
                key = self.stdscr.getch()

    def play_basic_closing_screen(self):
        self.renderer.print('Game complete!')
        key = ''
        while key not in [ord('q'), ord('Q')]: 
            key = self.stdscr.getch()
            self.draw() # in case of screen resize
            self.renderer.print('Game complete!')

    def draw(self):
        frame = Grid(self.frame_rows, self.frame_cols, ' ')
        for pane in self.panes:
            for i in range(self.board_rows):
                for j in range(self.board_cols):
                    orig_pos = Position(i,j)
                    pos = pane.to_coord(orig_pos)
                    frame[pane.position+pos] = self.board[orig_pos]

        jelly_state = Grid(self.frame_rows, self.frame_cols, -1)
        for pane in self.panes:
            for jelly in self.jellies:
                for p in jelly.positions:
                    pos = pane.to_coord(p)
                    jelly_state[pane.position+pos] = pane.jelly_state[p] 

        if self.selected_jelly != -1:
            pane = self.panes[self.selected_jelly//len(self.jellies)]
            jelly = self.jellies[self.selected_jelly % len(self.jellies)]
            for pos in jelly.positions:
                pos = pane.to_coord(pos)
                jelly_state[pane.position+pos] = JellyState.SELECTED

        jelly_component = Grid(self.frame_rows, self.frame_cols, -1)
        for i, jelly in enumerate(self.jellies):
            for j, pane in enumerate(self.panes):
                component_id = len(self.jellies) * j + i
                for pos in jelly.positions:
                    pos = pane.to_coord(pos)
                    jelly_component[pane.position+pos] = component_id

        self.renderer.draw(frame, jelly_state, jelly_component, self.render_messages)

def run(stdscr):
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--replay', default='', help='Path to replay file.')
    args = parser.parse_args()
    game = JelliesGame(stdscr, replay_file=args.replay)
    game.run()

if __name__ == '__main__': 
    curses.wrapper(run)