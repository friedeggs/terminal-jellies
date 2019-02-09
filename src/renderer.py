import math

import curses

from constants import JellyState, Direction, Justification
from util import log, get_window_size, out_of_bounds, Position, Grid

def generate_mappings(): 
    table = {
        ''' ///
            @@@
            @@@''': '-',

        ''' ///
            @@/ 
            @@/''': '+',

        ''' @@/
            @@/ 
            @@/''': ']',

        ''' /@@
            /@@ 
            /@@''': '[',

        ''' @@@
            @@@ 
            @@@''': '@',

        ''' ///
            @@/
            ///''': ']',

        ''' ///
            /@@
            ///''': '[',

        ''' /@/
            /@/
            ///''': '+', # '-',

        ''' /@/
            /@/
            /@/''': '@',

        ''' ///
            /@/
            ///''': '@',

# More cases:

        ''' @@@
            @@@
            @@/''': '+',

        ''' @@@
            @@/
            @@/''': '|',


        ''' @@@
            @@@
            @//''': '-',

        ''' @@@
            @@/
            @//''': '+',

        ''' @@/
            @@/
            //@''': '+',

        ''' ///
            ///
            ///''': ' ',

        ''' ///
            /@@
            @@@''': '+',

        ''' ///
            @@@
            //@''': '@', # '-'

        ''' /@/
            /@/
            @@/''': '@', # '|'

        ''' @/@
            @@@
            @@@''': '-',

        ''' @@@
            @@/
            @@@''': '|',
    }
    result = {}
    for k,v in table.items():
        key = ''.join(k.split()) # remove all whitespace
        result[key] = v
    # reflection
    for k,v in table.items():
        def flip_ver(w):
            """Flip about the vertical axis."""
            return [s[::-1] for s in w]
        def flip_hor(w):
            """Flip about the horizontal axis."""
            return w[::-1]
        def rotate(w):
            """Rotate 90 degrees counter-clockwise."""
            return [''.join([w[j][3-i-1] for j in range(3)]) for i in range(3)]
        def to_str(w):
            return ''.join([''.join(s) for s in w])
        window = [''.join(s.split()) for s in k.split('\n')]
        reflections = [flip_ver(window), flip_hor(window)]

        rotations = [window]
        should_add_rotations = True 
        for i in range(3):
            rotations += [rotate(rotations[-1])]
            if i % 2 == 0 and to_str(rotations[-1]) in result and to_str(rotations[-1]) != to_str(window):
                should_add_rotations = False 

        rotations_rfl = [flip_ver(window)]
        for i in range(3):
            rotations_rfl += [rotate(rotations_rfl[-1])]
            if i % 2 == 0 and to_str(rotations_rfl[-1]) in result and to_str(rotations_rfl[-1]) != to_str(window):
                should_add_rotations = False 

        if should_add_rotations:
            for w in rotations:
                result[to_str(w)] = v
            for w in rotations_rfl:
                result[to_str(w)] = v

        for w in reflections:
            if to_str(w) not in result:
                result[to_str(w)] = v

        if to_str(rotations[2]) not in result:
            result[to_str(rotations[2])] = v

        if to_str(rotations_rfl[2]) not in result:
            result[to_str(rotations_rfl[2])] = v
    return result

window_to_ascii_dict = generate_mappings()

class Renderer(object):
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.frame = None 

        self.mult_x = 1 
        self.mult_y = 1.8

        self.frames_per_transition = 1

        # Curses setup
        curses.init_pair(1, curses.COLOR_RED,       curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_GREEN,     curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_BLUE,      curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_YELLOW,    curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_MAGENTA,   curses.COLOR_BLACK)
        curses.init_pair(6, curses.COLOR_CYAN,      curses.COLOR_BLACK)
        curses.init_pair(7, curses.COLOR_WHITE,     curses.COLOR_BLACK)
        
        curses.curs_set(False) # Hide the cursor

    def ascii_representation(self, window, component_window, time_step=1.):
        self_char = window[1][1]
        self_comp = component_window[1][1]
        if self_char == 'X':
            return 'X'
        window_str = ''.join([''.join(w) for w in window])
        comp_window_str = ''.join([''.join(['X' if c == self_comp else '/' for c in w]) for w in component_window])
        def char_type(ch, c):
            if ch == self_char and ch.isdigit() and c == 'X':
                return '@'
            return '/'
        window_str = ''.join([char_type(ch, c) for ch, c in zip(window_str, comp_window_str)])
        if window_str not in window_to_ascii_dict: 
            return '@' 
        return window_to_ascii_dict[window_str]

    def char_to_color_pair(self, char):
        return int(char) if char.isdigit() else 0

    def _to_frame(self, window_x, window_y, time_step=0.0, moving_squares=None):
        # Solve: 
        #   window_x = int(((1 - time_step) * frame_x + time_step * (frame_x + direction_x)) * self.scale_x)
        #   window_y = int(((1 - time_step) * frame_y + time_step * (frame_y + direction_y)) * self.scale_y)
        direction = abs(moving_squares[min(
            self.frame_rows-1,int(window_x/self.scale_x))
        ][min(
            self.frame_cols-1,int(window_y/self.scale_y))
        ])
        delta = Position(direction=direction)

        frame_pos = Position(
            int(((window_x)/self.scale_x) - time_step * delta.x),
            int(((window_y)/self.scale_y) - time_step * delta.y)
        )
        if out_of_bounds(frame_pos, self.frame_rows, self.frame_cols):
            is_moving = False
        else:
            is_moving = moving_squares[frame_pos] > 0 if moving_squares else False
        if not is_moving:
            frame_pos = Position(
                int(window_x/self.scale_x),
                int(window_y/self.scale_y)
            )
        if out_of_bounds(frame_pos, self.frame_rows, self.frame_cols):
            return None, False
        return frame_pos, is_moving

    def _value_at(self, window_x, window_y, delta, time_step=0.0, arr=None, to_frame=None, moving_squares=None, return_future=False):
        CURRENT = 0
        FUTURE = 1
        if to_frame:
            if out_of_bounds(Position(window_x, window_y), self.window_rows, self.window_cols):
                frame_pos, is_moving = None, False
            else:
                frame_pos, is_moving = to_frame[window_x][window_y]
        else:
            frame_pos, is_moving = self._to_frame(window_x, window_y, delta, time_step, moving_squares)
        if not frame_pos:
            return ' '
        if out_of_bounds(frame_pos, self.frame_rows, self.frame_cols):
            is_neighbour_moving = False
        else:
            is_neighbour_moving = moving_squares[frame_pos] > 0 if moving_squares else False
        if is_neighbour_moving and not is_moving:
            return '.' if arr == self.frame else -1 
        if len(arr) != 2:
            curr = arr[frame_pos]
            future = None
        else:
            curr = arr[CURRENT][frame_pos]
            future = arr[FUTURE][frame_pos + delta] if is_moving else None
        return [curr, future][return_future]

    def animate_partial(self, frame, jelly_state, jelly_component, moving_squares, func_draw_additional): 
        """
        [curr_state, future_state]
        [curr_comp, future_comp]
        moving_squares: an integer indicating direction (0 if none) and positive sign if that square is moving in the indicated direction
        """
        self.frame = frame
        self.frame_rows = len(frame)
        self.frame_cols = len(frame[0])

        self.window_rows, self.window_cols = get_window_size()
        curses.resizeterm(self.window_rows+1, self.window_cols+1)
        # curses.flushinp()
        self.stdscr.clear()
        if self.window_rows <= self.frame_rows or self.window_cols <= self.frame_cols:
            self.print('Window size too small. Please resize your window.')
            return None, None 

        self.scale = min(
            self.window_rows/(self.frame_rows*self.mult_x),
            self.window_cols/(self.frame_cols*self.mult_y),
        )
        self.scale_x = self.scale * self.mult_x
        self.scale_y = self.scale * self.mult_y

        def delta_at(window_x, window_y):
            direction = abs(moving_squares[min(
                self.frame_rows-1,int(window_x/self.scale_x))
            ][min(
                self.frame_cols-1,int(window_y/self.scale_y))
            ])
            delta = Position(direction=direction)
            return delta
        delta_arr = [[delta_at(i,j) for j in range(self.window_cols)] for i in range(self.window_rows)]

        def value_at(arr, i, j):
            if i < 0 or j < 0 or i >= len(arr) or j >= len(arr[0]):
                return ' '
            return arr[i][j]

        def animate_partial_func(k):
            t = 1. * k / self.frames_per_transition
            to_frame = [[self._to_frame(i, j, t, moving_squares) for j in range(self.window_cols)] for i in range(self.window_rows)]
            frame_arr = [[self._value_at(i,j,delta_arr[i][j],t,self.frame,to_frame,moving_squares) for j in range(self.window_cols)] for i in range(self.window_rows)]
            comp_arr_next = [[self._value_at(i,j,delta_arr[i][j],t,jelly_component,to_frame,moving_squares,True) for j in range(self.window_cols)] for i in range(self.window_rows)]
            comp_arr_curr = [[self._value_at(i,j,delta_arr[i][j],t,jelly_component,to_frame,moving_squares) for j in range(self.window_cols)] for i in range(self.window_rows)]
            for i in range(self.window_rows):
                for j in range(self.window_cols):
                    direction = abs(moving_squares[min(
                        self.frame_rows-1,int(i/self.scale_x))
                    ][min(
                        self.frame_cols-1,int(j/self.scale_y))
                    ])
                    delta = Position(direction=direction)
                    window = [[value_at(frame_arr,i+di,j+dj) for dj in range(-1,2)] for di in range(-1,2)]
                    component_window_next = [[value_at(comp_arr_next,i+di,j+dj) for dj in range(-1,2)] for di in range(-1,2)]
                    component_window_curr = [[value_at(comp_arr_curr,i+di,j+dj) for dj in range(-1,2)] for di in range(-1,2)]
                    component_window = [[n if n else -c-1 for (n, c) in zip(nr, cr)] for (nr, cr) in zip(component_window_next, component_window_curr)]
                    char = self.ascii_representation(window, component_window, t)
                    if self._value_at(i,j,delta,t,jelly_state,to_frame,moving_squares) == JellyState.FLOATING: 
                        char = str(JellyState.FLOATING)
                    elif self._value_at(i,j,delta,0,jelly_state,to_frame,moving_squares) == JellyState.SELECTED and t == 1.: 
                        char = str(JellyState.SELECTED)
                    color_idx = self.char_to_color_pair(self._value_at(i,j,delta,t,self.frame,to_frame,moving_squares))
                    self.stdscr.addstr(i,j,char,curses.color_pair(color_idx))
            self.stdscr.refresh()
            func_draw_additional()
        num_frames = self.frames_per_transition + 1
        return animate_partial_func, num_frames

    def draw(self, frame, jelly_state, jelly_component, func_draw_additional): 
        """
        jelly_state: 2D array with same size as frame
        """
        self.frame = frame
        self.frame_rows = len(frame)
        self.frame_cols = len(frame[0])

        self.window_rows, self.window_cols = get_window_size()
        curses.resizeterm(self.window_rows+1, self.window_cols+1)
        # curses.flushinp()
        self.stdscr.clear() 
        if self.window_rows <= self.frame_rows or self.window_cols <= self.frame_cols:
            self.print('Window size too small. Please resize your window.')
            return False

        self.scale = min(
            self.window_rows/(self.frame_rows*self.mult_x),
            self.window_cols/(self.frame_cols*self.mult_y),
        )
        self.scale_x = self.scale * self.mult_x
        self.scale_y = self.scale * self.mult_y

        for i in range(self.window_rows):
            for j in range(self.window_cols):
                def value_at(window_x, window_y, arr=self.frame):
                    frame_pos = Position(
                        int((window_x)/self.scale_x),
                        int((window_y)/self.scale_y)
                    )
                    if out_of_bounds(frame_pos, self.frame_rows, self.frame_cols):
                        return ' '
                    return arr[frame_pos]

                window = [[value_at(i+di,j+dj) for dj in range(-1,2)] for di in range(-1,2)]
                component_window = [[value_at(i+di,j+dj,jelly_component) for dj in range(-1,2)] for di in range(-1,2)]
                char = self.ascii_representation(window, component_window)
                state = value_at(i,j,jelly_state)
                if state in [JellyState.FLOATING, JellyState.SELECTED]: 
                    char = str(state)
                color_idx = self.char_to_color_pair(value_at(i,j))
                self.stdscr.addstr(i,j,char,curses.color_pair(color_idx))
        self.stdscr.refresh()
        func_draw_additional()
        return True

    def print(self, message, location=None, wrap_around=False, justification=Justification.CENTER,pause=True):
        self.window_rows, self.window_cols = get_window_size()
        curses.resizeterm(self.window_rows+1, self.window_cols+1)
        # curses.flushinp()

        loc_y0 = location[1] if isinstance(location, list) else 0

        if wrap_around:
            line_width = self.window_cols - loc_y0 - 2 # padding
            lines = []
            for idx in range(math.ceil(len(message)/line_width)):
                lines += [message[idx*line_width:idx*line_width+line_width].ljust(line_width)]
        else:
            lines = [message]

        if justification == Justification.CENTER:
            loc_y = (self.window_cols - loc_y0 - len(lines[0]))//2 + loc_y0
        else:
            loc_y = loc_y0
        if location is not None:
            if isinstance(location, int):
                loc_x = location
            else:
                loc_x = location[0]
        else:
            loc_x = self.window_rows - 2

        for i, line in enumerate(lines):
            self.stdscr.addstr(loc_x+i,loc_y,' ' * len(line))
        self.stdscr.refresh()
        if pause:
            curses.napms(50)
        for i, line in enumerate(lines):
            self.stdscr.addstr(loc_x+i,loc_y,line)
        self.stdscr.refresh()

if __name__ == '__main__':
    print(window_to_ascii_dict)
    print(len(window_to_ascii_dict))
    def print_w(s):
        for i in range(3):
            print(s[3*i:3*(i+1)])
    for k, v in window_to_ascii_dict.items():
        print_w(k)
        print(v)
