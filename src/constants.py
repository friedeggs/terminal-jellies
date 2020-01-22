import curses
from enum import Enum, IntEnum

ENABLE_ANIMATION = True # Set to False to disable animation

KEY_BINDINGS_BASE = { 
# Edit this dict if you want to change the key bindings.
    'select_prev': ['w'],
    'select_next': ['s', '\t'],
    'arrow_left': ['a', curses.KEY_LEFT],
    'arrow_right': ['d', curses.KEY_RIGHT],
    'level_prev': ['p'],
    'level_next': ['n'],
    'undo': ['u'],
    'redo': ['y'],
    'level_restart': ['r'],
    'quit': ['q'],
    'toggle_welcome_message': ['m'],
    'toggle_animation': ['x'],
    'resize_event': [curses.KEY_RESIZE]
}

KEY_BINDINGS = {}
for k, v in KEY_BINDINGS_BASE.items():
    KEY_BINDINGS[k] = [el for ch in v for el in (list(set([ch,ch.upper()])) + list(set([ord(ch),ord(ch.upper())])) if isinstance(ch, str) else [ch])]
ALL_KEYS = [ch for v in KEY_BINDINGS.values() for ch in v]

class JellyState(IntEnum):
    FLOATING    = 1
    SELECTED    = 2 

    def __str__(self):
        if self.name == 'FLOATING':
            return '.'
        if self.name == 'SELECTED':
            return '*'
        return ' '

class Direction(IntEnum):
    NONE    = 0
    UP      = 1
    LEFT    = 2
    DOWN    = 3
    RIGHT   = 4

    def opposite(self):
        return Direction((self.value + 2 - 1) % 4 + 1)

class Orientation(IntEnum):
    UPRIGHT         = 1
    ROTATED_LEFT    = 2
    UPSIDE_DOWN     = 3
    ROTATED_RIGHT   = 4

class Square(object):
    BLANK   = '.'
    WALL    = 'X'

class Justification(Enum):
    LEFT    = 1
    CENTER  = 2

if __name__ == '__main__':
    print(str(JellyState.FLOATING))
    print(str(JellyState.SELECTED))
    print(KEY_BINDINGS)
    print(ALL_KEYS)