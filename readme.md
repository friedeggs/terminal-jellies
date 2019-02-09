# Running

Don't forget to install dependencies:

```pip install -r requirements.txt```

To run:

```python src/jellies.py [-r, --replay <replay_file>]```

The history of your commands will be stored in the file `replay.log` and you can reload a game from this file. Restoring from a replay file skips move animation. 

# Caveats/warnings
- the game can't handle screen resizes in the middle of the animation of a move
- the ASCII art is sometimes ambiguous and distances in the grid can be difficult to tell

# Why did you do this?

Great question. Yes, the basic shape primitive in this game is a square and no, curses does not support things that might have made this more reasonable and yes, not having a cursor makes selecting jellies cumbersome. But mashing arrow keys into a terminal application displaying ASCII art has a certain stripped-down old-school charm and this wasn't meant for heavy-duty gameplay anyways. 
