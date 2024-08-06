Requements: opencv-python, numpy, pillow

Choose mode in main: DEBUG, PLAY, SOLVE

Theres two options to load board:
- Write the path in the script in variable screenshot_path.
- Run the script through terminal and give the path as parameter like this:
    python Mimic_Capture.py ~/Desktop/1.png

* If the picture is not png (jpeg / jpg / webp), it will convert to png so in the next run you must change the extension in the path.

SOLVE: will solve your board and give you blocks to remove with higher benefit. (takes 60-120 secs)

PLAY: will load your board and will run the game in terminal.
      A block index consists of letter a-g for column index and numbers 1-7 for row index.
      For example: A1, A5, G4, C5
      In the end of game you will see record of your moves.
      
DEBUG: Different phones have different resolution. The program is aimed at the resolution of most phones.
* If the parameters not match the resolution the dots will not be in the center of the blocks and it will cause wrong results.
  To fix it do this steps:
  -  change mode to DEBUG
  -  run the script and look at the result image.
  -  the center dot should be in the center of Mimic block.
  -  change parameters in DEBUG section in main:
    -  use height_fix parameter first to change vertical location of the center dot.
    -  if it nessecary fix the horizontal location (probably dont have to) with width_fix parameter.
    -  now if the other dots not in the middle of their blocks use horizontal_fix, vertical_fix parameters.
  -  Finally, change the global parameters that in the head of the script to the correct values and switch mode.
 
  For any other questions: yarinl330@gmail.com
