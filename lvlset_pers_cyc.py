from collections import namedtuple
import mainModule
import ACTIVATE


################ PARAMETERS FOR THE PROGRAM ################

filename="snake.off"
direction=(1544, 1199)
skip_vertices=[]
type = "co"
# type = "oo"
num_of_intervals=2

############################################################


if __name__ == "__main__":
    script_file = namedtuple('script_file', ['desc', 'name', 'direction', 'skip_vertices'])

    sf = script_file(desc="", \
        name=filename, \
        direction=direction, \
        skip_vertices=skip_vertices)

    if type == "oo":
        mainModule.runMain(filename, direction, skip_vertices)
    elif type == "co":
        ACTIVATE.closedOpenStep2(sf, num_of_intervals)
