import os
import sys
import inspect


cmd_subfolder = os.path.split(inspect.getfile(inspect.currentframe()))[0]
print(cmd_subfolder)
if cmd_subfolder not in sys.path:
    pass
    sys.path.insert(0, cmd_subfolder)
