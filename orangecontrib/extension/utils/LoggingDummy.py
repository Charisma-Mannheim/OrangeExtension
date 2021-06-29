from datetime import datetime
from inspect import currentframe, getframeinfo, stack

class PrinLog:
    def __init__(self):
        self.frameinfo = getframeinfo(stack()[1][0])
        self.now = datetime.now()
        self.current_time = self.now.strftime("%H:%M:%S")
        self.modul = self.frameinfo.filename
        self.mod = self.modul.split("\\")
        self.module = self.mod[-1]
        self.line = self.frameinfo.lineno

    def prinlog(self,message):

     output = self.current_time +" " + self.module + " " + "line: " + str(self.line) + " ... " + message
     return print(output)


#def debuginfo(message):
  #  caller = getframeinfo(stack()[1][0])
 #   print "%s:%d - %s" % (caller.filename, caller.lineno, message)

#def grr(arg):
 #   debuginfo(arg)
