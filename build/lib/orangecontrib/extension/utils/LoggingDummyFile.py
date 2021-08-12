from datetime import datetime
from inspect import currentframe, getframeinfo, stack
import os

class PrinLog:
    def __init__(self, fileName = ""):
        self.frameinfo = getframeinfo(stack()[1][0])
        self.now = datetime.now()
        self.current_time = self.now.strftime("%H:%M:%S")
        self.modul = self.frameinfo.filename
        self.mod = self.modul.split("\\")
        self.module = self.mod[-1]
        self.line = self.frameinfo.lineno
        if (fileName == ""):
            self.logFile = "F:\LeslieNeuerMaster\SelbstGeschriebeneProgramme\AddOnsAKWeller\loggingDummy.txt"
        else:
            f = "\\" + fileName
            self.logFile = "F:\LeslieNeuerMaster\SelbstGeschriebeneProgramme\AddOnsAKWeller"+f
        print(self.logFile)


    def prinlog(self,message):

     output = self.current_time +" " + self.module + " " + "line: " + str(self.line) + " ... " + message
     #self.writeMsgToLog(output)
     return print(output)

    def writeMsgToLog(self,output):
        #DIESES FILE KÖNNTE AUCH KONSTANT OFFEN GEHALTEN WERDEN
        f = open(self.logFile, "a")
        f.write(output)
        f.write("\n")
        f.close()

#def debuginfo(message):
  #  caller = getframeinfo(stack()[1][0])
 #   print "%s:%d - %s" % (caller.filename, caller.lineno, message)

#def grr(arg):
 #   debuginfo(arg)
