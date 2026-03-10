import json

class Logger:

    def __init__(self,path="logs.json"):

        self.path = path
        self.logs = []

    def log(self,step,loss):

        entry = {
            "step":step,
            "loss":loss
        }

        self.logs.append(entry)

        with open(self.path,"w") as f:

            json.dump(self.logs,f,indent=2)