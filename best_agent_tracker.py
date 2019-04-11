
#Keeps track of best agent
class Best_Agent_tracker(object):

    def __init__(self):
        self.best_agent = None
        self.best_score = 0

    #Returns stored agent
    def get_best_agent(self):
        return self.best_agent

    #If provided agent gives better score it is chosen as best agent
    def update_best_agent(self, agent, score):
        if score > self.best_score or self.best_agent == None:

            if self.best_agent != None: print("New best agent! Score:", score, "Increase:", score - self.best_score)

            self.best_agent = agent
            self.best_score = score
