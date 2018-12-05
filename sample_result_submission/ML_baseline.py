

from pypownet.Agent import Agent, TreeSearchLineServiceStatus
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import pickle

import os

class Encoder(object):
    """docstring for encoder"""
    def __init__(self):
        self.keys = dict()
        self.values = list()

    def fit(self,data):
        for x in data:
            if not tuple(x) in self.keys.keys():
                self.keys[tuple(x)]= len(self.values)
                self.values.append(tuple(x))
        return self

    def encode(self, data):
        return [self.keys[tuple(d)] for d in data] 

    def decode(self, data):
        return [self.values[int(d)] for d in data]



class ML(Agent):
    """ The template to be used to create an agent: any controler of the power grid is expected to be a daughter of this
    class.
    """
    datafile = "data.dump"
    modelfile =  "model.dump"
    def __init__(self, environment):
        """Initialize a new agent."""
        #model = load('ml/model.joblib')
        
        datatime = os.path.getmtime(ML.datafile)
        
        if os.path.exists(ML.modelfile):
            modeltime = os.path.getmtime(ML.modelfile)
        else :
            modeltime =0 
        if datatime > modeltime:
            ML.train()
        
        self.model, self.enc= pickle.load(open(ML.modelfile, 'rb'))
        self.environment =environment
        #self.tt = TreeSearchLineServiceStatus(environment)


    def act(self, observation):
        """Produces an action given an observation of the environment."""
        obs = observation.as_array()
        act = self.enc.decode(self.model.predict([obs]))[0]
        ret = self.environment.action_space.get_do_nothing_action()
        for i in range(len(ret)):
            ret[i] = act[i]
        return ret

    def train():
        datas = pickle.load(open(ML.datafile, "rb"))
        X, y = list(zip(*datas))
        enc = Encoder().fit(y)
        y= enc.encode(y)
        model = GradientBoostingClassifier(n_estimators= 500, max_depth=100)
        model.fit(X,y)
        pickle.dump((model,enc), open(ML.modelfile, 'wb'))

# Examples of baselines agents



class TrainerAgent(Agent):
    """ The template to be used to create an agent: any controler of the power grid is expected to be a daughter of this
    class.
    """

    def __init__(self, environment):
        """Initialize a new agent."""
        self.agent = TreeSearchLineServiceStatus(environment)
        self.actions = list()
        self.environment = environment

    def act(self, observation):
        action = self.agent.act(observation)
        self.actions.append( (observation.as_array(), action.as_array())) 

        return action

    def __del__(self):
        old_actions = list()
        if os.path.exists(ML.datafile):
            old_actions = pickle.load(open(ML.datafile, "rb"))
        wr = old_actions + self.actions
        pickle.dump(wr, open(ML.datafile, 'wb'))
