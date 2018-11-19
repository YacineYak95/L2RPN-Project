



class MyAgent:
    """ The template to be used to create an agent: any controler of the power grid is expected to be a daughter of this
    class.
    """

    def __init__(self, environment):
        """Initialize a new agent."""
        #assert isinstance(environment, pypownet.environment.RunEnv)
        self.environment = environment

    def act(self, observation):
       
        #do_nothing_action = self.environment.action_space.get_do_nothing_action()

        return None#do_nothing_action

   