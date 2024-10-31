class SimpleActivator:
    "activation class modifier"
    def __init__(self, k_A: str):
        self.k_A = k_A

    def add_modifier(self, activator, eval_dict):
        k_A = eval_dict[self.k_A]
        return 1 + activator / k_A
    

class SimpleInhibitor:
    """inhibition class modifier"""
    def __init__(self,
                 k_I:str):
        super(SimpleInhibitor, self).__init__()
        self.k_I=k_I

    def add_modifier(self, inhibitor,eval_dict):
        k_I=eval_dict[self.k_I]
        return 1/ (1+ inhibitor/k_I) 
