class neuron:
    def __init__(self, inhib=False, on = False):
        self.inhib = inhib
        self.on = on
        self.t_delay = .1
        self.v_t = 105.1
        self.v_rest = 0
        self.v_min = 0
        self.r = 1
        self.c = 10
        self.clear()

    
    def is_spike(self):
        if self.v >= self.v_t:
            self.v = self.v_rest
            return True
        elif self.v < self.v_min:
            self.v = self.v_rest

        return False
    
    def clear(self):
        self.t_rest = -1
        self.v = self.v_rest
    

