class Automata: #Define the automata class
    def __init__(self, states, alphabet, transition_function, initial_state, final_states):
        self.states = states  #Declare object attributes
        self.alphabet = alphabet
        self.transition_function = transition_function
        self.initial_state = initial_state
        self.final_states = final_states
        self.current_state = initial_state
    
    def transition(self, symbol):#Define transition method acting on self, and the input symbol
        if (self.current_state, symbol) in self.transition_function:   #If the symbol has a valid state transition
            self.current_state = self.transition_function[(self.current_state, symbol)] #Do the transition function on the state.
    
    def is_accepting(self): #def function for if the string accepts into the machine (ends on a final state)
        return self.current_state in self.final_states #Return the state after string input

    def reset(self): #Resets the machine to unitial state
        self.current_state = self.initial_state
