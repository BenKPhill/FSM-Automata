#for rule 30
states1 = {"D01","A11","A01","D11","A02","D12", "A13", "D13", "D14", "N1", "N0", "A03"}  #Based on the diagram of the rule
alphabet = {0,1} #Input is always a 0 or 1
final_states1 = states1  #All states are final states
initial_state = "O" #Define O as the initial state


transition_function101 = { #Transition function, (state, input symbol):next state
    ("O",1):"D11",
    ("O",0):"D01",
    ("D11",1):"A11",
    ("D11",0):"D11",
    ("D01",1):"A01",
    ("D01",0):"D01",
    ("A11",1):"D12",
    ("A11",0):"A11",
    ("A01",1):"A02",
    ("A01",0):"A01",  
    }

transition_function30 = { #Transition function, (state, input symbol):next state
    ("O",1):"A11",
    ("O",0):"D01",
    ("A11",1):"D11",
    ("A11",0):"A11",
    ("D01",1):"A01",
    ("D01",0):"D01",
    ("D11",1):"D12",
    ("D11",0):"D11",
    ("A01",1):"A02",
    ("A01",0):"A01",  
    }

transition_function54 = { #Transition function, (state, input symbol):next state
    ("O",1):"A11",
    ("O",0):"D01",
    ("A11",1):"D11",
    ("A11",0):"A13",
    ("A13",1):"A12",
    ("A13",0):"A13",
    ("D01",1):"A01",
    ("D01",0):"D01",
    ("D11",1):"D12",
    ("D11",0):"D11",
    ("A01",1):"D02",
    ("A01",0):"A01",  
    }

transition_function14 = { #Transition function, (state, input symbol):next state
    ("O",1):"A11",
    ("O",0):"D01",
    ("A11",1):"D13",
    ("A11",0):"D11",
    ("D11",1):"D12",
    ("D11",0):"D11",
    ("D13",1):"D14",
    ("D13",0):"D13",
    ("D01",1):"A01",
    ("D01",0):"D01",
    ("A01",1):"A02",
    ("A01",0):"A01",  
    }

transition_function110 = { #Transition function, (state, input symbol):next state
    ("O",1):"D11",
    ("O",0):"D01",
    ("D11",1):"A11",
    ("D11",0):"D11",
    ("A11",1):"D12",
    ("A11",0):"A11",
    ("D01",1):"A01",
    ("D01",0):"D01",
    ("A01",1):"A02",
    ("A01",0):"A01",  
    }

transition_function126 = { #Transition function, (state, input symbol):next state
    ("O",1):"A11",
    ("O",0):"D01",
    ("A11",1):"A12",
    ("A11",0):"A11",
    ("A12",1):"D12",
    ("A12",0):"A12",
    ("D01",1):"A01",
    ("D01",0):"D01",
    ("A01",1):"A02",
    ("A01",0):"A01",  
    }

transition_function150 = { #Transition function, (state, input symbol):next state
    ("O",1):"A11",
    ("O",0):"D01",
    ("A11",1):"D11",
    ("A11",0):"A11",
    ("D11",1):"A12",
    ("D11",0):"D11",
    ("D01",1):"A01",
    ("D01",0):"D01",
    ("A01",1):"D02",
    ("A01",0):"A01",  
    }

transition_function90 = { #Transition function, (state, input symbol):next state
    ("O",1):"N11",
    ("O",0):"N01",
    ("N11",1):"A12",
    ("N11",0):"A11",
    ("A12",1):"D12",
    ("A12",0):"A12",
    ("A11",1):"D11",
    ("A11",0):"A11",
    ("N01",1):"D02",
    ("N01",0):"D01",
    ("D02",1):"A02",
    ("D02",0):"D02",
    ("D01",1):"A01",
    ("D01",0):"D01",
    }

transition_function22 = { #Transition function, (state, input symbol):next state
    ("O",1):"A11",
    ("O",0):"D01",
    ("A11",1):"D11",
    ("A11",0):"A11",
    ("D11",1):"D12",
    ("D11",0):"D11",
    ("D01",1):"A01",
    ("D01",0):"D01",
    ("A01",1):"D02",
    ("A01",0):"A01",  
    }

transition_function60 = { #Transition function, (state, input symbol):next state
    ("O",1):"N11",
    ("O",0):"N01",
    ("N11",1):"D11",
    ("N11",0):"A11",
    ("D11",1):"D12",
    ("D11",0):"D11",
    ("A11",1):"A12",
    ("A11",0):"A11",
    ("N01",1):"A01",
    ("N01",0):"D01",
    ("A01",1):"A02",
    ("A01",0):"A01",
    ("D01",1):"D01",
    ("D01",0):"D01",
    }

transition_function222 = { #Transition function, (state, input symbol):next state
    ("O",1):"N11",
    ("O",0):"D01",
    ("N11",1):"A12",
    ("N11",0):"A11",
    ("A12",1):"A13",
    ("A12",0):"A12",
    ("A11",1):"D11",
    ("A11",0):"A11",
    ("D01",1):"A01",
    ("D01",0):"D01",
    ("A01",1):"A02",
    ("A01",0):"A01",
    }

transition_function188 = { #Transition function, (state, input symbol):next state
    ("O",1):"N11",
    ("O",0):"N01",
    ("N11",1):"D11",
    ("N11",0):"A11",
    ("D11",1):"A13",
    ("D11",0):"D11",
    ("A11",1):"A12",
    ("A11",0):"A11",
    ("N01",1):"A02",
    ("N01",0):"D01",
    ("A02",1):"A03",
    ("A02",0):"A02",
    ("D01",1):"A01",
    ("D01",0):"D01",
    }

transition_function62 = { #Transition function, (state, input symbol):next state
    ("O",1):"N11",
    ("O",0):"D01",
    ("N11",1):"D11",
    ("N11",0):"A11",
    ("D11",1):"D12",
    ("D11",0):"D11",
    ("A11",1):"A12",
    ("A11",0):"A11",
    ("D01",1):"A01",
    ("D01",0):"D01",
    ("A01",1):"A02",
    ("A01",0):"A01",
    }

transition_function86 = { #Transition function, (state, input symbol):next state
    ("O",1):"N11",
    ("O",0):"D01",
    ("N11",1):"A12",
    ("N11",0):"A11",
    ("A12",1):"D12",
    ("A12",0):"A12",
    ("A11",1):"D11",
    ("A11",0):"A11",
    ("D01",1):"A01",
    ("D01",0):"D01",
    ("A01",1):"D02",
    ("A01",0):"A01",
    }

transition_function147 = { #Transition function, (state, input symbol):next state
    ("O",1):"A11",
    ("O",0):"N01",
    ("A11",1):"D11",
    ("A11",0):"A11",
    ("D11",1):"A12",
    ("D11",0):"D11",
    ("N01",1):"D01",
    ("N01",0):"A01",
    ("D01",1):"D02",
    ("D01",0):"D02",
    ("A01",1):"A02",
    ("A01",0):"A01",  
    }

transition_function4 = { #Transition function, (state, input symbol):next state
    ("O",1):"D11",
    ("O",0):"N01",
    ("D11",1):"D11",
    ("D11",0):"D12",
    ("D12",1):"D13",
    ("D12",0):"D12",
    ("N01",1):"A01",
    ("N01",0):"D01",
    ("A01",1):"D03",
    ("A01",0):"A01",
    ("D01",1):"D02",
    ("D01",0):"D01",  
    }

transition_function102 = { #Transition function, (state, input symbol):next state
    ("O",1):"D11",
    ("O",0):"D01",
    ("D11",1):"A11",
    ("D11",0):"D11",
    ("A11",1):"D12",
    ("A11",0):"A11",
    ("D01",1):"A01",
    ("D01",0):"D01",
    ("A01",1):"D02",
    ("A01",0):"A01",  
    } 

transition_function45 = { #Transition function, (state, input symbol):next state
    ("O",1):"N11",
    ("O",0):"N01",
    ("N11",1):"D12",
    ("N11",0):"D11",
    ("D12",1):"D13",
    ("D12",0):"D12",
    ("D11",1):"A11",
    ("D11",0):"D11",
    ("N01",1):"A02",
    ("N01",0):"A01",
    ("A02",1):"A03",
    ("A02",0):"A02",
    ("A01",1):"D01",
    ("A01",0):"A01",
    }

#2d automata
conways_game = { #Transition function, (state, input symbol):next state
    ("O",1):"D11",
    ("O",0):"D01",
    ("D11",1):"D12",
    ("D11",0):"D11",
    ("D12",1):"A11",
    ("D12",0):"D12",
    ("A11",1):"A12",
    ("A11",0):"A11",
    ("A12",1):"D13",
    ("A12",0):"A12", 
    ("D13",1):"D13",
    ("D13",0):"D13",
    ("D01",1):"D02",
    ("D01",0):"D01",
    ("D02",1):"D03",
    ("D02",0):"D02",
    ("D03",1):"A01",
    ("D03",0):"D03",
    ("A01",1):"D13",
    } 

Rule_30 = Automata(states1, alphabet, transition_function30, initial_state, final_states1) #Define rule32 object with the given attributes
Rule_101 = Automata(states1, alphabet, transition_function101, initial_state, final_states1)
Rule_54 = Automata(states1, alphabet, transition_function54, initial_state, final_states1)
Rule_14 = Automata(states1, alphabet, transition_function14, initial_state, final_states1)
Rule_110 = Automata(states1, alphabet, transition_function110, initial_state, final_states1)
Rule_126 = Automata(states1, alphabet, transition_function126, initial_state, final_states1)
Rule_150 = Automata(states1, alphabet, transition_function150, initial_state, final_states1)
Rule_90 = Automata(states1, alphabet, transition_function90, initial_state, final_states1)
Rule_22 = Automata(states1, alphabet, transition_function22, initial_state, final_states1)
Rule_60 = Automata(states1, alphabet, transition_function60, initial_state, final_states1)
Rule_222 = Automata(states1, alphabet, transition_function222, initial_state, final_states1)
Rule_188 = Automata(states1, alphabet, transition_function188, initial_state, final_states1)
Rule_62 = Automata(states1, alphabet, transition_function62, initial_state, final_states1)
Rule_86 = Automata(states1, alphabet, transition_function86, initial_state, final_states1)
Rule_147 = Automata(states1, alphabet, transition_function147, initial_state, final_states1)
Rule_4 =  Automata(states1, alphabet, transition_function4, initial_state, final_states1)
Rule_102 =  Automata(states1, alphabet, transition_function102, initial_state, final_states1)
Rule_45 =  Automata(states1, alphabet, transition_function45, initial_state, final_states1)
#2d rules
Conways_Game =  Automata(states1, alphabet, conways_game, initial_state, final_states1)
