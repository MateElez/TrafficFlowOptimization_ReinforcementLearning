import numpy as np
import traci
from typing import List, Dict, Tuple
from collections import deque
import random
from ..utils.sumo_utils import get_waiting_vehicles, get_controlled_lanes

class TrafficLightQLearning:
    def __init__(self, tl_id: str, phases: List[int], controlled_lanes: List[str],
                 alpha: float = 0.1,  # Optimalna vrijednost iz grid searcha
                 gamma: float = 0.9,  # Optimalna vrijednost iz grid searcha
                 epsilon: float = 0.2,  # Optimalna vrijednost iz grid searcha
                 min_epsilon: float = 0.01,
                 epsilon_decay: float = 0.995):  # Optimalna vrijednost iz grid searcha
        """
        Inicijalizacija Q-learning agenta za semafor.
        
        Args:
            tl_id: ID semafora
            phases: Lista mogućih faza
            controlled_lanes: Lista kontroliranih traka
            alpha: Stopa učenja (default: 0.1)
            gamma: Faktor diskontiranja (default: 0.9)
            epsilon: Vjerojatnost istraživanja (default: 0.2)
            min_epsilon: Minimalna vjerojatnost istraživanja
            epsilon_decay: Smanjenje epsilon-a (default: 0.995)
            experience_size: Veličina spremnika iskustava (povećana na 2000)
            batch_size: Veličina serije za učenje (povećana na 64)
        """
        self.tl_id = tl_id
        self.phases = phases
        self.controlled_lanes = controlled_lanes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.experience_size = 2000
        self.batch_size = 64
        
        # Q-tablica
        self.q_table = {}
        
        # Spremnik iskustava za experience replay
        self.experience = deque(maxlen=self.experience_size)
        
        # Brojač koraka od zadnje promjene faze
        self.steps_since_last_change = 0
        
        # Temperatura za Boltzmann strategiju
        self.temperature = 1.0
        self.min_temperature = 0.1
        self.temperature_decay = 0.995
        
        # Provjeri ima li semafor prilaze
        self.controlled_lanes = get_controlled_lanes(tl_id)
        if not self.controlled_lanes:
            print(f"Upozorenje: Semafor {tl_id} nema prilaze!")
        
    def get_state(self) -> Tuple:
        """
        Dohvaća trenutno stanje semafora.
        Stanje uključuje:
        - Broj vozila koja čekaju na svakoj traci
        - Prosječno vrijeme čekanja na svakoj traci
        - Duljina reda na svakoj traci
        - Brzina vozila na svakoj traci
        - Vrijeme od zadnje promjene faze
        """
        state = []
        
        for lane in self.controlled_lanes:
            # Dohvaćanje vozila na traci
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            
            # Broj vozila koja čekaju
            waiting = len([v for v in vehicles if traci.vehicle.getSpeed(v) < 0.1])
            state.append(waiting)
            
            # Prosječno vrijeme čekanja
            waiting_time = sum(traci.vehicle.getWaitingTime(v) for v in vehicles) / max(len(vehicles), 1)
            state.append(waiting_time)
            
            # Duljina reda
            queue_length = traci.lane.getLastStepHaltingNumber(lane)
            state.append(queue_length)
            
            # Prosječna brzina
            speed = traci.lane.getLastStepMeanSpeed(lane)
            state.append(speed)
        
        # Vrijeme od zadnje promjene faze
        state.append(self.steps_since_last_change)
        
        # Diskretizacija stanja
        state = tuple(int(x) for x in state)
        return state
    
    def get_reward(self) -> float:
        """
        Računa nagradu za trenutno stanje.
        Nagrada uključuje:
        - Kažnjavanje za čekanje vozila
        - Nagradu za propusnost
        - Kažnjavanje za česte promjene faze
        - Kažnjavanje za dugo čekanje
        """
        reward = 0.0
        
        # Kažnjavanje za čekanje
        for lane in self.controlled_lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            waiting = len([v for v in vehicles if traci.vehicle.getSpeed(v) < 0.1])
            reward -= waiting * 0.1  # Kažnjavanje po vozilu
            
            # Kažnjavanje za dugo čekanje
            for v in vehicles:
                waiting_time = traci.vehicle.getWaitingTime(v)
                if waiting_time > 30:  # Ako vozilo čeka više od 30 sekundi
                    reward -= waiting_time * 0.01
        
        # Nagrada za propusnost
        for lane in self.controlled_lanes:
            departed = traci.lane.getLastStepVehicleNumber(lane)
            reward += departed * 0.2  # Nagrada za svako vozilo koje prođe
        
        # Kažnjavanje za česte promjene faze
        if self.steps_since_last_change < 10:  # Ako je faza promijenjena u zadnjih 10 koraka
            reward -= 0.5
        
        return reward
    
    def choose_action(self, state: Tuple) -> int:
        """
        Odabire akciju na temelju trenutnog stanja.
        Koristi kombinaciju epsilon-greedy i Boltzmann strategije.
        """
        # Smanjivanje epsilon-a
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        # Smanjivanje temperature
        self.temperature = max(self.min_temperature, self.temperature * self.temperature_decay)
        
        if np.random.random() < self.epsilon:
            # Nasumična akcija (istraživanje)
            return np.random.randint(len(self.phases))
        else:
            # Boltzmann strategija
            q_values = [self.q_table.get((state, a), 0) for a in range(len(self.phases))]
            exp_q = np.exp(np.array(q_values) / self.temperature)
            probs = exp_q / exp_q.sum()
            return np.random.choice(len(self.phases), p=probs)
    
    def update_q_table(self, state: Tuple, action: int, reward: float, new_state: Tuple):
        """
        Ažurira Q-tablicu koristeći experience replay.
        """
        # Dodavanje iskustva u spremnik
        self.experience.append((state, action, reward, new_state))
        
        # Ažuriranje Q-tablice ako imamo dovoljno iskustava
        if len(self.experience) >= self.batch_size:
            # Odabir serije iskustava
            batch = random.sample(self.experience, self.batch_size)
            
            for s, a, r, s_new in batch:
                # Q-learning formula
                old_value = self.q_table.get((s, a), 0)
                next_max = max([self.q_table.get((s_new, a_new), 0) 
                              for a_new in range(len(self.phases))])
                new_value = old_value + self.alpha * (r + self.gamma * next_max - old_value)
                self.q_table[(s, a)] = new_value
        
        # Ažuriranje brojača koraka
        self.steps_since_last_change += 1 