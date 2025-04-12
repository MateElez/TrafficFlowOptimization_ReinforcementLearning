import numpy as np
from typing import Dict, List, Tuple
from ..simulation.qlearning import TrafficLightQLearning
from .sumo_utils import (
    initialize_simulation,
    load_trips,
    get_traffic_lights,
    get_traffic_light_phases,
    get_controlled_lanes,
    get_vehicle_data,
    save_network_state,
    load_network_state,
    close_simulation
)

def run_simulation_with_params(net_file: str, trips_file: str, 
                             alpha: float, gamma: float, epsilon: float,
                             epsilon_decay: float, episodes: int = 10, 
                             steps: int = 100) -> Tuple[float, Dict[str, float]]:
    """
    Pokreće simulaciju s zadanim parametrima i vraća prosječnu nagradu i statistiku.
    """
    # Inicijalizacija SUMO simulacije
    traci = initialize_simulation(net_file, trips_file)
    
    # Učitavanje ruta vozila
    num_vehicles = load_trips(trips_file)
    print(f"Učitano {num_vehicles} vozila iz {trips_file}")
    
    # Spremanje početnog stanja
    save_network_state("Input/initial_state.xml")
    
    # Inicijalizacija agenata za semafore
    traffic_lights = get_traffic_lights()
    agents = {}
    
    for tl_id in traffic_lights:
        phases = get_traffic_light_phases(tl_id)
        controlled_lanes = get_controlled_lanes(tl_id)
        
        if not controlled_lanes:
            continue
            
        agents[tl_id] = TrafficLightQLearning(
            tl_id=tl_id,
            phases=phases,
            controlled_lanes=controlled_lanes,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay
        )
    
    # Inicijalizacija statistike
    total_rewards = []
    stats = {
        'waiting_times': [],
        'queue_lengths': [],
        'speeds': [],
        'vehicles': []
    }
    
    # Glavna petlja učenja
    for episode in range(episodes):
        # Resetiranje simulacije
        load_network_state("Input/initial_state.xml")
        
        # Inicijalizacija stanja za epizodu
        states = {tl_id: agent.get_state() for tl_id, agent in agents.items()}
        episode_reward = 0
        
        for step in range(steps):
            # Prikupljanje podataka o vozilima
            vehicle_data = get_vehicle_data()
            
            # Ažuriranje statistike
            if vehicle_data:
                waiting_time = sum(data.get('waiting_time', 0) for data in vehicle_data.values()) / len(vehicle_data)
                queue_length = sum(1 for data in vehicle_data.values() if data.get('speed', 0) < 0.1)
                avg_speed = sum(data.get('speed', 0) for data in vehicle_data.values()) / len(vehicle_data)
                
                stats['waiting_times'].append(waiting_time)
                stats['queue_lengths'].append(queue_length)
                stats['speeds'].append(avg_speed)
                stats['vehicles'].append(len(vehicle_data))
            
            # Ažuriranje Q-tablice za svaki semafor
            for tl_id, agent in agents.items():
                # Odabir akcije
                action = agent.choose_action(states[tl_id])
                
                # Izvršavanje akcije
                traci.trafficlight.setPhase(tl_id, action)
                
                # Dobivanje novog stanja i nagrade
                new_state = agent.get_state()
                reward = agent.get_reward()
                
                # Ažuriranje Q-tablice
                agent.update_q_table(states[tl_id], action, reward, new_state)
                
                # Ažuriranje stanja
                states[tl_id] = new_state
                episode_reward += reward
            
            # Napredovanje simulacije
            traci.simulationStep()
        
        total_rewards.append(episode_reward)
        
        # Ispisivanje napretka
        if (episode + 1) % 5 == 0:
            print(f"Epizoda {episode + 1}/{episodes}, "
                  f"Prosječna nagrada: {np.mean(total_rewards[-5:]):.2f}, "
                  f"Prosječno vrijeme čekanja: {np.mean(stats['waiting_times'][-steps:]):.2f}s")
    
    # Zatvaranje simulacije
    close_simulation()
    
    # Računanje prosječnih vrijednosti
    avg_stats = {
        'waiting_time': np.mean(stats['waiting_times']),
        'queue_length': np.mean(stats['queue_lengths']),
        'speed': np.mean(stats['speeds']),
        'vehicles': np.mean(stats['vehicles'])
    }
    
    return np.mean(total_rewards), avg_stats

def grid_search(net_file: str, trips_file: str) -> Dict[str, float]:
    """
    Izvodi grid search za pronalaženje optimalnih parametara.
    """
    # Definicija grid-a parametara
    alphas = [0.1]  # Samo jedna vrijednost za alpha
    gammas = [0.9, 0.95]  # Dvije vrijednosti za gamma
    epsilons = [0.1, 0.2]  # Dvije vrijednosti za epsilon
    epsilon_decays = [0.995]  # Samo jedna vrijednost za epsilon_decay
    
    best_params = None
    best_reward = float('-inf')
    best_stats = None
    
    # Grid search
    for alpha in alphas:
        for gamma in gammas:
            for epsilon in epsilons:
                for epsilon_decay in epsilon_decays:
                    print(f"\nTestiranje parametara: "
                          f"alpha={alpha}, gamma={gamma}, "
                          f"epsilon={epsilon}, epsilon_decay={epsilon_decay}")
                    
                    try:
                        avg_reward, stats = run_simulation_with_params(
                            net_file, trips_file,
                            alpha, gamma, epsilon, epsilon_decay,
                            episodes=3,  # Smanjen broj epizoda
                            steps=100
                        )
                        
                        print(f"Prosječna nagrada: {avg_reward:.2f}")
                        print(f"Prosječno vrijeme čekanja: {stats['waiting_time']:.2f}s")
                        print(f"Prosječna duljina reda: {stats['queue_length']:.2f}")
                        print(f"Prosječna brzina: {stats['speed']:.2f}m/s")
                        print(f"Prosječan broj vozila: {stats['vehicles']:.2f}")
                        
                        if avg_reward > best_reward:
                            best_reward = avg_reward
                            best_stats = stats
                            best_params = {
                                'alpha': alpha,
                                'gamma': gamma,
                                'epsilon': epsilon,
                                'epsilon_decay': epsilon_decay,
                                'reward': avg_reward,
                                **stats
                            }
                            print("Novi najbolji rezultat!")
                            
                    except Exception as e:
                        print(f"Greška pri testiranju parametara: {e}")
                        continue
    
    return best_params

def main():
    # Pokretanje grid search-a
    best_params = grid_search(
        net_file="Input/osm.net.xml",
        trips_file="Input/osm.passenger.trips.xml"
    )
    
    # Ispis najboljih parametara
    print("\nNajbolji parametri:")
    for param, value in best_params.items():
        print(f"{param}: {value}")

if __name__ == "__main__":
    main() 