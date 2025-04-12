import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from ..simulation.standard_simulation import run_standard_simulation, SimulationStats
from ..simulation.qlearning import TrafficLightQLearning
from .sumo_utils import (
    initialize_simulation,
    load_trips,
    get_traffic_lights,
    get_traffic_light_phases,
    get_controlled_lanes,
    get_waiting_vehicles,
    get_vehicle_data,
    save_network_state,
    close_simulation,
    load_network_state
)

def run_simulation(simulation_type: str, net_file: str, trips_file: str, 
                  episodes: int = 10, steps: int = 100,
                  qlearning_params: dict = None) -> SimulationStats:
    """
    Pokreće simulaciju odabranog tipa.
    
    Args:
        simulation_type: Tip simulacije ('standard', 'qlearning')
        net_file: Putanja do SUMO mrežne datoteke
        trips_file: Putanja do datoteke s rutama vozila
        episodes: Broj epizoda (za RL simulacije)
        steps: Broj koraka po epizodi
        qlearning_params: Parametri za Q-learning (ako je simulation_type='qlearning')
    
    Returns:
        SimulationStats objekt s prikupljenim statistikama
    """
    if simulation_type == 'standard':
        return run_standard_simulation(net_file, trips_file, steps)
    elif simulation_type == 'qlearning':
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
                print(f"Upozorenje: Semafor {tl_id} nema kontroliranih traka")
                continue
                
            # Koristi optimalne parametre ako su dostupni
            if qlearning_params:
                agents[tl_id] = TrafficLightQLearning(
                    tl_id=tl_id,
                    phases=phases,
                    controlled_lanes=controlled_lanes,
                    **qlearning_params
                )
            else:
                agents[tl_id] = TrafficLightQLearning(
                    tl_id=tl_id,
                    phases=phases,
                    controlled_lanes=controlled_lanes
                )
            print(f"Agent inicijaliziran za semafor {tl_id} s {len(phases)} faza")
        
        # Inicijalizacija statistike
        stats = SimulationStats()
        
        # Glavna petlja učenja
        for episode in range(episodes):
            print(f"\nEpizoda {episode + 1}/{episodes}")
            
            # Resetiranje simulacije
            load_network_state("Input/initial_state.xml")
            
            # Inicijalizacija stanja za epizodu
            states = {tl_id: agent.get_state() for tl_id, agent in agents.items()}
            total_reward = 0
            
            for step in range(steps):
                # Prikupljanje podataka o vozilima
                vehicle_data = get_vehicle_data()
                
                # Prikupljanje statistike
                current_stats = {
                    'waiting_time': 0,
                    'queue_length': 0,
                    'speed': 0,
                    'stops': 0
                }
                
                for vehicle_id, data in vehicle_data.items():
                    current_stats['waiting_time'] += data.get('waiting_time', 0)
                    if data.get('speed', 0) < 0.1:
                        current_stats['queue_length'] += 1
                    current_stats['speed'] += data.get('speed', 0)
                    current_stats['stops'] += data.get('stops', 0)
                
                if vehicle_data:
                    stats.waiting_times.append(current_stats['waiting_time'] / len(vehicle_data))
                    stats.queue_lengths.append(current_stats['queue_length'])
                    stats.vehicle_speeds.append(current_stats['speed'] / len(vehicle_data))
                    stats.vehicle_counts.append(len(vehicle_data))
                    stats.stops_count.append(current_stats['stops'])
                
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
                    total_reward += reward
                
                # Napredovanje simulacije
                traci.simulationStep()
                
                # Ispisivanje napretka
                if (step + 1) % 10 == 0:  # Ispis svakih 10 koraka
                    print(f"Korak {step + 1}/{steps}, "
                          f"Broj vozila: {len(vehicle_data)}, "
                          f"Ukupna nagrada: {total_reward:.2f}")
            
            # Ispisivanje statistike za epizodu
            print(f"Epizoda {episode + 1} završena. "
                  f"Ukupna nagrada: {total_reward:.2f}, "
                  f"Broj vozila: {len(vehicle_data)}")
        
        # Zatvaranje simulacije
        close_simulation()
        
        return stats
    else:
        raise ValueError(f"Nepoznat tip simulacije: {simulation_type}")

def compare_simulations(simulation_types: List[str], net_file: str, trips_file: str,
                       episodes: int = 10, steps: int = 100, qlearning_params: dict = None) -> Dict[str, Dict[str, float]]:
    """
    Uspoređuje različite tipove simulacija.
    
    Args:
        simulation_types: Lista tipova simulacija za usporedbu
        net_file: Putanja do SUMO mrežne datoteke
        trips_file: Putanja do datoteke s rutama vozila
        episodes: Broj epizoda (za RL simulacije)
        steps: Broj koraka po epizodi
        qlearning_params: Optimalni parametri iz grid searcha
    
    Returns:
        Rječnik s usporednim statistikama
    """
    comparison = {}
    
    for sim_type in simulation_types:
        print(f"\nPokretanje {sim_type} simulacije...")
        if sim_type == 'qlearning' and qlearning_params:
            stats = run_simulation(sim_type, net_file, trips_file, episodes, steps, qlearning_params)
        else:
            stats = run_simulation(sim_type, net_file, trips_file, episodes, steps)
        
        comparison[sim_type] = {
            'avg_waiting_time': np.mean(stats.waiting_times),
            'avg_queue_length': np.mean(stats.queue_lengths),
            'avg_speed': np.mean(stats.vehicle_speeds),
            'avg_vehicles': np.mean(stats.vehicle_counts),
            'total_stops': np.sum(stats.stops_count)
        }
    
    return comparison

def plot_comparison(comparison: Dict[str, Dict[str, float]]):
    """
    Crtanje grafikona za usporedbu simulacija.
    """
    metrics = ['avg_waiting_time', 'avg_queue_length', 'avg_speed', 'avg_vehicles', 'total_stops']
    labels = ['Prosječno vrijeme čekanja', 'Prosječna duljina reda', 
              'Prosječna brzina', 'Prosječan broj vozila', 'Ukupan broj zaustavljanja']
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, [comparison['standard'][m] for m in metrics], width, label='Standardna')
    rects2 = ax.bar(x + width/2, [comparison['qlearning'][m] for m in metrics], width, label='Q-learning')
    
    ax.set_ylabel('Vrijednost')
    ax.set_title('Usporedba standardne i Q-learning simulacije')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    
    fig.tight_layout()
    plt.savefig('simulation_comparison.png')
    plt.close()

def main():
    # Usporedba standardne i Q-learning simulacije s optimalnim parametrima
    comparison = compare_simulations(
        simulation_types=['standard', 'qlearning'],
        net_file="Input/osm.net.xml",
        trips_file="Input/osm.passenger.trips.xml",
        episodes=50,    # Povećan broj epizoda za bolju usporedbu
        steps=500,      # Povećan broj koraka
        qlearning_params={  # Optimalni parametri iz grid searcha
            'alpha': 0.1,
            'gamma': 0.9,
            'epsilon': 0.2,
            'epsilon_decay': 0.995
        }
    )
    
    # Ispis rezultata
    print("\nUsporedba simulacija s optimalnim parametrima:")
    for sim_type, stats in comparison.items():
        print(f"\n{sim_type} simulacija:")
        for metric, value in stats.items():
            print(f"{metric}: {value:.2f}")
    
    # Crtanje grafikona
    plot_comparison(comparison)
    print("\nGrafikon usporedbe spremljen u 'simulation_comparison.png'")

if __name__ == "__main__":
    main() 