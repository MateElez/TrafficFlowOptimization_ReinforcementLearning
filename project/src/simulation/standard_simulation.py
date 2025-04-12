import traci
import xml.etree.ElementTree as ET
from typing import Dict, List
from ..utils.sumo_utils import (
    initialize_simulation,
    load_trips,
    get_traffic_lights,
    get_vehicle_data,
    close_simulation
)

class SimulationStats:
    def __init__(self):
        self.waiting_times = []
        self.queue_lengths = []
        self.vehicle_speeds = []
        self.vehicle_counts = []
        self.stops_count = []

def run_standard_simulation(net_file: str, trips_file: str, steps: int = 1000) -> SimulationStats:
    """
    Pokreće standardnu simulaciju bez RL-a.
    
    Args:
        net_file: Putanja do SUMO mrežne datoteke
        trips_file: Putanja do datoteke s rutama vozila
        steps: Broj koraka simulacije
    
    Returns:
        SimulationStats objekt s prikupljenim statistikama
    """
    # Inicijalizacija simulacije
    traci = initialize_simulation(net_file, trips_file)
    
    # Učitavanje ruta
    num_vehicles = load_trips(trips_file)
    print(f"Učitano {num_vehicles} vozila iz {trips_file}")
    
    # Inicijalizacija statistike
    stats = SimulationStats()
    
    # Glavna petlja simulacije
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
            # Vrijeme čekanja
            current_stats['waiting_time'] += data.get('waiting_time', 0)
            
            # Duljina reda (ako je vozilo zaustavljeno)
            if data.get('speed', 0) < 0.1:
                current_stats['queue_length'] += 1
            
            # Brzina
            current_stats['speed'] += data.get('speed', 0)
            
            # Broj zaustavljanja
            current_stats['stops'] += data.get('stops', 0)
        
        # Ažuriranje statistike
        if vehicle_data:
            stats.waiting_times.append(current_stats['waiting_time'] / len(vehicle_data))
            stats.queue_lengths.append(current_stats['queue_length'])
            stats.vehicle_speeds.append(current_stats['speed'] / len(vehicle_data))
            stats.vehicle_counts.append(len(vehicle_data))
            stats.stops_count.append(current_stats['stops'])
        
        # Napredovanje simulacije
        traci.simulationStep()
        
        # Ispisivanje napretka
        if (step + 1) % 100 == 0:
            print(f"Korak {step + 1}/{steps}, "
                  f"Broj vozila: {len(vehicle_data)}, "
                  f"Prosječno vrijeme čekanja: {stats.waiting_times[-1]:.2f}s")
    
    # Zatvaranje simulacije
    close_simulation()
    
    return stats 