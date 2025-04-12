import traci
import xml.etree.ElementTree as ET
import os
from typing import List, Dict, Tuple

def initialize_simulation(net_file: str, trips_file: str) -> None:
    """Inicijalizira SUMO simulaciju s datim mrežom i rutama"""
    sumo_cmd = ["sumo", "-n", net_file, "--route-files", trips_file, "--quit-on-end", "--ignore-route-errors", "--no-warnings"]
    traci.start(sumo_cmd)
    return traci

def load_trips(trips_file: str) -> int:
    """Učitava rute iz trips datoteke i vraća broj vozila"""
    tree = ET.parse(trips_file)
    root = tree.getroot()
    return len(root.findall('.//trip'))

def get_traffic_lights() -> List[str]:
    """Dohvaća listu ID-ova svih semafora u mreži"""
    return traci.trafficlight.getIDList()

def get_traffic_light_phases(tl_id: str) -> List[str]:
    """Dohvaća sve faze za zadani semafor"""
    return traci.trafficlight.getAllProgramLogics(tl_id)[0].phases

def save_network_state(filename: str = "initial_state.xml") -> None:
    """Sprema početno stanje mreže"""
    traci.simulation.saveState(filename)

def load_network_state(filename: str = "initial_state.xml") -> None:
    """Učitava početno stanje mreže"""
    traci.simulation.loadState(filename)

def get_vehicle_count() -> int:
    """Dohvaća broj aktivnih vozila u simulaciji"""
    return traci.vehicle.getIDCount()

def get_waiting_vehicles(lane_id: str) -> int:
    """Dohvaća broj vozila koja čekaju na zadanoj traci"""
    return traci.lane.getLastStepHaltingNumber(lane_id)

def get_controlled_lanes(tl_id: str) -> List[str]:
    """Dohvaća listu traka koje kontrolira zadani semafor"""
    return traci.trafficlight.getControlledLanes(tl_id)

def set_traffic_light_phase(tl_id: str, phase: int) -> None:
    """Postavlja fazu semafora"""
    traci.trafficlight.setPhase(tl_id, phase)

def simulation_step() -> None:
    """Napravi jedan korak simulacije"""
    traci.simulationStep()

def close_simulation() -> None:
    """Zatvori SUMO simulaciju"""
    traci.close()

def get_vehicle_data() -> Dict[str, Dict[str, float]]:
    """
    Dohvaća podatke o svim vozilima u simulaciji.
    
    Returns:
        Rječnik s ID-ovima vozila kao ključevima i njihovim podacima kao vrijednostima.
        Podaci uključuju: vrijeme čekanja, brzinu i broj zaustavljanja.
    """
    vehicle_data = {}
    for veh_id in traci.vehicle.getIDList():
        vehicle_data[veh_id] = {
            'waiting_time': traci.vehicle.getWaitingTime(veh_id),
            'speed': traci.vehicle.getSpeed(veh_id),
            'stops': traci.vehicle.getStopState(veh_id)
        }
    return vehicle_data 