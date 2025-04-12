import xml.etree.ElementTree as ET
import traci
import csv
import os
from collections import defaultdict

def generate_full_mapping(net_file, osm_file, output_csv="street_mapping.csv"):
    """Generira potpunu mapu edge-ova prema imenima ulica"""
    # Prvo učitaj imena i tipove iz SUMO mreže
    print("Učitavanje SUMO mreže...")
    sumo_names = {}
    edge_types = {}
    tree = ET.parse(net_file)
    for edge in tree.findall('edge'):
        edge_id = edge.get('id')
        if edge_id and not edge_id.startswith(':'):  # Preskoči interne edge-ove
            name = edge.get('name', '')
            if name:
                sumo_names[edge_id] = name
            edge_type = edge.get('type', '')
            if edge_type:
                edge_types[edge_id] = edge_type
    
    print(f"Pronađeno {len(sumo_names)} imenovanih edge-ova u SUMO mreži")
    if sumo_names:
        print(f"Primjer SUMO edge ID-a i imena: {next(iter(sumo_names.items()))}")
    
    # Zatim učitaj imena iz OSM-a kao backup
    print("\nUčitavanje OSM podataka...")
    osm_data = {}
    tree = ET.parse(osm_file)
    count_ways = 0
    count_named_ways = 0
    
    for way in tree.findall('way'):
        if way.find('tag/[@k="highway"]') is not None:
            count_ways += 1
            name_tag = way.find('tag/[@k="name"]')
            if name_tag is not None:
                count_named_ways += 1
                osm_data[way.get('id')] = name_tag.get('v')
    
    print(f"Pronađeno {count_ways} cesta u OSM podacima")
    print(f"Od toga {count_named_ways} cesta ima imena")
    
    traci.start(["sumo", "-n", net_file, "--quit-on-end"])
    
    # Debug ispis za SUMO edge ID-ove
    edge_ids = traci.edge.getIDList()
    non_internal_edges = [e for e in edge_ids if not e.startswith(':')]
    print(f"\nPronađeno {len(edge_ids)} edge-ova u SUMO mreži")
    print(f"Od toga {len(non_internal_edges)} nisu interni edge-ovi")
    
    # Analiza nepoznatih edge-ova po tipu
    unknown_edges_by_type = defaultdict(int)
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['edge_id', 'street_name', 'source', 'edge_type'])
        
        unknown_count = 0
        total_count = 0
        sumo_named_count = 0
        osm_named_count = 0
        
        for edge_id in edge_ids:
            if edge_id.startswith(':'):  # Preskoči interne edge-ove
                continue
                
            total_count += 1
            
            # Prvo pokušaj naći ime iz SUMO mreže
            street_name = sumo_names.get(edge_id, '')
            source = 'SUMO'
            
            if not street_name:
                # Ako nema imena u SUMO mreži, pokušaj naći u OSM podacima
                osm_id = edge_id.lstrip('-').split('#')[0].split('_')[0]
                if osm_id.isdigit():
                    street_name = osm_data.get(osm_id, '')
                    source = 'OSM'
            
            if not street_name:
                street_name = "Nepoznato"
                source = '-'
                unknown_count += 1
                # Zabilježi tip nepoznatog edge-a
                edge_type = edge_types.get(edge_id, 'unknown')
                unknown_edges_by_type[edge_type] += 1
            elif source == 'SUMO':
                sumo_named_count += 1
            else:
                osm_named_count += 1
            
            writer.writerow([
                edge_id,
                street_name,
                source,
                edge_types.get(edge_id, 'unknown')
            ])
    
    print(f"\nStatistika:")
    print(f"Ukupno edge-ova: {len(edge_ids)}")
    print(f"Ne-internih edge-ova: {total_count}")
    print(f"Edge-ova s imenom iz SUMO: {sumo_named_count}")
    print(f"Edge-ova s imenom iz OSM: {osm_named_count}")
    print(f"Nepoznatih imena: {unknown_count} ({(unknown_count/total_count)*100:.1f}%)")
    
    print("\nAnaliza nepoznatih edge-ova po tipu:")
    for edge_type, count in sorted(unknown_edges_by_type.items(), key=lambda x: x[1], reverse=True):
        print(f"{edge_type}: {count} ({count/unknown_count*100:.1f}%)")
    
    traci.close() 