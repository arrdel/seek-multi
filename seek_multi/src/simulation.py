"""
SEEK-Multi Simulation Environment

Provides simulation capabilities for testing multi-agent
collaborative inspection.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import time
from collections import defaultdict

from .shared_dsg import SharedDynamicSceneGraph, NodeType, DSGNode
from .distributed_rsn import DistributedRSN
from .agent import InspectionAgent
from .coordinator import MultiAgentCoordinator, CoordinationMode


@dataclass
class SimulatedObject:
    """A simulated object in the environment."""
    object_id: str
    object_class: str
    position: np.ndarray
    room_id: str
    visible: bool = True


@dataclass
class SimulatedDetection:
    """A simulated object detection."""
    object_class: str
    position: np.ndarray
    confidence: float
    is_true_positive: bool


class SimulationEnvironment:
    """
    Simulated environment for multi-agent inspection testing.
    
    Features:
    - Configurable room layouts
    - Object placement
    - Simulated sensors with noise
    - Agent movement simulation
    """
    
    def __init__(
        self,
        detection_range: float = 5.0,
        true_positive_rate: float = 0.9,
        false_positive_rate: float = 0.05,
        movement_speed: float = 1.0
    ):
        """
        Initialize simulation environment.
        
        Args:
            detection_range: Range at which objects can be detected
            true_positive_rate: Probability of detecting present object
            false_positive_rate: Probability of false detection
            movement_speed: Agent movement speed (m/s)
        """
        self.detection_range = detection_range
        self.true_positive_rate = true_positive_rate
        self.false_positive_rate = false_positive_rate
        self.movement_speed = movement_speed
        
        # Environment state
        self.objects: Dict[str, SimulatedObject] = {}
        self.rooms: Dict[str, Dict] = {}
        self.dsg: Optional[SharedDynamicSceneGraph] = None
        
        # Agent state
        self.agent_positions: Dict[str, np.ndarray] = {}
        self.agent_paths: Dict[str, List[np.ndarray]] = defaultdict(list)
        
        # Simulation state
        self.sim_time = 0.0
        self.time_step = 0.1
        
        # Metrics
        self.total_detections = 0
        self.true_positives = 0
        self.false_positives = 0
    
    def load_environment(
        self,
        rooms: List[Dict],
        connections: List[Tuple[str, str, float]],
        objects: List[Dict]
    ) -> SharedDynamicSceneGraph:
        """
        Load environment configuration.
        
        Args:
            rooms: Room definitions
            connections: Room connections
            objects: Object placements
            
        Returns:
            Generated DSG
        """
        # Store room info
        for room in rooms:
            self.rooms[room["id"]] = room
        
        # Create DSG
        self.dsg = SharedDynamicSceneGraph.from_blueprint(
            agent_id="simulation",
            rooms=rooms,
            connections=connections
        )
        
        # Add objects
        for obj in objects:
            sim_obj = SimulatedObject(
                object_id=obj["id"],
                object_class=obj["class"],
                position=np.array(obj["position"]),
                room_id=obj["room_id"]
            )
            self.objects[obj["id"]] = sim_obj
            
            # Add to DSG
            self.dsg.add_node(
                node_id=obj["id"],
                node_type=NodeType.OBJECT,
                position=sim_obj.position,
                semantic_class=obj["class"],
                parent_id=obj["room_id"]
            )
        
        return self.dsg
    
    def place_agent(
        self,
        agent_id: str,
        position: np.ndarray
    ) -> None:
        """Place an agent at a position."""
        self.agent_positions[agent_id] = position.copy()
        self.agent_paths[agent_id] = [position.copy()]
    
    def move_agent(
        self,
        agent_id: str,
        target_position: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        """
        Move agent toward target position.
        
        Args:
            agent_id: Agent identifier
            target_position: Target position
            
        Returns:
            Tuple of (new_position, reached_target)
        """
        if agent_id not in self.agent_positions:
            return target_position, True
        
        current = self.agent_positions[agent_id]
        direction = target_position - current
        distance = np.linalg.norm(direction)
        
        if distance < 0.1:
            return current, True
        
        # Move toward target
        step_size = min(self.movement_speed * self.time_step, distance)
        new_position = current + (direction / distance) * step_size
        
        self.agent_positions[agent_id] = new_position
        self.agent_paths[agent_id].append(new_position.copy())
        
        reached = np.linalg.norm(new_position - target_position) < 0.1
        return new_position, reached
    
    def simulate_detection(
        self,
        agent_id: str,
        target_class: str,
        thorough_search: bool = False
    ) -> List[SimulatedDetection]:
        """
        Simulate object detection from agent's current position.
        
        Args:
            agent_id: Agent identifier
            target_class: Object class to detect
            thorough_search: Whether doing thorough room search
            
        Returns:
            List of detections
        """
        if agent_id not in self.agent_positions:
            return []
        
        agent_pos = self.agent_positions[agent_id]
        detections = []
        
        # Check each object
        for obj_id, obj in self.objects.items():
            distance = np.linalg.norm(agent_pos - obj.position)
            
            # Check if in range
            effective_range = self.detection_range * (2.0 if thorough_search else 1.0)
            if distance > effective_range:
                continue
            
            # Check if target class
            if obj.object_class != target_class:
                continue
            
            # Simulate detection probability
            range_factor = 1.0 - (distance / effective_range)
            detection_prob = self.true_positive_rate * range_factor
            
            if np.random.random() < detection_prob:
                # True positive
                # Add noise to position
                noise = np.random.randn(3) * 0.5
                detected_pos = obj.position + noise
                
                confidence = min(0.95, detection_prob + np.random.random() * 0.1)
                
                detections.append(SimulatedDetection(
                    object_class=target_class,
                    position=detected_pos,
                    confidence=confidence,
                    is_true_positive=True
                ))
                
                self.total_detections += 1
                self.true_positives += 1
        
        # Simulate false positives
        if np.random.random() < self.false_positive_rate:
            # Random false detection
            false_pos = agent_pos + np.random.randn(3) * 3
            
            detections.append(SimulatedDetection(
                object_class=target_class,
                position=false_pos,
                confidence=np.random.random() * 0.5,
                is_true_positive=False
            ))
            
            self.total_detections += 1
            self.false_positives += 1
        
        return detections
    
    def get_agent_room(self, agent_id: str) -> Optional[str]:
        """Get the room containing an agent."""
        if agent_id not in self.agent_positions:
            return None
        
        pos = self.agent_positions[agent_id]
        
        # Find closest room
        min_dist = float('inf')
        closest_room = None
        
        for room_id, room in self.rooms.items():
            room_pos = np.array(room["position"])
            dist = np.linalg.norm(pos[:2] - room_pos[:2])
            if dist < min_dist:
                min_dist = dist
                closest_room = room_id
        
        return closest_room
    
    def compute_optimal_distance(
        self,
        start_positions: Dict[str, np.ndarray],
        target_class: str
    ) -> float:
        """
        Compute optimal (shortest) distance to find target.
        
        Returns the minimum distance any single agent would need
        to travel to find the nearest target object.
        """
        # Find all objects of target class
        targets = [
            obj for obj in self.objects.values()
            if obj.object_class == target_class
        ]
        
        if not targets:
            return float('inf')
        
        min_distance = float('inf')
        
        for agent_id, start_pos in start_positions.items():
            for target in targets:
                dist = np.linalg.norm(start_pos - target.position)
                min_distance = min(min_distance, dist)
        
        return min_distance
    
    def get_agent_total_distance(self, agent_id: str) -> float:
        """Get total distance traveled by an agent."""
        if agent_id not in self.agent_paths:
            return 0.0
        
        path = self.agent_paths[agent_id]
        if len(path) < 2:
            return 0.0
        
        total = 0.0
        for i in range(1, len(path)):
            total += np.linalg.norm(path[i] - path[i-1])
        
        return total
    
    def step(self, dt: float = None) -> None:
        """Advance simulation time."""
        if dt is None:
            dt = self.time_step
        self.sim_time += dt
    
    def reset(self) -> None:
        """Reset simulation state."""
        self.agent_positions.clear()
        self.agent_paths.clear()
        self.sim_time = 0.0
        self.total_detections = 0
        self.true_positives = 0
        self.false_positives = 0


def create_office_environment() -> Tuple[List[Dict], List[Tuple], List[Dict]]:
    """
    Create a sample office environment configuration.
    
    Returns:
        Tuple of (rooms, connections, objects)
    """
    rooms = [
        {"id": "entrance", "name": "entrance", "position": [0, 0, 0], "num_locations": 4},
        {"id": "lobby", "name": "lobby", "position": [5, 0, 0], "num_locations": 6},
        {"id": "hallway_1", "name": "hallway", "position": [10, 0, 0], "num_locations": 4},
        {"id": "office_1", "name": "office", "position": [10, 5, 0], "num_locations": 8},
        {"id": "office_2", "name": "office", "position": [10, -5, 0], "num_locations": 8},
        {"id": "conference", "name": "conference_room", "position": [15, 0, 0], "num_locations": 10},
        {"id": "kitchen", "name": "kitchen", "position": [15, 5, 0], "num_locations": 6},
        {"id": "break_room", "name": "break_room", "position": [15, -5, 0], "num_locations": 6},
        {"id": "hallway_2", "name": "hallway", "position": [20, 0, 0], "num_locations": 4},
        {"id": "restroom", "name": "restroom", "position": [20, 5, 0], "num_locations": 4},
        {"id": "storage", "name": "storage", "position": [20, -5, 0], "num_locations": 4},
        {"id": "server_room", "name": "server_room", "position": [25, 0, 0], "num_locations": 4},
    ]
    
    connections = [
        ("entrance", "lobby", 5.0),
        ("lobby", "hallway_1", 5.0),
        ("hallway_1", "office_1", 5.1),
        ("hallway_1", "office_2", 5.1),
        ("hallway_1", "conference", 5.0),
        ("conference", "kitchen", 5.1),
        ("conference", "break_room", 5.1),
        ("conference", "hallway_2", 5.0),
        ("hallway_2", "restroom", 5.1),
        ("hallway_2", "storage", 5.1),
        ("hallway_2", "server_room", 5.0),
        ("kitchen", "hallway_2", 5.1),
        ("break_room", "hallway_2", 5.1),
    ]
    
    objects = [
        {"id": "fire_ext_1", "class": "fire_extinguisher", "position": [5, 2, 0], "room_id": "lobby"},
        {"id": "fire_ext_2", "class": "fire_extinguisher", "position": [15, 7, 0], "room_id": "kitchen"},
        {"id": "fire_ext_3", "class": "fire_extinguisher", "position": [25, 0, 0], "room_id": "server_room"},
        {"id": "coffee_1", "class": "coffee_mug", "position": [15, 6, 0], "room_id": "kitchen"},
        {"id": "coffee_2", "class": "coffee_mug", "position": [15, -4, 0], "room_id": "break_room"},
        {"id": "laptop_1", "class": "laptop", "position": [10, 5, 0], "room_id": "office_1"},
        {"id": "laptop_2", "class": "laptop", "position": [10, -5, 0], "room_id": "office_2"},
        {"id": "laptop_3", "class": "laptop", "position": [14, 0, 0], "room_id": "conference"},
        {"id": "printer_1", "class": "printer", "position": [20, -6, 0], "room_id": "storage"},
    ]
    
    return rooms, connections, objects


def run_simulation(
    num_agents: int = 2,
    target_object: str = "fire_extinguisher",
    max_steps: int = 500,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run a multi-agent inspection simulation.
    
    Args:
        num_agents: Number of agents
        target_object: Object to search for
        max_steps: Maximum simulation steps
        verbose: Print progress
        
    Returns:
        Simulation results
    """
    # Create environment
    rooms, connections, objects = create_office_environment()
    
    sim = SimulationEnvironment()
    dsg = sim.load_environment(rooms, connections, objects)
    
    # Create coordinator
    coordinator = MultiAgentCoordinator(
        coordination_mode=CoordinationMode.HYBRID
    )
    
    # Create RSN
    rsn = DistributedRSN(agent_id="coordinator")
    rsn.register_rooms([(r["id"], r["name"]) for r in rooms])
    
    # Create agents
    initial_positions = {}
    for i in range(num_agents):
        agent_id = f"agent_{i}"
        
        # Create agent
        agent = InspectionAgent(
            agent_id=agent_id,
            dsg=dsg,
            rsn=rsn
        )
        
        # Starting position near entrance
        start_pos = np.array([0, i * 2 - num_agents/2, 0])
        initial_positions[agent_id] = start_pos
        sim.place_agent(agent_id, start_pos)
        
        # Register with coordinator
        coordinator.register_agent(agent)
    
    # Initialize coordinator
    coordinator.set_shared_dsg(dsg)
    coordinator.set_shared_rsn(rsn)
    coordinator.initialize_from_blueprint(rooms, connections)
    
    # Compute optimal distance for SPL
    optimal_distance = sim.compute_optimal_distance(initial_positions, target_object)
    
    # Run simulation
    if verbose:
        print(f"Starting simulation with {num_agents} agents")
        print(f"Target: {target_object}")
        print(f"Optimal distance: {optimal_distance:.2f}m")
    
    found = False
    steps = 0
    
    # Initialize belief
    rsn.initialize_belief(target_object)
    
    # Compute initial plan
    belief = rsn.get_belief(target_object)
    agent_rooms = {
        aid: sim.get_agent_room(aid)
        for aid in initial_positions
    }
    
    coordinator.planner.compute_coordinated_plan(
        target_object=target_object,
        room_probabilities=belief.room_probabilities,
        dsg=dsg,
        agent_positions=agent_rooms
    )
    
    while not found and steps < max_steps:
        steps += 1
        sim.step()
        
        for agent_id in initial_positions:
            # Get current action
            action = coordinator.planner.get_agent_action(agent_id)
            
            if action is None:
                continue
            
            # Get target room position
            target_room = action.target_room
            if target_room in dsg.nodes:
                target_pos = dsg.nodes[target_room].position
            else:
                continue
            
            # Move agent
            new_pos, reached = sim.move_agent(agent_id, target_pos)
            
            if reached:
                # Simulate detection
                detections = sim.simulate_detection(
                    agent_id,
                    target_object,
                    thorough_search=(action.action_type.value == "search_room")
                )
                
                # Process detections
                for det in detections:
                    if det.is_true_positive and det.confidence > 0.7:
                        found = True
                        if verbose:
                            print(f"Agent {agent_id} found {target_object} at step {steps}")
                        break
                
                # Update belief
                rsn.update_belief_from_observation(
                    object_class=target_object,
                    room_id=target_room,
                    detected=len(detections) > 0,
                    confidence=max([d.confidence for d in detections]) if detections else 1.0,
                    thorough_search=True
                )
                
                # Report action complete
                coordinator.planner.report_action_complete(
                    agent_id=agent_id,
                    room_id=target_room,
                    found_target=found
                )
                
                # Replan if not found
                if not found:
                    belief = rsn.get_belief(target_object)
                    agent_rooms = {
                        aid: sim.get_agent_room(aid)
                        for aid in initial_positions
                    }
                    coordinator.planner.compute_coordinated_plan(
                        target_object=target_object,
                        room_probabilities=belief.room_probabilities,
                        dsg=dsg,
                        agent_positions=agent_rooms
                    )
        
        if verbose and steps % 50 == 0:
            print(f"Step {steps}, Rooms searched: {len(coordinator.planner.searched_rooms)}")
    
    # Compute results
    total_distance = sum(
        sim.get_agent_total_distance(aid)
        for aid in initial_positions
    )
    
    spl = 1.0 * (optimal_distance / max(total_distance, optimal_distance)) if found else 0.0
    
    results = {
        "success": found,
        "steps": steps,
        "total_distance": total_distance,
        "optimal_distance": optimal_distance,
        "spl": spl,
        "rooms_searched": len(coordinator.planner.searched_rooms),
        "detections": sim.total_detections,
        "true_positives": sim.true_positives,
        "false_positives": sim.false_positives,
        "num_agents": num_agents
    }
    
    if verbose:
        print(f"\nResults:")
        print(f"  Success: {found}")
        print(f"  Steps: {steps}")
        print(f"  Total distance: {total_distance:.2f}m")
        print(f"  SPL: {spl:.3f}")
        print(f"  Rooms searched: {len(coordinator.planner.searched_rooms)}")
    
    return results


if __name__ == "__main__":
    # Run simulation with different numbers of agents
    print("=" * 50)
    print("Single Agent")
    print("=" * 50)
    run_simulation(num_agents=1)
    
    print("\n" + "=" * 50)
    print("Two Agents")
    print("=" * 50)
    run_simulation(num_agents=2)
    
    print("\n" + "=" * 50)
    print("Three Agents")
    print("=" * 50)
    run_simulation(num_agents=3)
