"""
Shared Dynamic Scene Graph for Multi-Agent SEEK

Implements a distributed DSG that can be updated and merged
across multiple agents.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np
from enum import Enum
import copy
import time


class NodeType(Enum):
    """Types of nodes in the DSG."""
    OBJECT = 1
    LOCATION = 2
    ROOM = 3
    BUILDING = 4


@dataclass
class DSGNode:
    """
    Node in the Dynamic Scene Graph.
    
    Attributes:
        node_id: Unique identifier
        node_type: Type of node (object, location, room, building)
        position: 3D position
        orientation: Quaternion orientation
        semantic_class: Semantic label
        properties: Additional node properties
        parent_id: ID of parent node
        timestamp: Last update time
        updated_by: Agent that last updated this node
    """
    node_id: str
    node_type: NodeType
    position: np.ndarray
    orientation: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1]))
    semantic_class: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    timestamp: float = 0.0
    updated_by: str = ""
    
    def to_dict(self) -> Dict:
        """Convert node to dictionary."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "position": self.position.tolist(),
            "orientation": self.orientation.tolist(),
            "semantic_class": self.semantic_class,
            "properties": self.properties,
            "parent_id": self.parent_id,
            "timestamp": self.timestamp,
            "updated_by": self.updated_by
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "DSGNode":
        """Create node from dictionary."""
        return cls(
            node_id=data["node_id"],
            node_type=NodeType(data["node_type"]),
            position=np.array(data["position"]),
            orientation=np.array(data["orientation"]),
            semantic_class=data["semantic_class"],
            properties=data.get("properties", {}),
            parent_id=data.get("parent_id"),
            timestamp=data.get("timestamp", 0.0),
            updated_by=data.get("updated_by", "")
        )


@dataclass
class DSGEdge:
    """
    Edge in the Dynamic Scene Graph.
    
    Attributes:
        source_id: Source node ID
        target_id: Target node ID
        edge_type: Type of relationship
        weight: Edge weight (e.g., distance, traversability)
        properties: Additional edge properties
        timestamp: Last update time
    """
    source_id: str
    target_id: str
    edge_type: str
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert edge to dictionary."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type,
            "weight": self.weight,
            "properties": self.properties,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "DSGEdge":
        """Create edge from dictionary."""
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            edge_type=data["edge_type"],
            weight=data.get("weight", 1.0),
            properties=data.get("properties", {}),
            timestamp=data.get("timestamp", 0.0)
        )


class SharedDynamicSceneGraph:
    """
    A Dynamic Scene Graph that supports distributed updates
    and merging from multiple agents.
    
    The graph is hierarchical with layers:
    - Layer 1: Objects/Locations
    - Layer 2: Rooms/Regions
    - Layer 3: Buildings
    
    Supports:
    - Distributed updates with conflict resolution
    - Incremental graph merging
    - Semantic queries
    - Path planning on the graph
    """
    
    def __init__(self, agent_id: str):
        """
        Initialize the shared DSG.
        
        Args:
            agent_id: ID of the owning agent
        """
        self.agent_id = agent_id
        
        # Nodes organized by layer
        self.nodes: Dict[str, DSGNode] = {}
        self.nodes_by_layer: Dict[NodeType, Set[str]] = {
            NodeType.OBJECT: set(),
            NodeType.LOCATION: set(),
            NodeType.ROOM: set(),
            NodeType.BUILDING: set()
        }
        
        # Edges
        self.edges: Dict[Tuple[str, str], DSGEdge] = {}
        self.adjacency: Dict[str, Set[str]] = {}
        
        # Version control for merging
        self.version = 0
        self.update_log: List[Dict] = []
        self.max_log_size = 1000
        
        # Precomputed distances for planning
        self.distance_matrix: Dict[Tuple[str, str], float] = {}
        self.distance_matrix_valid = False
    
    def add_node(
        self,
        node_id: str,
        node_type: NodeType,
        position: np.ndarray,
        semantic_class: str = "",
        orientation: np.ndarray = None,
        properties: Dict = None,
        parent_id: str = None
    ) -> DSGNode:
        """
        Add a node to the graph.
        
        Args:
            node_id: Unique identifier
            node_type: Type of node
            position: 3D position
            semantic_class: Semantic label
            orientation: Quaternion orientation
            properties: Additional properties
            parent_id: Parent node ID
            
        Returns:
            The created node
        """
        if orientation is None:
            orientation = np.array([0, 0, 0, 1])
        if properties is None:
            properties = {}
        
        node = DSGNode(
            node_id=node_id,
            node_type=node_type,
            position=position,
            orientation=orientation,
            semantic_class=semantic_class,
            properties=properties,
            parent_id=parent_id,
            timestamp=time.time(),
            updated_by=self.agent_id
        )
        
        self.nodes[node_id] = node
        self.nodes_by_layer[node_type].add(node_id)
        self.adjacency[node_id] = set()
        
        # Add edge to parent if specified
        if parent_id and parent_id in self.nodes:
            self.add_edge(node_id, parent_id, "parent_child")
        
        # Invalidate distance matrix
        self.distance_matrix_valid = False
        
        # Log update
        self._log_update("add_node", node.to_dict())
        self.version += 1
        
        return node
    
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        weight: float = 1.0,
        properties: Dict = None
    ) -> Optional[DSGEdge]:
        """
        Add an edge between two nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Type of relationship
            weight: Edge weight
            properties: Additional properties
            
        Returns:
            The created edge, or None if nodes don't exist
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            return None
        
        if properties is None:
            properties = {}
        
        edge = DSGEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            properties=properties,
            timestamp=time.time()
        )
        
        self.edges[(source_id, target_id)] = edge
        self.adjacency[source_id].add(target_id)
        self.adjacency[target_id].add(source_id)
        
        self.distance_matrix_valid = False
        self._log_update("add_edge", edge.to_dict())
        self.version += 1
        
        return edge
    
    def update_node(
        self,
        node_id: str,
        position: np.ndarray = None,
        orientation: np.ndarray = None,
        properties: Dict = None,
        semantic_class: str = None
    ) -> Optional[DSGNode]:
        """Update an existing node."""
        if node_id not in self.nodes:
            return None
        
        node = self.nodes[node_id]
        
        if position is not None:
            node.position = position
        if orientation is not None:
            node.orientation = orientation
        if properties is not None:
            node.properties.update(properties)
        if semantic_class is not None:
            node.semantic_class = semantic_class
        
        node.timestamp = time.time()
        node.updated_by = self.agent_id
        
        self.distance_matrix_valid = False
        self._log_update("update_node", node.to_dict())
        self.version += 1
        
        return node
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[DSGNode]:
        """Get all nodes of a specific type."""
        return [self.nodes[nid] for nid in self.nodes_by_layer[node_type]]
    
    def get_nodes_by_semantic_class(self, semantic_class: str) -> List[DSGNode]:
        """Get all nodes with a specific semantic class."""
        return [n for n in self.nodes.values() if n.semantic_class == semantic_class]
    
    def get_room_nodes(self) -> List[DSGNode]:
        """Get all room-level nodes."""
        return self.get_nodes_by_type(NodeType.ROOM)
    
    def get_children(self, node_id: str) -> List[DSGNode]:
        """Get all children of a node."""
        return [n for n in self.nodes.values() if n.parent_id == node_id]
    
    def get_objects_in_room(self, room_id: str) -> List[DSGNode]:
        """Get all objects within a room."""
        objects = []
        for node in self.nodes.values():
            if node.node_type == NodeType.OBJECT and node.parent_id == room_id:
                objects.append(node)
        return objects
    
    def find_room_for_position(self, position: np.ndarray) -> Optional[str]:
        """Find the room containing a given position."""
        rooms = self.get_room_nodes()
        
        min_dist = float('inf')
        closest_room = None
        
        for room in rooms:
            # Simple distance-based assignment
            # In practice, would use room boundaries
            dist = np.linalg.norm(position[:2] - room.position[:2])
            if dist < min_dist:
                min_dist = dist
                closest_room = room.node_id
        
        return closest_room
    
    def compute_distance_matrix(self) -> None:
        """Compute shortest path distances between all room nodes."""
        rooms = list(self.nodes_by_layer[NodeType.ROOM])
        n = len(rooms)
        
        # Initialize with edge weights
        dist = {(r1, r2): float('inf') for r1 in rooms for r2 in rooms}
        for room_id in rooms:
            dist[(room_id, room_id)] = 0
        
        for (src, tgt), edge in self.edges.items():
            if src in rooms and tgt in rooms:
                dist[(src, tgt)] = edge.weight
                dist[(tgt, src)] = edge.weight
        
        # Floyd-Warshall algorithm
        for k in rooms:
            for i in rooms:
                for j in rooms:
                    if dist[(i, k)] + dist[(k, j)] < dist[(i, j)]:
                        dist[(i, j)] = dist[(i, k)] + dist[(k, j)]
        
        self.distance_matrix = dist
        self.distance_matrix_valid = True
    
    def get_distance(self, room1: str, room2: str) -> float:
        """Get the shortest distance between two rooms."""
        if not self.distance_matrix_valid:
            self.compute_distance_matrix()
        return self.distance_matrix.get((room1, room2), float('inf'))
    
    def find_shortest_path(
        self,
        start_id: str,
        goal_id: str
    ) -> Tuple[List[str], float]:
        """
        Find shortest path between two nodes using A*.
        
        Args:
            start_id: Starting node ID
            goal_id: Goal node ID
            
        Returns:
            Tuple of (path as list of node IDs, total distance)
        """
        if start_id not in self.nodes or goal_id not in self.nodes:
            return [], float('inf')
        
        import heapq
        
        # A* search
        start_pos = self.nodes[start_id].position
        goal_pos = self.nodes[goal_id].position
        
        def heuristic(node_id: str) -> float:
            pos = self.nodes[node_id].position
            return np.linalg.norm(pos - goal_pos)
        
        open_set = [(heuristic(start_id), 0, start_id, [start_id])]
        closed_set = set()
        
        while open_set:
            _, g, current, path = heapq.heappop(open_set)
            
            if current == goal_id:
                return path, g
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            for neighbor in self.adjacency.get(current, []):
                if neighbor in closed_set:
                    continue
                
                edge_key = (current, neighbor)
                if edge_key not in self.edges:
                    edge_key = (neighbor, current)
                
                edge_weight = self.edges.get(edge_key, DSGEdge(current, neighbor, "")).weight
                new_g = g + edge_weight
                f = new_g + heuristic(neighbor)
                
                heapq.heappush(open_set, (f, new_g, neighbor, path + [neighbor]))
        
        return [], float('inf')
    
    def merge(self, other: "SharedDynamicSceneGraph") -> "SharedDynamicSceneGraph":
        """
        Merge another DSG into this one.
        
        Uses timestamp-based conflict resolution.
        
        Args:
            other: Another DSG to merge
            
        Returns:
            The merged DSG (self)
        """
        # Merge nodes
        for node_id, node in other.nodes.items():
            if node_id not in self.nodes:
                # New node
                self.nodes[node_id] = copy.deepcopy(node)
                self.nodes_by_layer[node.node_type].add(node_id)
                self.adjacency[node_id] = set()
            else:
                # Conflict resolution: use newer timestamp
                if node.timestamp > self.nodes[node_id].timestamp:
                    old_type = self.nodes[node_id].node_type
                    self.nodes[node_id] = copy.deepcopy(node)
                    # Update layer tracking if type changed
                    if old_type != node.node_type:
                        self.nodes_by_layer[old_type].discard(node_id)
                        self.nodes_by_layer[node.node_type].add(node_id)
        
        # Merge edges
        for edge_key, edge in other.edges.items():
            if edge_key not in self.edges:
                self.edges[edge_key] = copy.deepcopy(edge)
                self.adjacency[edge_key[0]].add(edge_key[1])
                self.adjacency[edge_key[1]].add(edge_key[0])
            else:
                # Use newer timestamp
                if edge.timestamp > self.edges[edge_key].timestamp:
                    self.edges[edge_key] = copy.deepcopy(edge)
        
        self.distance_matrix_valid = False
        self.version += 1
        
        return self
    
    def get_incremental_updates(self, since_version: int) -> List[Dict]:
        """Get updates since a specific version."""
        return [
            update for update in self.update_log
            if update.get("version", 0) > since_version
        ]
    
    def apply_incremental_update(self, update: Dict) -> None:
        """Apply an incremental update from another agent."""
        update_type = update.get("type")
        data = update.get("data", {})
        
        if update_type == "add_node":
            node = DSGNode.from_dict(data)
            if node.node_id not in self.nodes or node.timestamp > self.nodes[node.node_id].timestamp:
                self.nodes[node.node_id] = node
                self.nodes_by_layer[node.node_type].add(node.node_id)
                if node.node_id not in self.adjacency:
                    self.adjacency[node.node_id] = set()
                    
        elif update_type == "update_node":
            node = DSGNode.from_dict(data)
            if node.node_id in self.nodes and node.timestamp > self.nodes[node.node_id].timestamp:
                self.nodes[node.node_id] = node
                
        elif update_type == "add_edge":
            edge = DSGEdge.from_dict(data)
            edge_key = (edge.source_id, edge.target_id)
            if edge_key not in self.edges or edge.timestamp > self.edges[edge_key].timestamp:
                self.edges[edge_key] = edge
                self.adjacency[edge.source_id].add(edge.target_id)
                self.adjacency[edge.target_id].add(edge.source_id)
        
        self.distance_matrix_valid = False
    
    def _log_update(self, update_type: str, data: Dict) -> None:
        """Log an update for incremental sharing."""
        self.update_log.append({
            "type": update_type,
            "data": data,
            "version": self.version,
            "timestamp": time.time(),
            "agent_id": self.agent_id
        })
        
        # Trim log if too large
        if len(self.update_log) > self.max_log_size:
            self.update_log = self.update_log[-self.max_log_size // 2:]
    
    def to_dict(self) -> Dict:
        """Convert entire DSG to dictionary."""
        return {
            "agent_id": self.agent_id,
            "version": self.version,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "edges": {f"{k[0]}_{k[1]}": e.to_dict() for k, e in self.edges.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "SharedDynamicSceneGraph":
        """Create DSG from dictionary."""
        dsg = cls(data["agent_id"])
        dsg.version = data.get("version", 0)
        
        for node_data in data.get("nodes", {}).values():
            node = DSGNode.from_dict(node_data)
            dsg.nodes[node.node_id] = node
            dsg.nodes_by_layer[node.node_type].add(node.node_id)
            dsg.adjacency[node.node_id] = set()
        
        for edge_data in data.get("edges", {}).values():
            edge = DSGEdge.from_dict(edge_data)
            dsg.edges[(edge.source_id, edge.target_id)] = edge
            dsg.adjacency[edge.source_id].add(edge.target_id)
            dsg.adjacency[edge.target_id].add(edge.source_id)
        
        return dsg
    
    @classmethod
    def from_blueprint(
        cls,
        agent_id: str,
        rooms: List[Dict],
        connections: List[Tuple[str, str, float]]
    ) -> "SharedDynamicSceneGraph":
        """
        Create DSG from blueprint data.
        
        Args:
            agent_id: Agent ID
            rooms: List of room dictionaries with id, name, position
            connections: List of (room1_id, room2_id, distance) tuples
            
        Returns:
            Initialized DSG
        """
        dsg = cls(agent_id)
        
        # Add room nodes
        for room in rooms:
            dsg.add_node(
                node_id=room["id"],
                node_type=NodeType.ROOM,
                position=np.array(room["position"]),
                semantic_class=room.get("name", room["id"]),
                properties=room.get("properties", {})
            )
        
        # Add location nodes within rooms
        for room in rooms:
            room_node = dsg.nodes[room["id"]]
            # Sample location nodes
            for i in range(room.get("num_locations", 4)):
                offset = np.random.randn(3) * 2
                offset[2] = 0  # Keep on same floor
                loc_pos = room_node.position + offset
                
                dsg.add_node(
                    node_id=f"{room['id']}_loc_{i}",
                    node_type=NodeType.LOCATION,
                    position=loc_pos,
                    parent_id=room["id"]
                )
        
        # Add edges between rooms
        for room1_id, room2_id, distance in connections:
            dsg.add_edge(room1_id, room2_id, "connected", weight=distance)
        
        return dsg
