"""
Multi-Agent Coordinator for SEEK-Multi

Central coordinator that manages multiple inspection agents
for collaborative object-goal navigation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any, Callable
import numpy as np
from enum import Enum
import time
import threading
from collections import defaultdict

from .agent import InspectionAgent, AgentState
from .shared_dsg import SharedDynamicSceneGraph, NodeType
from .distributed_rsn import DistributedRSN
from .collaborative_planner import CollaborativePlanner, CoordinatedPlan, ActionType
from .task_allocator import TaskAllocator, Task, AgentCapabilities, AllocationMethod
from .communication import CommunicationProtocol, Message, MessageType, ConsensusProtocol


class CoordinationMode(Enum):
    """Coordination modes for multi-agent system."""
    CENTRALIZED = "centralized"  # Central coordinator makes all decisions
    DISTRIBUTED = "distributed"  # Agents make local decisions
    HYBRID = "hybrid"  # Mix of central and local decisions


@dataclass
class InspectionMission:
    """
    Represents a multi-agent inspection mission.
    
    Attributes:
        mission_id: Unique identifier
        target_object: Object class to find
        start_time: Mission start time
        timeout: Maximum mission duration
        status: Current mission status
        result: Mission result when complete
    """
    mission_id: str
    target_object: str
    start_time: float = 0.0
    timeout: float = float('inf')
    status: str = "pending"  # pending, active, completed, failed
    result: Dict = field(default_factory=dict)


class MultiAgentCoordinator:
    """
    Coordinates multiple inspection agents for collaborative
    object-goal navigation.
    
    Features:
    - Centralized mission management
    - Distributed belief fusion
    - Task allocation and load balancing
    - Conflict resolution
    - Performance monitoring
    """
    
    def __init__(
        self,
        coordination_mode: CoordinationMode = CoordinationMode.HYBRID,
        replan_interval: float = 5.0,
        belief_fusion_interval: float = 1.0
    ):
        """
        Initialize multi-agent coordinator.
        
        Args:
            coordination_mode: How agents coordinate
            replan_interval: Seconds between replanning
            belief_fusion_interval: Seconds between belief fusion
        """
        self.coordination_mode = coordination_mode
        self.replan_interval = replan_interval
        self.belief_fusion_interval = belief_fusion_interval
        
        # Agents
        self.agents: Dict[str, InspectionAgent] = {}
        self.agent_capabilities: Dict[str, AgentCapabilities] = {}
        
        # Shared state
        self.shared_dsg: Optional[SharedDynamicSceneGraph] = None
        self.shared_rsn: Optional[DistributedRSN] = None
        
        # Planning
        self.planner: Optional[CollaborativePlanner] = None
        self.task_allocator: Optional[TaskAllocator] = None
        
        # Current mission
        self.current_mission: Optional[InspectionMission] = None
        self.current_plan: Optional[CoordinatedPlan] = None
        
        # Consensus for belief fusion
        self.consensus = ConsensusProtocol(num_agents=1)
        
        # Tracking
        self.mission_history: List[InspectionMission] = []
        self.performance_metrics: Dict[str, Any] = defaultdict(list)
        
        # Running state
        self._running = False
        self._coordinator_thread: Optional[threading.Thread] = None
        self._last_replan_time = 0.0
        self._last_fusion_time = 0.0
    
    def register_agent(
        self,
        agent: InspectionAgent,
        capabilities: List[str] = None,
        max_load: float = 10.0
    ) -> None:
        """
        Register an agent with the coordinator.
        
        Args:
            agent: The inspection agent
            capabilities: Agent capabilities
            max_load: Maximum task load for agent
        """
        self.agents[agent.agent_id] = agent
        
        self.agent_capabilities[agent.agent_id] = AgentCapabilities(
            agent_id=agent.agent_id,
            capabilities=set(capabilities or []),
            max_load=max_load
        )
        
        # Register agent with all other agents
        for other_id, other_agent in self.agents.items():
            if other_id != agent.agent_id:
                agent.register_other_agent(
                    other_id,
                    other_agent.status.position
                )
                other_agent.register_other_agent(
                    agent.agent_id,
                    agent.status.position
                )
        
        # Update consensus protocol
        self.consensus = ConsensusProtocol(num_agents=len(self.agents))
        
        # Update planner if exists
        if self.planner:
            self.planner = CollaborativePlanner(
                agent_ids=list(self.agents.keys())
            )
        
        # Update task allocator if exists
        if self.task_allocator:
            self.task_allocator = TaskAllocator(
                agents=list(self.agent_capabilities.values()),
                method=AllocationMethod.AUCTION
            )
    
    def set_shared_dsg(self, dsg: SharedDynamicSceneGraph) -> None:
        """Set the shared Dynamic Scene Graph."""
        self.shared_dsg = dsg
        
        # Distribute to all agents
        for agent in self.agents.values():
            agent.dsg = dsg
    
    def set_shared_rsn(self, rsn: DistributedRSN) -> None:
        """Set the shared Relational Semantic Network."""
        self.shared_rsn = rsn
        
        # Create individual RSNs for agents (they share model weights)
        for agent in self.agents.values():
            agent.rsn = rsn
    
    def initialize_from_blueprint(
        self,
        rooms: List[Dict],
        connections: List[Tuple[str, str, float]]
    ) -> None:
        """
        Initialize shared DSG from blueprint.
        
        Args:
            rooms: List of room dictionaries
            connections: Room connections with distances
        """
        self.shared_dsg = SharedDynamicSceneGraph.from_blueprint(
            agent_id="coordinator",
            rooms=rooms,
            connections=connections
        )
        
        # Create shared RSN
        self.shared_rsn = DistributedRSN(
            agent_id="coordinator"
        )
        
        # Register rooms with RSN
        room_info = [
            (room["id"], room.get("name", room["id"]))
            for room in rooms
        ]
        self.shared_rsn.register_rooms(room_info)
        
        # Initialize planner
        self.planner = CollaborativePlanner(
            agent_ids=list(self.agents.keys())
        )
        
        # Initialize task allocator
        self.task_allocator = TaskAllocator(
            agents=list(self.agent_capabilities.values()),
            method=AllocationMethod.AUCTION
        )
        
        # Distribute to agents
        for agent in self.agents.values():
            agent.dsg = self.shared_dsg
            agent.rsn = self.shared_rsn
            agent.set_planner(self.planner)
    
    def start_mission(
        self,
        target_object: str,
        initial_positions: Dict[str, np.ndarray],
        timeout: float = float('inf')
    ) -> InspectionMission:
        """
        Start a collaborative inspection mission.
        
        Args:
            target_object: Object class to find
            initial_positions: Starting positions for each agent
            timeout: Maximum mission duration
            
        Returns:
            The created mission
        """
        # Create mission
        mission = InspectionMission(
            mission_id=f"mission_{time.time()}",
            target_object=target_object,
            start_time=time.time(),
            timeout=timeout,
            status="active"
        )
        self.current_mission = mission
        
        # Initialize belief
        self.shared_rsn.initialize_belief(target_object)
        
        # Update agent positions
        for agent_id, position in initial_positions.items():
            if agent_id in self.agents:
                self.agents[agent_id].status.position = position
                self.agents[agent_id].status.current_room = (
                    self.shared_dsg.find_room_for_position(position)
                )
                self.agent_capabilities[agent_id].current_position = (
                    self.agents[agent_id].status.current_room
                )
        
        # Compute initial plan
        self._compute_coordinated_plan()
        
        # Start agents
        for agent_id, agent in self.agents.items():
            position = initial_positions.get(agent_id, np.zeros(3))
            agent.start_inspection(target_object, position)
        
        # Start coordinator loop
        self._running = True
        self._coordinator_thread = threading.Thread(target=self._coordinator_loop)
        self._coordinator_thread.start()
        
        return mission
    
    def stop_mission(self) -> Dict:
        """Stop the current mission."""
        self._running = False
        
        if self._coordinator_thread:
            self._coordinator_thread.join(timeout=5.0)
        
        # Stop all agents
        for agent in self.agents.values():
            agent.stop_inspection()
        
        # Finalize mission
        if self.current_mission:
            self.current_mission.status = "stopped"
            self.current_mission.result = self._compute_mission_result()
            self.mission_history.append(self.current_mission)
            
            result = self.current_mission.result
            self.current_mission = None
            return result
        
        return {}
    
    def _coordinator_loop(self) -> None:
        """Main coordinator loop."""
        while self._running and self.current_mission:
            try:
                current_time = time.time()
                
                # Check mission timeout
                if current_time - self.current_mission.start_time > self.current_mission.timeout:
                    self._mission_timeout()
                    break
                
                # Periodic belief fusion
                if current_time - self._last_fusion_time > self.belief_fusion_interval:
                    self._fuse_agent_beliefs()
                    self._last_fusion_time = current_time
                
                # Periodic replanning
                if current_time - self._last_replan_time > self.replan_interval:
                    self._check_replan_needed()
                    self._last_replan_time = current_time
                
                # Check for mission completion
                if self._check_mission_complete():
                    self._mission_complete()
                    break
                
                # Monitor agent status
                self._monitor_agents()
                
                # Update metrics
                self._update_metrics()
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Coordinator error: {e}")
                self.current_mission.status = "error"
                break
    
    def _compute_coordinated_plan(self) -> CoordinatedPlan:
        """Compute a new coordinated plan for all agents."""
        if not self.shared_rsn or not self.shared_dsg:
            return None
        
        # Get current belief
        belief = self.shared_rsn.get_belief(self.current_mission.target_object)
        
        # Get agent positions
        agent_positions = {}
        for agent_id, agent in self.agents.items():
            if agent.status.current_room:
                agent_positions[agent_id] = agent.status.current_room
        
        # Compute plan
        self.current_plan = self.planner.compute_coordinated_plan(
            target_object=self.current_mission.target_object,
            room_probabilities=belief.room_probabilities,
            dsg=self.shared_dsg,
            agent_positions=agent_positions
        )
        
        return self.current_plan
    
    def _fuse_agent_beliefs(self) -> None:
        """Fuse beliefs from all agents."""
        if not self.current_mission:
            return
        
        target_object = self.current_mission.target_object
        
        # Collect beliefs from all agents
        agent_beliefs = []
        for agent_id, agent in self.agents.items():
            belief = agent.rsn.get_belief(target_object)
            agent_beliefs.append((
                agent_id,
                belief.to_array(list(belief.room_probabilities.keys())),
                belief.confidence
            ))
        
        if len(agent_beliefs) < 2:
            return
        
        # Fuse using consensus
        room_ids = list(self.shared_rsn.get_belief(target_object).room_probabilities.keys())
        local_belief = self.shared_rsn.get_belief(target_object).to_array(room_ids)
        
        fused_belief = self.consensus.fuse_beliefs(
            local_belief=local_belief,
            neighbor_beliefs=agent_beliefs
        )
        
        # Update shared RSN
        fused_dict = {rid: p for rid, p in zip(room_ids, fused_belief)}
        self.shared_rsn.beliefs[target_object].room_probabilities = fused_dict
    
    def _check_replan_needed(self) -> bool:
        """Check if replanning is needed."""
        if not self.current_plan:
            self._compute_coordinated_plan()
            return True
        
        # Check if any agent has completed their plan
        all_complete = True
        for agent_id, agent in self.agents.items():
            plan = self.current_plan.get_agent_plan(agent_id)
            if plan and plan.get_current_action() is not None:
                all_complete = False
                break
        
        if all_complete:
            self._compute_coordinated_plan()
            return True
        
        # Check for significant belief changes
        belief = self.shared_rsn.get_belief(self.current_mission.target_object)
        entropy = self.shared_rsn.get_entropy(self.current_mission.target_object)
        
        # Replan if belief has changed significantly (low entropy = more certainty)
        if entropy < 1.0:  # High confidence in location
            self._compute_coordinated_plan()
            return True
        
        return False
    
    def _check_mission_complete(self) -> bool:
        """Check if mission is complete (target found)."""
        for agent in self.agents.values():
            if agent.status.state == AgentState.IDLE and self._running:
                # Agent finished - check if found target
                belief = agent.rsn.get_belief(self.current_mission.target_object)
                max_prob = max(belief.room_probabilities.values()) if belief.room_probabilities else 0
                if max_prob > 0.95:
                    return True
        
        return False
    
    def _mission_complete(self) -> None:
        """Handle mission completion."""
        self.current_mission.status = "completed"
        self.current_mission.result = self._compute_mission_result()
        self.mission_history.append(self.current_mission)
        self._running = False
    
    def _mission_timeout(self) -> None:
        """Handle mission timeout."""
        self.current_mission.status = "timeout"
        self.current_mission.result = self._compute_mission_result()
        self.mission_history.append(self.current_mission)
        self._running = False
    
    def _compute_mission_result(self) -> Dict:
        """Compute mission result metrics."""
        if not self.current_mission:
            return {}
        
        elapsed_time = time.time() - self.current_mission.start_time
        
        # Get final belief
        belief = self.shared_rsn.get_belief(self.current_mission.target_object)
        
        # Find most likely location
        if belief.room_probabilities:
            best_room = max(
                belief.room_probabilities.items(),
                key=lambda x: x[1]
            )
        else:
            best_room = (None, 0.0)
        
        # Compute total distance traveled
        total_distance = sum(
            self.performance_metrics.get("agent_distances", {}).get(aid, 0)
            for aid in self.agents
        )
        
        # Rooms searched
        rooms_searched = len(self.planner.searched_rooms) if self.planner else 0
        
        return {
            "mission_id": self.current_mission.mission_id,
            "target_object": self.current_mission.target_object,
            "status": self.current_mission.status,
            "elapsed_time": elapsed_time,
            "total_distance": total_distance,
            "rooms_searched": rooms_searched,
            "most_likely_room": best_room[0],
            "confidence": best_room[1],
            "num_agents": len(self.agents),
            "final_belief": belief.room_probabilities
        }
    
    def _monitor_agents(self) -> None:
        """Monitor agent status and handle issues."""
        for agent_id, agent in self.agents.items():
            # Check for errors
            if agent.status.state == AgentState.ERROR:
                print(f"Agent {agent_id} in error state: {agent.status.error_message}")
                # Could trigger recovery here
            
            # Check battery
            if agent.status.battery_level < 10:
                print(f"Agent {agent_id} low battery: {agent.status.battery_level}%")
            
            # Update position tracking
            self.agent_capabilities[agent_id].current_position = agent.status.current_room
    
    def _update_metrics(self) -> None:
        """Update performance metrics."""
        if not self.current_mission:
            return
        
        # Track agent positions for distance calculation
        for agent_id, agent in self.agents.items():
            if agent_id not in self.performance_metrics["agent_distances"]:
                self.performance_metrics["agent_distances"][agent_id] = 0.0
                self.performance_metrics["agent_last_pos"][agent_id] = agent.status.position.copy()
            else:
                last_pos = self.performance_metrics["agent_last_pos"][agent_id]
                distance = np.linalg.norm(agent.status.position - last_pos)
                self.performance_metrics["agent_distances"][agent_id] += distance
                self.performance_metrics["agent_last_pos"][agent_id] = agent.status.position.copy()
        
        # Track belief entropy over time
        belief = self.shared_rsn.get_belief(self.current_mission.target_object)
        entropy = self.shared_rsn.get_entropy(self.current_mission.target_object)
        self.performance_metrics["entropy_history"].append((time.time(), entropy))
    
    def get_status(self) -> Dict[str, Any]:
        """Get current coordinator status."""
        return {
            "coordination_mode": self.coordination_mode.value,
            "num_agents": len(self.agents),
            "mission": {
                "id": self.current_mission.mission_id if self.current_mission else None,
                "status": self.current_mission.status if self.current_mission else None,
                "target": self.current_mission.target_object if self.current_mission else None,
                "elapsed": (
                    time.time() - self.current_mission.start_time
                    if self.current_mission else 0
                )
            },
            "agents": {
                aid: agent.get_status()
                for aid, agent in self.agents.items()
            },
            "rooms_searched": len(self.planner.searched_rooms) if self.planner else 0,
            "running": self._running
        }
    
    def compute_spl(self, optimal_distance: float) -> float:
        """
        Compute Success weighted by Path Length (SPL).
        
        Args:
            optimal_distance: Shortest path distance to target
            
        Returns:
            SPL metric
        """
        if not self.current_mission or self.current_mission.status != "completed":
            return 0.0
        
        # Total distance traveled by all agents
        total_distance = sum(
            self.performance_metrics.get("agent_distances", {}).values()
        )
        
        # SPL = success * (optimal / max(actual, optimal))
        spl = 1.0 * (optimal_distance / max(total_distance, optimal_distance))
        return spl
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report for the mission."""
        if not self.current_mission:
            return {}
        
        result = self._compute_mission_result()
        
        # Add additional metrics
        result["spl"] = self.compute_spl(result.get("total_distance", 1.0) * 0.5)  # Estimate optimal
        result["entropy_history"] = self.performance_metrics.get("entropy_history", [])
        result["agent_distances"] = self.performance_metrics.get("agent_distances", {})
        
        return result
