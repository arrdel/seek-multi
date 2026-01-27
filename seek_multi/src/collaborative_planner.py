"""
Collaborative Planner for Multi-Agent SEEK

Implements distributed planning algorithms for coordinated
multi-robot object-goal navigation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np
from enum import Enum
import heapq
import time
from collections import defaultdict


class ActionType(Enum):
    """Types of actions in the global plan."""
    MOVE_TO_ROOM = "move_to_room"
    SEARCH_ROOM = "search_room"
    INSPECT_OBJECT = "inspect_object"
    WAIT = "wait"
    COORDINATE = "coordinate"


@dataclass
class GlobalAction:
    """
    An action in the global planning space.
    
    Attributes:
        action_type: Type of action
        target_room: Target room ID
        priority: Action priority
        estimated_cost: Estimated execution cost
        estimated_duration: Estimated time to complete
        dependencies: Actions that must complete first
    """
    action_type: ActionType
    target_room: str
    priority: float = 0.0
    estimated_cost: float = 0.0
    estimated_duration: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    
    def __hash__(self):
        return hash((self.action_type, self.target_room))


@dataclass
class AgentPlan:
    """
    Plan for a single agent.
    
    Attributes:
        agent_id: Agent identifier
        actions: Ordered list of actions
        estimated_total_cost: Total estimated cost
        current_action_idx: Index of current action
    """
    agent_id: str
    actions: List[GlobalAction]
    estimated_total_cost: float = 0.0
    current_action_idx: int = 0
    
    def get_current_action(self) -> Optional[GlobalAction]:
        """Get the current action to execute."""
        if self.current_action_idx < len(self.actions):
            return self.actions[self.current_action_idx]
        return None
    
    def advance(self) -> bool:
        """Advance to next action. Returns True if more actions exist."""
        self.current_action_idx += 1
        return self.current_action_idx < len(self.actions)


@dataclass
class CoordinatedPlan:
    """
    Coordinated plan for multiple agents.
    
    Attributes:
        agent_plans: Dictionary of agent plans
        coordination_points: Points where agents need to synchronize
        estimated_completion_time: Estimated time to find target
    """
    agent_plans: Dict[str, AgentPlan]
    coordination_points: List[Dict] = field(default_factory=list)
    estimated_completion_time: float = 0.0
    
    def get_agent_plan(self, agent_id: str) -> Optional[AgentPlan]:
        return self.agent_plans.get(agent_id)


class CollaborativePlanner:
    """
    Collaborative planner for multi-agent object-goal navigation.
    
    Implements:
    - Distributed MDP solving
    - Task partitioning
    - Conflict resolution
    - Dynamic replanning
    """
    
    def __init__(
        self,
        agent_ids: List[str],
        discount_factor: float = 0.95,
        coordination_penalty: float = 0.1
    ):
        """
        Initialize the collaborative planner.
        
        Args:
            agent_ids: List of agent identifiers
            discount_factor: MDP discount factor
            coordination_penalty: Penalty for agents visiting same room
        """
        self.agent_ids = agent_ids
        self.num_agents = len(agent_ids)
        self.discount_factor = discount_factor
        self.coordination_penalty = coordination_penalty
        
        # State tracking
        self.agent_positions: Dict[str, str] = {}  # agent_id -> room_id
        self.visited_rooms: Dict[str, Set[str]] = {
            aid: set() for aid in agent_ids
        }  # agent_id -> set of visited rooms
        self.searched_rooms: Set[str] = set()  # Globally searched rooms
        
        # Planning state
        self.current_plan: Optional[CoordinatedPlan] = None
        self.value_function: Dict[Tuple, float] = {}
        
        # Intentions for conflict avoidance
        self.agent_intentions: Dict[str, GlobalAction] = {}
    
    def compute_joint_value(
        self,
        room_probabilities: Dict[str, float],
        room_costs: Dict[Tuple[str, str], float],
        search_costs: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute value function for each room.
        
        Uses the probability of finding the target object
        and the cost to reach/search each room.
        
        Args:
            room_probabilities: P(target in room) for each room
            room_costs: Cost to travel between rooms
            search_costs: Cost to search each room
            
        Returns:
            Value for visiting each room
        """
        values = {}
        
        for room_id, prob in room_probabilities.items():
            if room_id in self.searched_rooms:
                # Already searched, lower value
                values[room_id] = 0.0
                continue
            
            # Value = probability * reward - expected future cost
            # Simplified: value proportional to probability
            search_cost = search_costs.get(room_id, 1.0)
            values[room_id] = prob / (search_cost + 0.1)
        
        return values
    
    def compute_coordinated_plan(
        self,
        target_object: str,
        room_probabilities: Dict[str, float],
        dsg,  # SharedDynamicSceneGraph
        agent_positions: Dict[str, str]
    ) -> CoordinatedPlan:
        """
        Compute a coordinated plan for all agents.
        
        Uses task allocation to assign rooms to agents
        while minimizing overlap and maximizing coverage.
        
        Args:
            target_object: Target object class
            room_probabilities: Probability distribution over rooms
            dsg: Shared Dynamic Scene Graph
            agent_positions: Current position of each agent
            
        Returns:
            Coordinated plan for all agents
        """
        self.agent_positions = agent_positions.copy()
        
        # Get all rooms
        room_nodes = dsg.get_room_nodes()
        room_ids = [r.node_id for r in room_nodes]
        
        # Compute search costs
        search_costs = {}
        for room in room_nodes:
            # Estimate search cost based on room size
            search_costs[room.node_id] = room.properties.get("area", 10.0) / 5.0
        
        # Compute room values
        room_values = self.compute_joint_value(
            room_probabilities,
            {},  # Will use DSG for distances
            search_costs
        )
        
        # Task allocation using auction-based approach
        agent_plans = self._auction_based_allocation(
            room_ids,
            room_values,
            room_probabilities,
            dsg,
            agent_positions,
            search_costs
        )
        
        # Detect and resolve conflicts
        agent_plans = self._resolve_conflicts(agent_plans, dsg)
        
        # Compute coordination points
        coordination_points = self._compute_coordination_points(agent_plans)
        
        # Estimate completion time
        completion_time = self._estimate_completion_time(agent_plans, room_probabilities)
        
        plan = CoordinatedPlan(
            agent_plans=agent_plans,
            coordination_points=coordination_points,
            estimated_completion_time=completion_time
        )
        
        self.current_plan = plan
        return plan
    
    def _auction_based_allocation(
        self,
        room_ids: List[str],
        room_values: Dict[str, float],
        room_probabilities: Dict[str, float],
        dsg,
        agent_positions: Dict[str, str],
        search_costs: Dict[str, float]
    ) -> Dict[str, AgentPlan]:
        """
        Allocate rooms to agents using auction-based mechanism.
        
        Each agent bids on rooms based on value and distance.
        Rooms are assigned to highest bidder.
        """
        # Initialize agent plans
        agent_plans = {
            aid: AgentPlan(agent_id=aid, actions=[])
            for aid in self.agent_ids
        }
        
        # Available rooms (not yet assigned)
        available_rooms = set(room_ids) - self.searched_rooms
        
        # Sort rooms by value for greedy assignment
        sorted_rooms = sorted(
            available_rooms,
            key=lambda r: room_values.get(r, 0),
            reverse=True
        )
        
        # Track agent workloads
        agent_costs = {aid: 0.0 for aid in self.agent_ids}
        agent_current_room = agent_positions.copy()
        
        # Assign rooms round-robin by value
        for room_id in sorted_rooms:
            if room_values.get(room_id, 0) <= 0:
                continue
            
            # Find best agent for this room
            best_agent = None
            best_bid = float('-inf')
            
            for agent_id in self.agent_ids:
                # Compute bid: value - travel cost
                current_room = agent_current_room.get(agent_id)
                if current_room:
                    travel_cost = dsg.get_distance(current_room, room_id)
                else:
                    travel_cost = 0
                
                # Bid considers value, travel cost, and current workload balance
                workload_factor = 1.0 / (1.0 + agent_costs[agent_id])
                bid = room_values[room_id] * workload_factor - travel_cost
                
                if bid > best_bid:
                    best_bid = bid
                    best_agent = agent_id
            
            if best_agent:
                # Add move action
                agent_plans[best_agent].actions.append(GlobalAction(
                    action_type=ActionType.MOVE_TO_ROOM,
                    target_room=room_id,
                    priority=room_probabilities.get(room_id, 0),
                    estimated_cost=dsg.get_distance(
                        agent_current_room.get(best_agent, room_id),
                        room_id
                    )
                ))
                
                # Add search action
                agent_plans[best_agent].actions.append(GlobalAction(
                    action_type=ActionType.SEARCH_ROOM,
                    target_room=room_id,
                    priority=room_probabilities.get(room_id, 0),
                    estimated_cost=search_costs.get(room_id, 1.0)
                ))
                
                # Update tracking
                agent_current_room[best_agent] = room_id
                agent_costs[best_agent] += (
                    dsg.get_distance(
                        agent_positions.get(best_agent, room_id),
                        room_id
                    ) + search_costs.get(room_id, 1.0)
                )
        
        # Compute total costs
        for agent_id, plan in agent_plans.items():
            plan.estimated_total_cost = sum(a.estimated_cost for a in plan.actions)
        
        return agent_plans
    
    def _resolve_conflicts(
        self,
        agent_plans: Dict[str, AgentPlan],
        dsg
    ) -> Dict[str, AgentPlan]:
        """
        Resolve conflicts where multiple agents target same room.
        
        Uses priority-based resolution.
        """
        # Find conflicts
        room_assignments: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        
        for agent_id, plan in agent_plans.items():
            for idx, action in enumerate(plan.actions):
                if action.action_type in [ActionType.MOVE_TO_ROOM, ActionType.SEARCH_ROOM]:
                    room_assignments[action.target_room].append((agent_id, idx))
        
        # Resolve conflicts (keep highest priority assignment)
        for room_id, assignments in room_assignments.items():
            if len(assignments) <= 1:
                continue
            
            # Sort by priority (embedded in action)
            # Keep only one agent for search
            search_assignments = [
                (aid, idx) for aid, idx in assignments
                if agent_plans[aid].actions[idx].action_type == ActionType.SEARCH_ROOM
            ]
            
            if len(search_assignments) > 1:
                # Keep the one with highest probability consideration
                # (first one in sorted order)
                for aid, idx in search_assignments[1:]:
                    # Change search to just move-through
                    agent_plans[aid].actions[idx] = GlobalAction(
                        action_type=ActionType.WAIT,
                        target_room=room_id,
                        estimated_cost=0.1
                    )
        
        return agent_plans
    
    def _compute_coordination_points(
        self,
        agent_plans: Dict[str, AgentPlan]
    ) -> List[Dict]:
        """Identify points where agents should coordinate."""
        coordination_points = []
        
        # Find rooms where multiple agents will be present
        time_positions: Dict[float, Dict[str, str]] = defaultdict(dict)
        
        for agent_id, plan in agent_plans.items():
            cumulative_time = 0.0
            for action in plan.actions:
                cumulative_time += action.estimated_duration or action.estimated_cost
                time_positions[cumulative_time][agent_id] = action.target_room
        
        # Check for co-location
        for time_point, positions in time_positions.items():
            room_agents: Dict[str, List[str]] = defaultdict(list)
            for agent_id, room_id in positions.items():
                room_agents[room_id].append(agent_id)
            
            for room_id, agents in room_agents.items():
                if len(agents) > 1:
                    coordination_points.append({
                        "time": time_point,
                        "room": room_id,
                        "agents": agents,
                        "type": "co-location"
                    })
        
        return coordination_points
    
    def _estimate_completion_time(
        self,
        agent_plans: Dict[str, AgentPlan],
        room_probabilities: Dict[str, float]
    ) -> float:
        """
        Estimate expected time to find target.
        
        Uses probability-weighted expected search time.
        """
        # Compute expected rooms searched before finding target
        cumulative_prob = 0.0
        expected_time = 0.0
        
        # Interleave agent actions by time
        all_actions = []
        for agent_id, plan in agent_plans.items():
            cumulative_time = 0.0
            for action in plan.actions:
                if action.action_type == ActionType.SEARCH_ROOM:
                    all_actions.append((
                        cumulative_time,
                        action.target_room,
                        room_probabilities.get(action.target_room, 0)
                    ))
                cumulative_time += action.estimated_cost
        
        # Sort by time
        all_actions.sort(key=lambda x: x[0])
        
        # Compute expected time
        for time_point, room_id, prob in all_actions:
            if cumulative_prob >= 0.99:
                break
            
            # Probability of finding at this room given not found before
            conditional_prob = prob / (1 - cumulative_prob + 1e-10)
            expected_time += conditional_prob * time_point
            cumulative_prob += prob
        
        return expected_time
    
    def get_agent_action(
        self,
        agent_id: str
    ) -> Optional[GlobalAction]:
        """Get the current action for an agent."""
        if self.current_plan is None:
            return None
        
        plan = self.current_plan.get_agent_plan(agent_id)
        if plan is None:
            return None
        
        return plan.get_current_action()
    
    def report_action_complete(
        self,
        agent_id: str,
        room_id: str,
        found_target: bool
    ) -> bool:
        """
        Report that an agent completed an action.
        
        Args:
            agent_id: Agent identifier
            room_id: Room where action was completed
            found_target: Whether target was found
            
        Returns:
            True if target was found and planning complete
        """
        if found_target:
            return True
        
        # Update tracking
        self.searched_rooms.add(room_id)
        self.visited_rooms[agent_id].add(room_id)
        
        # Advance agent's plan
        if self.current_plan:
            plan = self.current_plan.get_agent_plan(agent_id)
            if plan:
                plan.advance()
        
        return False
    
    def update_intention(
        self,
        agent_id: str,
        intention: GlobalAction
    ) -> None:
        """Update an agent's declared intention."""
        self.agent_intentions[agent_id] = intention
    
    def check_intention_conflict(
        self,
        agent_id: str,
        intended_room: str
    ) -> List[str]:
        """Check if intention conflicts with other agents."""
        conflicts = []
        for other_id, intention in self.agent_intentions.items():
            if other_id != agent_id and intention.target_room == intended_room:
                conflicts.append(other_id)
        return conflicts
    
    def replan(
        self,
        target_object: str,
        room_probabilities: Dict[str, float],
        dsg,
        agent_positions: Dict[str, str],
        trigger_reason: str = "periodic"
    ) -> CoordinatedPlan:
        """
        Trigger replanning with updated information.
        
        Called when:
        - Significant belief update
        - Agent completes task
        - Conflict detected
        - Periodic replanning
        """
        return self.compute_coordinated_plan(
            target_object,
            room_probabilities,
            dsg,
            agent_positions
        )


class DecentralizedMDPSolver:
    """
    Solves decentralized MDPs for multi-agent planning.
    
    Uses factored value iteration with coordination constraints.
    """
    
    def __init__(
        self,
        num_agents: int,
        discount: float = 0.95,
        convergence_threshold: float = 0.001
    ):
        self.num_agents = num_agents
        self.discount = discount
        self.convergence_threshold = convergence_threshold
    
    def solve(
        self,
        states: List[str],
        actions: List[ActionType],
        transition_probs: Dict,
        rewards: Dict,
        initial_values: Dict = None
    ) -> Tuple[Dict[str, Dict[str, ActionType]], Dict[str, float]]:
        """
        Solve the decentralized MDP.
        
        Uses value iteration with agent-specific action selection.
        
        Returns:
            Tuple of (policy per agent, value function)
        """
        # Initialize value function
        if initial_values is None:
            values = {s: 0.0 for s in states}
        else:
            values = initial_values.copy()
        
        # Value iteration
        iteration = 0
        while True:
            iteration += 1
            max_delta = 0.0
            new_values = {}
            
            for state in states:
                # Compute value for each action
                action_values = []
                
                for action in actions:
                    # Expected value of taking action
                    expected_value = rewards.get((state, action), 0.0)
                    
                    # Sum over possible next states
                    for next_state in states:
                        prob = transition_probs.get((state, action, next_state), 0.0)
                        expected_value += self.discount * prob * values.get(next_state, 0.0)
                    
                    action_values.append((action, expected_value))
                
                # Take max
                best_value = max(av[1] for av in action_values) if action_values else 0.0
                new_values[state] = best_value
                
                max_delta = max(max_delta, abs(best_value - values.get(state, 0.0)))
            
            values = new_values
            
            if max_delta < self.convergence_threshold:
                break
            
            if iteration > 1000:
                print("Warning: Value iteration did not converge")
                break
        
        # Extract policy
        policy = {}
        for state in states:
            action_values = []
            for action in actions:
                expected_value = rewards.get((state, action), 0.0)
                for next_state in states:
                    prob = transition_probs.get((state, action, next_state), 0.0)
                    expected_value += self.discount * prob * values.get(next_state, 0.0)
                action_values.append((action, expected_value))
            
            if action_values:
                best_action = max(action_values, key=lambda x: x[1])[0]
                policy[state] = best_action
        
        return policy, values
