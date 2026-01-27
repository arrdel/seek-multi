"""
Task Allocator for Multi-Agent SEEK

Implements task allocation algorithms for distributing inspection
tasks among multiple robots.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np
from enum import Enum
import heapq
from collections import defaultdict


class AllocationMethod(Enum):
    """Task allocation methods."""
    GREEDY = "greedy"
    AUCTION = "auction"
    HUNGARIAN = "hungarian"
    MARKET_BASED = "market_based"


@dataclass
class Task:
    """
    Represents an inspection task.
    
    Attributes:
        task_id: Unique identifier
        target_room: Room to inspect
        priority: Task priority (higher = more important)
        estimated_cost: Estimated cost to complete
        deadline: Optional deadline
        required_capabilities: Required agent capabilities
        dependencies: Tasks that must complete first
    """
    task_id: str
    target_room: str
    priority: float = 1.0
    estimated_cost: float = 1.0
    deadline: Optional[float] = None
    required_capabilities: Set[str] = field(default_factory=set)
    dependencies: List[str] = field(default_factory=list)
    assigned_agent: Optional[str] = None
    status: str = "pending"  # pending, assigned, in_progress, completed


@dataclass
class AgentCapabilities:
    """
    Describes an agent's capabilities and current state.
    
    Attributes:
        agent_id: Agent identifier
        capabilities: Set of capabilities (e.g., "fly", "climb")
        current_load: Current task load
        max_load: Maximum task load
        current_position: Current room position
        speed: Movement speed factor
    """
    agent_id: str
    capabilities: Set[str] = field(default_factory=set)
    current_load: float = 0.0
    max_load: float = 10.0
    current_position: Optional[str] = None
    speed: float = 1.0
    
    def can_perform(self, task: Task) -> bool:
        """Check if agent can perform a task."""
        return task.required_capabilities.issubset(self.capabilities)
    
    def available_capacity(self) -> float:
        """Get available capacity."""
        return max(0.0, self.max_load - self.current_load)


class TaskAllocator:
    """
    Allocates inspection tasks to multiple agents.
    
    Supports multiple allocation strategies:
    - Greedy: Assign tasks to nearest capable agent
    - Auction: Agents bid on tasks
    - Hungarian: Optimal assignment
    - Market-based: Dynamic pricing and allocation
    """
    
    def __init__(
        self,
        agents: List[AgentCapabilities],
        method: AllocationMethod = AllocationMethod.AUCTION
    ):
        """
        Initialize task allocator.
        
        Args:
            agents: List of agent capabilities
            method: Allocation method to use
        """
        self.agents = {a.agent_id: a for a in agents}
        self.method = method
        
        # Task tracking
        self.tasks: Dict[str, Task] = {}
        self.completed_tasks: Set[str] = set()
        
        # Allocation tracking
        self.agent_tasks: Dict[str, List[str]] = defaultdict(list)
        self.task_history: List[Dict] = []
        
        # Market-based allocation state
        self.task_prices: Dict[str, float] = {}
        self.agent_budgets: Dict[str, float] = {}
    
    def add_tasks(self, tasks: List[Task]) -> None:
        """Add tasks to be allocated."""
        for task in tasks:
            self.tasks[task.task_id] = task
    
    def update_agent_position(
        self,
        agent_id: str,
        position: str
    ) -> None:
        """Update an agent's current position."""
        if agent_id in self.agents:
            self.agents[agent_id].current_position = position
    
    def allocate(
        self,
        dsg,  # SharedDynamicSceneGraph for distances
        room_probabilities: Dict[str, float] = None
    ) -> Dict[str, List[Task]]:
        """
        Allocate pending tasks to agents.
        
        Args:
            dsg: Scene graph for computing distances
            room_probabilities: Optional probabilities for prioritization
            
        Returns:
            Dictionary mapping agent_id to list of assigned tasks
        """
        if self.method == AllocationMethod.GREEDY:
            return self._greedy_allocation(dsg, room_probabilities)
        elif self.method == AllocationMethod.AUCTION:
            return self._auction_allocation(dsg, room_probabilities)
        elif self.method == AllocationMethod.HUNGARIAN:
            return self._hungarian_allocation(dsg, room_probabilities)
        elif self.method == AllocationMethod.MARKET_BASED:
            return self._market_based_allocation(dsg, room_probabilities)
        else:
            return self._greedy_allocation(dsg, room_probabilities)
    
    def _greedy_allocation(
        self,
        dsg,
        room_probabilities: Dict[str, float] = None
    ) -> Dict[str, List[Task]]:
        """
        Greedy task allocation.
        
        Assigns each task to the nearest capable agent.
        """
        allocation: Dict[str, List[Task]] = {aid: [] for aid in self.agents}
        
        # Get pending tasks sorted by priority
        pending_tasks = [
            t for t in self.tasks.values()
            if t.status == "pending"
        ]
        pending_tasks.sort(key=lambda t: -t.priority)
        
        for task in pending_tasks:
            best_agent = None
            best_cost = float('inf')
            
            for agent_id, agent in self.agents.items():
                # Check capability and capacity
                if not agent.can_perform(task):
                    continue
                if agent.available_capacity() < task.estimated_cost:
                    continue
                
                # Compute cost (distance)
                if agent.current_position:
                    cost = dsg.get_distance(agent.current_position, task.target_room)
                else:
                    cost = 0
                
                cost = cost / agent.speed  # Adjust for agent speed
                
                if cost < best_cost:
                    best_cost = cost
                    best_agent = agent_id
            
            if best_agent:
                task.assigned_agent = best_agent
                task.status = "assigned"
                allocation[best_agent].append(task)
                self.agents[best_agent].current_load += task.estimated_cost
                self.agent_tasks[best_agent].append(task.task_id)
        
        return allocation
    
    def _auction_allocation(
        self,
        dsg,
        room_probabilities: Dict[str, float] = None
    ) -> Dict[str, List[Task]]:
        """
        Auction-based task allocation.
        
        Agents bid on tasks, highest bidder wins.
        Bid = task_value - cost_to_complete
        """
        allocation: Dict[str, List[Task]] = {aid: [] for aid in self.agents}
        
        # Get pending tasks
        pending_tasks = [
            t for t in self.tasks.values()
            if t.status == "pending"
        ]
        
        for task in pending_tasks:
            # Collect bids from all capable agents
            bids: List[Tuple[str, float]] = []
            
            for agent_id, agent in self.agents.items():
                if not agent.can_perform(task):
                    continue
                if agent.available_capacity() < task.estimated_cost:
                    continue
                
                # Compute bid
                # Higher probability rooms have higher value
                task_value = task.priority
                if room_probabilities and task.target_room in room_probabilities:
                    task_value *= (1 + room_probabilities[task.target_room])
                
                # Cost to reach
                if agent.current_position:
                    cost = dsg.get_distance(agent.current_position, task.target_room)
                else:
                    cost = 0
                
                cost = cost / agent.speed
                
                # Bid
                bid = task_value - 0.5 * cost  # Weight cost less than value
                
                # Workload balance factor
                load_factor = 1.0 - (agent.current_load / agent.max_load)
                bid *= (0.5 + 0.5 * load_factor)
                
                bids.append((agent_id, bid))
            
            if bids:
                # Highest bidder wins
                winner = max(bids, key=lambda x: x[1])[0]
                
                task.assigned_agent = winner
                task.status = "assigned"
                allocation[winner].append(task)
                self.agents[winner].current_load += task.estimated_cost
                self.agent_tasks[winner].append(task.task_id)
        
        return allocation
    
    def _hungarian_allocation(
        self,
        dsg,
        room_probabilities: Dict[str, float] = None
    ) -> Dict[str, List[Task]]:
        """
        Optimal assignment using Hungarian algorithm.
        
        Minimizes total cost of assignment.
        """
        from scipy.optimize import linear_sum_assignment
        
        allocation: Dict[str, List[Task]] = {aid: [] for aid in self.agents}
        
        # Get pending tasks
        pending_tasks = [
            t for t in self.tasks.values()
            if t.status == "pending"
        ]
        
        if not pending_tasks:
            return allocation
        
        agent_list = list(self.agents.keys())
        n_agents = len(agent_list)
        n_tasks = len(pending_tasks)
        
        # Create cost matrix
        # Pad to make square if needed
        size = max(n_agents, n_tasks)
        cost_matrix = np.full((size, size), 1e6)
        
        for i, agent_id in enumerate(agent_list):
            agent = self.agents[agent_id]
            
            for j, task in enumerate(pending_tasks):
                if not agent.can_perform(task):
                    continue
                if agent.available_capacity() < task.estimated_cost:
                    continue
                
                # Cost = distance - value
                if agent.current_position:
                    distance = dsg.get_distance(agent.current_position, task.target_room)
                else:
                    distance = 0
                
                value = task.priority
                if room_probabilities and task.target_room in room_probabilities:
                    value *= (1 + room_probabilities[task.target_room])
                
                cost_matrix[i, j] = distance / agent.speed - value
        
        # Solve assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Apply assignment
        for i, j in zip(row_ind, col_ind):
            if i < n_agents and j < n_tasks:
                if cost_matrix[i, j] < 1e5:  # Valid assignment
                    agent_id = agent_list[i]
                    task = pending_tasks[j]
                    
                    task.assigned_agent = agent_id
                    task.status = "assigned"
                    allocation[agent_id].append(task)
                    self.agents[agent_id].current_load += task.estimated_cost
                    self.agent_tasks[agent_id].append(task.task_id)
        
        return allocation
    
    def _market_based_allocation(
        self,
        dsg,
        room_probabilities: Dict[str, float] = None
    ) -> Dict[str, List[Task]]:
        """
        Market-based task allocation with dynamic pricing.
        
        Tasks have prices, agents have budgets.
        Prices adjust based on demand.
        """
        allocation: Dict[str, List[Task]] = {aid: [] for aid in self.agents}
        
        # Initialize budgets and prices
        for agent_id in self.agents:
            if agent_id not in self.agent_budgets:
                self.agent_budgets[agent_id] = 10.0
        
        pending_tasks = [
            t for t in self.tasks.values()
            if t.status == "pending"
        ]
        
        for task in pending_tasks:
            if task.task_id not in self.task_prices:
                # Initial price based on priority and probability
                price = task.priority
                if room_probabilities and task.target_room in room_probabilities:
                    price *= (1 + room_probabilities[task.target_room])
                self.task_prices[task.task_id] = price
        
        # Multiple rounds of bidding
        max_rounds = 5
        for round_num in range(max_rounds):
            # Each agent selects preferred task they can afford
            preferences: Dict[str, Tuple[str, float]] = {}  # agent -> (task_id, utility)
            
            for agent_id, agent in self.agents.items():
                if agent.available_capacity() <= 0:
                    continue
                
                best_task = None
                best_utility = float('-inf')
                
                for task in pending_tasks:
                    if task.status != "pending":
                        continue
                    if not agent.can_perform(task):
                        continue
                    
                    price = self.task_prices[task.task_id]
                    if price > self.agent_budgets[agent_id]:
                        continue
                    
                    # Utility = value - price - cost
                    value = task.priority
                    if room_probabilities and task.target_room in room_probabilities:
                        value *= (1 + room_probabilities[task.target_room])
                    
                    if agent.current_position:
                        cost = dsg.get_distance(agent.current_position, task.target_room)
                    else:
                        cost = 0
                    
                    utility = value - price - 0.1 * cost
                    
                    if utility > best_utility:
                        best_utility = utility
                        best_task = task.task_id
                
                if best_task:
                    preferences[agent_id] = (best_task, best_utility)
            
            # Find tasks with multiple interested agents
            task_demand: Dict[str, List[str]] = defaultdict(list)
            for agent_id, (task_id, _) in preferences.items():
                task_demand[task_id].append(agent_id)
            
            # Adjust prices and assign
            for task_id, interested_agents in task_demand.items():
                if len(interested_agents) > 1:
                    # Increase price
                    self.task_prices[task_id] *= 1.1
                elif len(interested_agents) == 1:
                    # Assign to single interested agent
                    agent_id = interested_agents[0]
                    task = self.tasks[task_id]
                    
                    task.assigned_agent = agent_id
                    task.status = "assigned"
                    allocation[agent_id].append(task)
                    
                    self.agent_budgets[agent_id] -= self.task_prices[task_id]
                    self.agents[agent_id].current_load += task.estimated_cost
                    self.agent_tasks[agent_id].append(task_id)
            
            # Decrease prices of tasks with no demand
            for task in pending_tasks:
                if task.status == "pending" and task.task_id not in task_demand:
                    self.task_prices[task.task_id] *= 0.9
        
        return allocation
    
    def report_task_complete(
        self,
        task_id: str,
        agent_id: str,
        success: bool,
        result: Any = None
    ) -> None:
        """Report task completion."""
        if task_id not in self.tasks:
            return
        
        task = self.tasks[task_id]
        task.status = "completed" if success else "failed"
        
        if agent_id in self.agents:
            self.agents[agent_id].current_load -= task.estimated_cost
        
        self.completed_tasks.add(task_id)
        
        self.task_history.append({
            "task_id": task_id,
            "agent_id": agent_id,
            "success": success,
            "result": result,
            "timestamp": None  # Would use time.time() in practice
        })
    
    def get_agent_workload(self, agent_id: str) -> float:
        """Get current workload for an agent."""
        if agent_id in self.agents:
            return self.agents[agent_id].current_load
        return 0.0
    
    def rebalance(
        self,
        dsg,
        room_probabilities: Dict[str, float] = None
    ) -> Dict[str, List[Task]]:
        """
        Rebalance task assignments.
        
        Called when workloads become unbalanced or new information
        changes priorities.
        """
        # Unassign all pending/assigned tasks
        for task in self.tasks.values():
            if task.status in ["pending", "assigned"]:
                if task.assigned_agent:
                    self.agents[task.assigned_agent].current_load -= task.estimated_cost
                task.assigned_agent = None
                task.status = "pending"
        
        # Clear agent task lists
        self.agent_tasks = defaultdict(list)
        
        # Re-allocate
        return self.allocate(dsg, room_probabilities)
    
    def generate_room_tasks(
        self,
        rooms: List[str],
        room_probabilities: Dict[str, float],
        search_costs: Dict[str, float] = None
    ) -> List[Task]:
        """
        Generate inspection tasks for rooms.
        
        Args:
            rooms: List of room IDs
            room_probabilities: Probability of target in each room
            search_costs: Cost to search each room
            
        Returns:
            List of generated tasks
        """
        tasks = []
        
        for room_id in rooms:
            priority = room_probabilities.get(room_id, 0.1)
            cost = search_costs.get(room_id, 1.0) if search_costs else 1.0
            
            task = Task(
                task_id=f"search_{room_id}",
                target_room=room_id,
                priority=priority,
                estimated_cost=cost
            )
            tasks.append(task)
        
        return tasks
