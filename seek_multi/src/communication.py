"""
Communication Protocol for Multi-Agent SEEK

Implements message passing, belief sharing, and coordination protocols
between multiple inspection agents.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from collections import deque
import time
import threading
import json


class MessageType(Enum):
    """Types of messages exchanged between agents."""
    # Belief and observation sharing
    BELIEF_UPDATE = "belief_update"
    OBSERVATION = "observation"
    OBJECT_DETECTION = "object_detection"
    
    # Coordination messages
    TASK_ASSIGNMENT = "task_assignment"
    TASK_COMPLETION = "task_completion"
    TASK_REQUEST = "task_request"
    
    # Planning coordination
    INTENTION = "intention"
    PLAN_UPDATE = "plan_update"
    CONFLICT_RESOLUTION = "conflict_resolution"
    
    # Status messages
    HEARTBEAT = "heartbeat"
    STATUS_UPDATE = "status_update"
    EMERGENCY = "emergency"
    
    # RSN updates
    RSN_UPDATE = "rsn_update"
    RSN_QUERY = "rsn_query"
    RSN_RESPONSE = "rsn_response"
    
    # DSG updates
    DSG_UPDATE = "dsg_update"
    DSG_MERGE_REQUEST = "dsg_merge_request"
    DSG_MERGE_RESPONSE = "dsg_merge_response"


@dataclass
class Message:
    """
    Message structure for inter-agent communication.
    
    Attributes:
        msg_type: Type of message
        sender_id: ID of the sending agent
        receiver_id: ID of the receiving agent (None for broadcast)
        timestamp: Time when message was created
        payload: Message content
        priority: Message priority (higher = more urgent)
        msg_id: Unique message identifier
    """
    msg_type: MessageType
    sender_id: str
    receiver_id: Optional[str]
    timestamp: float
    payload: Dict[str, Any]
    priority: int = 0
    msg_id: str = ""
    
    def __post_init__(self):
        if not self.msg_id:
            self.msg_id = f"{self.sender_id}_{self.timestamp}_{id(self)}"
    
    def to_dict(self) -> Dict:
        """Convert message to dictionary for serialization."""
        return {
            "msg_type": self.msg_type.value,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "priority": self.priority,
            "msg_id": self.msg_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Message":
        """Create message from dictionary."""
        return cls(
            msg_type=MessageType(data["msg_type"]),
            sender_id=data["sender_id"],
            receiver_id=data["receiver_id"],
            timestamp=data["timestamp"],
            payload=data["payload"],
            priority=data.get("priority", 0),
            msg_id=data.get("msg_id", "")
        )


@dataclass
class BeliefUpdate:
    """
    Structure for sharing belief updates between agents.
    
    Attributes:
        room_id: ID of the room being updated
        object_class: Target object class
        probability: Updated probability of finding object
        confidence: Confidence in this estimate
        evidence_type: Type of evidence (observation, inference, shared)
    """
    room_id: str
    object_class: str
    probability: float
    confidence: float
    evidence_type: str
    observations: List[Dict] = field(default_factory=list)


class CommunicationProtocol:
    """
    Manages communication between multiple inspection agents.
    
    Implements:
    - Message passing with priority queues
    - Belief fusion and consensus
    - Bandwidth-aware communication
    - Fault-tolerant message delivery
    """
    
    def __init__(
        self,
        agent_id: str,
        max_queue_size: int = 1000,
        communication_range: float = float('inf'),
        bandwidth_limit: float = float('inf')
    ):
        """
        Initialize communication protocol.
        
        Args:
            agent_id: Unique identifier for this agent
            max_queue_size: Maximum messages in queue
            communication_range: Maximum communication range (meters)
            bandwidth_limit: Maximum bandwidth (messages/second)
        """
        self.agent_id = agent_id
        self.max_queue_size = max_queue_size
        self.communication_range = communication_range
        self.bandwidth_limit = bandwidth_limit
        
        # Message queues
        self.incoming_queue: deque = deque(maxlen=max_queue_size)
        self.outgoing_queue: deque = deque(maxlen=max_queue_size)
        
        # Connected agents and their positions
        self.connected_agents: Dict[str, Dict] = {}
        
        # Message history for deduplication
        self.received_msg_ids: set = set()
        self.max_history_size = 10000
        
        # Callbacks for message handling
        self.message_handlers: Dict[MessageType, List[callable]] = {
            msg_type: [] for msg_type in MessageType
        }
        
        # Communication statistics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_dropped": 0,
            "bytes_sent": 0,
            "bytes_received": 0
        }
        
        # Threading for async communication
        self._running = False
        self._lock = threading.Lock()
    
    def register_handler(
        self,
        msg_type: MessageType,
        handler: callable
    ) -> None:
        """Register a callback handler for a message type."""
        self.message_handlers[msg_type].append(handler)
    
    def register_agent(
        self,
        agent_id: str,
        position: np.ndarray,
        capabilities: Dict[str, Any] = None
    ) -> None:
        """Register a new agent for communication."""
        self.connected_agents[agent_id] = {
            "position": position,
            "capabilities": capabilities or {},
            "last_heartbeat": time.time(),
            "status": "active"
        }
    
    def update_agent_position(
        self,
        agent_id: str,
        position: np.ndarray
    ) -> None:
        """Update the position of a connected agent."""
        if agent_id in self.connected_agents:
            self.connected_agents[agent_id]["position"] = position
    
    def is_in_range(
        self,
        agent_id: str,
        my_position: np.ndarray
    ) -> bool:
        """Check if an agent is within communication range."""
        if agent_id not in self.connected_agents:
            return False
        
        if self.communication_range == float('inf'):
            return True
        
        other_pos = self.connected_agents[agent_id]["position"]
        distance = np.linalg.norm(my_position - other_pos)
        return distance <= self.communication_range
    
    def send_message(
        self,
        msg_type: MessageType,
        payload: Dict[str, Any],
        receiver_id: Optional[str] = None,
        priority: int = 0
    ) -> Message:
        """
        Send a message to another agent or broadcast.
        
        Args:
            msg_type: Type of message
            payload: Message content
            receiver_id: Target agent (None for broadcast)
            priority: Message priority
            
        Returns:
            The created message
        """
        message = Message(
            msg_type=msg_type,
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            timestamp=time.time(),
            payload=payload,
            priority=priority
        )
        
        with self._lock:
            self.outgoing_queue.append(message)
            self.stats["messages_sent"] += 1
        
        return message
    
    def broadcast_belief_update(
        self,
        belief_update: BeliefUpdate
    ) -> Message:
        """Broadcast a belief update to all connected agents."""
        payload = {
            "room_id": belief_update.room_id,
            "object_class": belief_update.object_class,
            "probability": belief_update.probability,
            "confidence": belief_update.confidence,
            "evidence_type": belief_update.evidence_type,
            "observations": belief_update.observations
        }
        return self.send_message(
            MessageType.BELIEF_UPDATE,
            payload,
            priority=1
        )
    
    def broadcast_object_detection(
        self,
        object_class: str,
        position: np.ndarray,
        confidence: float,
        room_id: str
    ) -> Message:
        """Broadcast an object detection to all agents."""
        payload = {
            "object_class": object_class,
            "position": position.tolist(),
            "confidence": confidence,
            "room_id": room_id,
            "detector_position": None  # Will be filled by sender
        }
        return self.send_message(
            MessageType.OBJECT_DETECTION,
            payload,
            priority=2  # High priority for detections
        )
    
    def send_intention(
        self,
        target_room: str,
        action_type: str,
        estimated_duration: float
    ) -> Message:
        """Broadcast the agent's intended action for coordination."""
        payload = {
            "target_room": target_room,
            "action_type": action_type,
            "estimated_duration": estimated_duration,
            "start_time": time.time()
        }
        return self.send_message(
            MessageType.INTENTION,
            payload,
            priority=1
        )
    
    def receive_messages(self) -> List[Message]:
        """Get all pending incoming messages."""
        messages = []
        with self._lock:
            while self.incoming_queue:
                messages.append(self.incoming_queue.popleft())
        return messages
    
    def process_incoming(self, message: Message) -> bool:
        """
        Process an incoming message.
        
        Args:
            message: The received message
            
        Returns:
            True if message was processed, False if duplicate
        """
        # Check for duplicates
        if message.msg_id in self.received_msg_ids:
            return False
        
        # Add to history
        self.received_msg_ids.add(message.msg_id)
        if len(self.received_msg_ids) > self.max_history_size:
            # Remove oldest entries
            self.received_msg_ids = set(
                list(self.received_msg_ids)[-self.max_history_size:]
            )
        
        # Check if message is for us
        if (message.receiver_id is not None and 
            message.receiver_id != self.agent_id):
            return False
        
        # Add to incoming queue
        with self._lock:
            self.incoming_queue.append(message)
            self.stats["messages_received"] += 1
        
        # Call registered handlers
        for handler in self.message_handlers[message.msg_type]:
            try:
                handler(message)
            except Exception as e:
                print(f"Error in message handler: {e}")
        
        return True
    
    def get_outgoing_messages(
        self,
        my_position: np.ndarray
    ) -> Dict[str, List[Message]]:
        """
        Get messages to send, organized by recipient.
        
        Args:
            my_position: Current position of this agent
            
        Returns:
            Dictionary mapping agent IDs to messages
        """
        messages_by_recipient: Dict[str, List[Message]] = {}
        
        with self._lock:
            while self.outgoing_queue:
                message = self.outgoing_queue.popleft()
                
                if message.receiver_id:
                    # Direct message
                    if self.is_in_range(message.receiver_id, my_position):
                        if message.receiver_id not in messages_by_recipient:
                            messages_by_recipient[message.receiver_id] = []
                        messages_by_recipient[message.receiver_id].append(message)
                else:
                    # Broadcast
                    for agent_id in self.connected_agents:
                        if self.is_in_range(agent_id, my_position):
                            if agent_id not in messages_by_recipient:
                                messages_by_recipient[agent_id] = []
                            messages_by_recipient[agent_id].append(message)
        
        return messages_by_recipient
    
    def send_heartbeat(self, status: Dict[str, Any]) -> Message:
        """Send a heartbeat message with current status."""
        return self.send_message(
            MessageType.HEARTBEAT,
            {"status": status},
            priority=0
        )
    
    def check_agent_status(self, timeout: float = 5.0) -> List[str]:
        """Check for agents that haven't sent heartbeats recently."""
        current_time = time.time()
        inactive_agents = []
        
        for agent_id, info in self.connected_agents.items():
            if current_time - info["last_heartbeat"] > timeout:
                info["status"] = "inactive"
                inactive_agents.append(agent_id)
        
        return inactive_agents


class ConsensusProtocol:
    """
    Implements consensus algorithms for distributed belief fusion.
    
    Uses average consensus for merging probability distributions
    from multiple agents.
    """
    
    def __init__(self, num_agents: int, convergence_threshold: float = 0.01):
        """
        Initialize consensus protocol.
        
        Args:
            num_agents: Number of agents in the system
            convergence_threshold: Threshold for convergence detection
        """
        self.num_agents = num_agents
        self.convergence_threshold = convergence_threshold
        self.iteration = 0
    
    def fuse_beliefs(
        self,
        local_belief: np.ndarray,
        neighbor_beliefs: List[Tuple[str, np.ndarray, float]],
        weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Fuse beliefs from multiple agents using weighted average.
        
        Args:
            local_belief: This agent's belief vector
            neighbor_beliefs: List of (agent_id, belief, confidence) tuples
            weights: Optional custom weights per agent
            
        Returns:
            Fused belief vector
        """
        if not neighbor_beliefs:
            return local_belief
        
        # Default equal weights
        if weights is None:
            total_agents = len(neighbor_beliefs) + 1
            weights = {aid: 1.0 / total_agents for aid, _, _ in neighbor_beliefs}
            local_weight = 1.0 / total_agents
        else:
            local_weight = weights.get("self", 1.0 / (len(neighbor_beliefs) + 1))
        
        # Weighted average
        fused = local_weight * local_belief
        
        for agent_id, belief, confidence in neighbor_beliefs:
            agent_weight = weights.get(agent_id, 1.0 / (len(neighbor_beliefs) + 1))
            # Weight by confidence as well
            fused += agent_weight * confidence * belief
        
        # Normalize
        fused = fused / (fused.sum() + 1e-10)
        
        return fused
    
    def bayesian_fusion(
        self,
        prior: np.ndarray,
        observations: List[Tuple[np.ndarray, float]]
    ) -> np.ndarray:
        """
        Bayesian belief fusion for multiple observations.
        
        Args:
            prior: Prior probability distribution
            observations: List of (likelihood, confidence) tuples
            
        Returns:
            Posterior probability distribution
        """
        posterior = prior.copy()
        
        for likelihood, confidence in observations:
            # Weight likelihood by confidence
            weighted_likelihood = confidence * likelihood + (1 - confidence) * np.ones_like(likelihood) / len(likelihood)
            posterior = posterior * weighted_likelihood
            posterior = posterior / (posterior.sum() + 1e-10)
        
        return posterior
    
    def covariance_intersection(
        self,
        estimates: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Covariance intersection for fusing estimates with unknown correlation.
        
        Args:
            estimates: List of (mean, covariance) tuples
            
        Returns:
            Fused (mean, covariance) tuple
        """
        if len(estimates) == 1:
            return estimates[0]
        
        # Optimize omega for minimum determinant
        # Simplified: use equal weights
        n = len(estimates)
        omega = [1.0 / n] * n
        
        # Fused covariance
        P_inv = sum(w * np.linalg.inv(P) for w, (_, P) in zip(omega, estimates))
        P_fused = np.linalg.inv(P_inv)
        
        # Fused mean
        mean_fused = P_fused @ sum(
            w * np.linalg.inv(P) @ x 
            for w, (x, P) in zip(omega, estimates)
        )
        
        return mean_fused, P_fused
