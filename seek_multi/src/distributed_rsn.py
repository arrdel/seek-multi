"""
Distributed Relational Semantic Network for Multi-Agent SEEK

Implements a distributed RSN that can share and fuse semantic
knowledge across multiple agents.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
import time


# Room types commonly found in indoor environments
ROOM_TYPES = [
    "kitchen", "bathroom", "bedroom", "living_room", "dining_room",
    "office", "hallway", "entrance", "garage", "basement",
    "attic", "closet", "laundry_room", "storage", "balcony",
    "patio", "lobby", "conference_room", "break_room", "restroom",
    "elevator", "stairwell", "reception", "waiting_room", "lab",
    "server_room", "utility_room"
]


class SemanticMLP(nn.Module):
    """
    Multi-layer perceptron for predicting object-room probabilities.
    """
    
    def __init__(
        self,
        input_dim: int = 384,  # BGE-small embedding dimension
        hidden_dims: List[int] = [256, 128, 64],
        num_room_types: int = 27,
        dropout: float = 0.1
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # Output heads
        self.room_type_head = nn.Sequential(
            nn.Linear(prev_dim, num_room_types),
            nn.Softmax(dim=-1)
        )
        
        self.visibility_head = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Text embedding of shape (batch, input_dim)
            
        Returns:
            Tuple of (room_type_probs, visibility_prob)
        """
        features = self.backbone(x)
        room_probs = self.room_type_head(features)
        visibility = self.visibility_head(features)
        return room_probs, visibility


@dataclass
class ObjectBeliefState:
    """
    Tracks belief state for a target object across rooms.
    
    Attributes:
        object_class: Target object class
        room_probabilities: Probability of object in each room
        last_updated: Timestamp of last update
        observations: History of observations
        confidence: Overall confidence in the belief
    """
    object_class: str
    room_probabilities: Dict[str, float]
    last_updated: float = 0.0
    observations: List[Dict] = field(default_factory=list)
    confidence: float = 0.5
    
    def to_array(self, room_ids: List[str]) -> np.ndarray:
        """Convert to numpy array in specific room order."""
        return np.array([self.room_probabilities.get(rid, 0.0) for rid in room_ids])
    
    @classmethod
    def from_array(
        cls,
        object_class: str,
        probs: np.ndarray,
        room_ids: List[str]
    ) -> "ObjectBeliefState":
        """Create from numpy array."""
        return cls(
            object_class=object_class,
            room_probabilities={rid: p for rid, p in zip(room_ids, probs)},
            last_updated=time.time()
        )


class DistributedRSN:
    """
    Distributed Relational Semantic Network for multi-agent systems.
    
    Features:
    - Local belief maintenance
    - Belief fusion from other agents
    - Bayesian updates from observations
    - Knowledge distillation from LLMs
    """
    
    def __init__(
        self,
        agent_id: str,
        room_types: List[str] = None,
        embedding_model: str = "bge-small",
        device: str = "cpu"
    ):
        """
        Initialize the distributed RSN.
        
        Args:
            agent_id: Unique agent identifier
            room_types: List of room type names
            embedding_model: Name of text embedding model
            device: Computation device
        """
        self.agent_id = agent_id
        self.room_types = room_types or ROOM_TYPES
        self.device = device
        
        # Initialize MLP
        self.mlp = SemanticMLP(num_room_types=len(self.room_types))
        self.mlp.to(device)
        
        # Room mappings
        self.room_id_to_type: Dict[str, str] = {}
        self.room_type_to_ids: Dict[str, List[str]] = defaultdict(list)
        
        # Belief states per target object
        self.beliefs: Dict[str, ObjectBeliefState] = {}
        
        # Prior knowledge (from training)
        self.prior_room_probs: Dict[str, np.ndarray] = {}
        
        # Text embedding cache
        self.embedding_cache: Dict[str, np.ndarray] = {}
        
        # Update history for merging
        self.update_history: List[Dict] = []
        self.version = 0
        
        # Observation model parameters
        self.detection_true_positive_rate = 0.9
        self.detection_false_positive_rate = 0.1
    
    def register_rooms(
        self,
        rooms: List[Tuple[str, str]]
    ) -> None:
        """
        Register rooms from the DSG.
        
        Args:
            rooms: List of (room_id, room_type) tuples
        """
        self.room_id_to_type.clear()
        self.room_type_to_ids.clear()
        
        for room_id, room_type in rooms:
            self.room_id_to_type[room_id] = room_type
            self.room_type_to_ids[room_type].append(room_id)
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get text embedding for an object class."""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        # Simplified: use random embedding for demo
        # In practice, use actual embedding model
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        
        self.embedding_cache[text] = embedding
        return embedding
    
    def predict_room_type_probs(
        self,
        object_class: str
    ) -> Tuple[np.ndarray, float]:
        """
        Predict probability of finding object in each room type.
        
        Args:
            object_class: Target object class name
            
        Returns:
            Tuple of (room_type_probabilities, visibility_probability)
        """
        # Get embedding
        embedding = self.get_text_embedding(object_class)
        embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(self.device)
        
        # Forward pass
        self.mlp.eval()
        with torch.no_grad():
            room_probs, visibility = self.mlp(embedding_tensor)
        
        return room_probs.cpu().numpy()[0], visibility.cpu().numpy()[0, 0]
    
    def predict_room_probabilities(
        self,
        object_class: str
    ) -> Dict[str, float]:
        """
        Predict probability of finding object in each specific room.
        
        Maps room type probabilities to specific room instances.
        
        Args:
            object_class: Target object class
            
        Returns:
            Dictionary mapping room_id to probability
        """
        room_type_probs, _ = self.predict_room_type_probs(object_class)
        
        room_probs = {}
        
        for room_id, room_type in self.room_id_to_type.items():
            try:
                type_idx = self.room_types.index(room_type)
                type_prob = room_type_probs[type_idx]
            except (ValueError, IndexError):
                type_prob = 1.0 / len(self.room_types)  # Uniform prior
            
            # Distribute probability among rooms of same type
            num_rooms_of_type = len(self.room_type_to_ids.get(room_type, [room_id]))
            room_probs[room_id] = type_prob / num_rooms_of_type
        
        # Normalize
        total = sum(room_probs.values())
        if total > 0:
            room_probs = {k: v / total for k, v in room_probs.items()}
        
        return room_probs
    
    def initialize_belief(
        self,
        object_class: str
    ) -> ObjectBeliefState:
        """
        Initialize belief state for a target object.
        
        Args:
            object_class: Target object class
            
        Returns:
            Initialized belief state
        """
        room_probs = self.predict_room_probabilities(object_class)
        
        belief = ObjectBeliefState(
            object_class=object_class,
            room_probabilities=room_probs,
            last_updated=time.time(),
            confidence=0.5  # Initial moderate confidence
        )
        
        self.beliefs[object_class] = belief
        return belief
    
    def get_belief(self, object_class: str) -> ObjectBeliefState:
        """Get current belief for an object, initializing if needed."""
        if object_class not in self.beliefs:
            return self.initialize_belief(object_class)
        return self.beliefs[object_class]
    
    def update_belief_from_observation(
        self,
        object_class: str,
        room_id: str,
        detected: bool,
        confidence: float = 1.0,
        thorough_search: bool = False
    ) -> ObjectBeliefState:
        """
        Update belief based on an observation.
        
        Uses Bayesian update with observation model.
        
        Args:
            object_class: Target object class
            room_id: Room where observation was made
            detected: Whether object was detected
            confidence: Detection confidence
            thorough_search: Whether room was thoroughly searched
            
        Returns:
            Updated belief state
        """
        belief = self.get_belief(object_class)
        
        # Get current prior
        prior = belief.room_probabilities.copy()
        
        # Observation model
        if thorough_search:
            p_detect_if_present = self.detection_true_positive_rate
            p_detect_if_absent = self.detection_false_positive_rate
        else:
            # Partial observation has lower accuracy
            p_detect_if_present = 0.5 * self.detection_true_positive_rate
            p_detect_if_absent = 0.5 * self.detection_false_positive_rate
        
        # Bayesian update for the observed room
        if detected:
            # Increase probability for this room
            likelihood_ratio = p_detect_if_present / p_detect_if_absent
            prior[room_id] = prior.get(room_id, 0.01) * likelihood_ratio * confidence
        else:
            # Decrease probability for this room
            likelihood_ratio = (1 - p_detect_if_present) / (1 - p_detect_if_absent)
            prior[room_id] = prior.get(room_id, 0.01) * likelihood_ratio
        
        # Normalize
        total = sum(prior.values())
        if total > 0:
            prior = {k: v / total for k, v in prior.items()}
        
        # Update belief
        belief.room_probabilities = prior
        belief.last_updated = time.time()
        belief.observations.append({
            "room_id": room_id,
            "detected": detected,
            "confidence": confidence,
            "thorough_search": thorough_search,
            "timestamp": time.time(),
            "agent_id": self.agent_id
        })
        
        # Update confidence based on number of observations
        belief.confidence = min(0.95, 0.5 + 0.1 * len(belief.observations))
        
        # Log update
        self._log_update(object_class, room_id, detected, confidence)
        
        return belief
    
    def fuse_belief_from_agent(
        self,
        object_class: str,
        other_belief: Dict[str, float],
        other_confidence: float,
        other_agent_id: str
    ) -> ObjectBeliefState:
        """
        Fuse belief from another agent.
        
        Uses weighted average based on confidence levels.
        
        Args:
            object_class: Target object class
            other_belief: Other agent's room probabilities
            other_confidence: Other agent's confidence level
            other_agent_id: ID of the other agent
            
        Returns:
            Fused belief state
        """
        local_belief = self.get_belief(object_class)
        
        # Weight by confidence
        local_weight = local_belief.confidence
        other_weight = other_confidence
        total_weight = local_weight + other_weight
        
        # Fuse probabilities
        fused_probs = {}
        all_rooms = set(local_belief.room_probabilities.keys()) | set(other_belief.keys())
        
        for room_id in all_rooms:
            local_prob = local_belief.room_probabilities.get(room_id, 0.0)
            other_prob = other_belief.get(room_id, 0.0)
            fused_probs[room_id] = (
                local_weight * local_prob + other_weight * other_prob
            ) / total_weight
        
        # Normalize
        total = sum(fused_probs.values())
        if total > 0:
            fused_probs = {k: v / total for k, v in fused_probs.items()}
        
        # Update belief
        local_belief.room_probabilities = fused_probs
        local_belief.confidence = min(0.95, (local_weight + other_weight) / 2 + 0.1)
        local_belief.last_updated = time.time()
        
        return local_belief
    
    def get_top_rooms(
        self,
        object_class: str,
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """Get top k rooms most likely to contain the object."""
        belief = self.get_belief(object_class)
        sorted_rooms = sorted(
            belief.room_probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_rooms[:k]
    
    def get_entropy(self, object_class: str) -> float:
        """Compute entropy of belief distribution."""
        belief = self.get_belief(object_class)
        probs = np.array(list(belief.room_probabilities.values()))
        probs = probs[probs > 0]  # Avoid log(0)
        return -np.sum(probs * np.log(probs + 1e-10))
    
    def _log_update(
        self,
        object_class: str,
        room_id: str,
        detected: bool,
        confidence: float
    ) -> None:
        """Log an update for sharing with other agents."""
        self.update_history.append({
            "type": "observation",
            "object_class": object_class,
            "room_id": room_id,
            "detected": detected,
            "confidence": confidence,
            "timestamp": time.time(),
            "agent_id": self.agent_id,
            "version": self.version
        })
        self.version += 1
    
    def get_updates_since(self, version: int) -> List[Dict]:
        """Get updates since a specific version."""
        return [u for u in self.update_history if u.get("version", 0) > version]
    
    def apply_remote_update(self, update: Dict) -> None:
        """Apply an update from another agent."""
        if update["type"] == "observation":
            self.update_belief_from_observation(
                object_class=update["object_class"],
                room_id=update["room_id"],
                detected=update["detected"],
                confidence=update["confidence"] * 0.9  # Slightly discount remote observations
            )
    
    def save_model(self, path: str) -> None:
        """Save MLP weights."""
        torch.save(self.mlp.state_dict(), path)
    
    def load_model(self, path: str) -> None:
        """Load MLP weights."""
        self.mlp.load_state_dict(torch.load(path, map_location=self.device))
    
    def train_from_llm_data(
        self,
        training_data: List[Tuple[str, Dict[str, float], float]],
        epochs: int = 100,
        learning_rate: float = 0.001
    ) -> None:
        """
        Train MLP from LLM-generated data.
        
        Args:
            training_data: List of (object_name, room_type_probs, visibility) tuples
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        self.mlp.train()
        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for object_name, room_probs, visibility in training_data:
                # Get embedding
                embedding = self.get_text_embedding(object_name)
                embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(self.device)
                
                # Create target tensors
                room_target = torch.zeros(len(self.room_types))
                for room_type, prob in room_probs.items():
                    if room_type in self.room_types:
                        idx = self.room_types.index(room_type)
                        room_target[idx] = prob
                room_target = room_target.unsqueeze(0).to(self.device)
                
                visibility_target = torch.FloatTensor([[visibility]]).to(self.device)
                
                # Forward pass
                room_pred, vis_pred = self.mlp(embedding_tensor)
                
                # Loss
                room_loss = nn.KLDivLoss(reduction='batchmean')(
                    torch.log(room_pred + 1e-10),
                    room_target
                )
                vis_loss = nn.BCELoss()(vis_pred, visibility_target)
                loss = room_loss + 0.5 * vis_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(training_data):.4f}")
        
        self.mlp.eval()
