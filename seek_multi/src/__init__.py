"""
SEEK-Multi: Collaborative Multi-Agent Inspection Framework

This package implements a multi-robot coordination system for semantic-guided
object-goal navigation in inspection tasks.
"""

from .agent import InspectionAgent
from .coordinator import MultiAgentCoordinator
from .communication import CommunicationProtocol, Message, MessageType
from .distributed_rsn import DistributedRSN
from .shared_dsg import SharedDynamicSceneGraph
from .collaborative_planner import CollaborativePlanner
from .task_allocator import TaskAllocator

__version__ = "1.0.0"
__all__ = [
    "InspectionAgent",
    "MultiAgentCoordinator", 
    "CommunicationProtocol",
    "Message",
    "MessageType",
    "DistributedRSN",
    "SharedDynamicSceneGraph",
    "CollaborativePlanner",
    "TaskAllocator",
]
