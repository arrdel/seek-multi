"""
Inspection Agent for Multi-Agent SEEK

Implements an individual inspection agent that coordinates
with other agents for collaborative object-goal navigation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
from enum import Enum
import time
import threading

from .communication import (
    CommunicationProtocol, Message, MessageType, 
    BeliefUpdate, ConsensusProtocol
)
from .shared_dsg import SharedDynamicSceneGraph, NodeType
from .distributed_rsn import DistributedRSN, ObjectBeliefState
from .collaborative_planner import CollaborativePlanner, GlobalAction, ActionType


class AgentState(Enum):
    """States of an inspection agent."""
    IDLE = "idle"
    NAVIGATING = "navigating"
    SEARCHING = "searching"
    INSPECTING = "inspecting"
    COORDINATING = "coordinating"
    ERROR = "error"


@dataclass
class AgentStatus:
    """
    Current status of an agent.
    
    Attributes:
        state: Current agent state
        position: Current 3D position
        current_room: Current room ID
        current_action: Current action being executed
        battery_level: Battery percentage
        error_message: Error message if in ERROR state
    """
    state: AgentState = AgentState.IDLE
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    current_room: Optional[str] = None
    current_action: Optional[GlobalAction] = None
    battery_level: float = 100.0
    error_message: str = ""


class InspectionAgent:
    """
    An autonomous inspection agent for multi-robot object-goal navigation.
    
    Features:
    - Communication with other agents
    - Shared belief maintenance
    - Coordinated planning
    - Local execution of search and navigation
    """
    
    def __init__(
        self,
        agent_id: str,
        dsg: SharedDynamicSceneGraph,
        rsn: DistributedRSN,
        communication_range: float = float('inf'),
        capabilities: List[str] = None
    ):
        """
        Initialize inspection agent.
        
        Args:
            agent_id: Unique agent identifier
            dsg: Shared Dynamic Scene Graph
            rsn: Distributed Relational Semantic Network
            communication_range: Maximum communication range
            capabilities: List of agent capabilities
        """
        self.agent_id = agent_id
        self.dsg = dsg
        self.rsn = rsn
        self.capabilities = set(capabilities or [])
        
        # Agent status
        self.status = AgentStatus()
        
        # Communication
        self.comm = CommunicationProtocol(
            agent_id=agent_id,
            communication_range=communication_range
        )
        self._setup_message_handlers()
        
        # Consensus protocol for belief fusion
        self.consensus = ConsensusProtocol(num_agents=1)  # Will be updated
        
        # Current target
        self.target_object: Optional[str] = None
        
        # Tracking other agents
        self.other_agents: Dict[str, AgentStatus] = {}
        
        # Planner (will be shared or individual)
        self.planner: Optional[CollaborativePlanner] = None
        
        # Local controllers
        self.navigation_controller: Optional[Callable] = None
        self.search_controller: Optional[Callable] = None
        self.inspection_controller: Optional[Callable] = None
        
        # Observations
        self.observations: List[Dict] = []
        
        # Running state
        self._running = False
        self._update_thread: Optional[threading.Thread] = None
    
    def _setup_message_handlers(self) -> None:
        """Setup handlers for different message types."""
        self.comm.register_handler(
            MessageType.BELIEF_UPDATE,
            self._handle_belief_update
        )
        self.comm.register_handler(
            MessageType.OBJECT_DETECTION,
            self._handle_object_detection
        )
        self.comm.register_handler(
            MessageType.INTENTION,
            self._handle_intention
        )
        self.comm.register_handler(
            MessageType.HEARTBEAT,
            self._handle_heartbeat
        )
        self.comm.register_handler(
            MessageType.DSG_UPDATE,
            self._handle_dsg_update
        )
        self.comm.register_handler(
            MessageType.RSN_UPDATE,
            self._handle_rsn_update
        )
    
    def _handle_belief_update(self, message: Message) -> None:
        """Handle incoming belief update from another agent."""
        payload = message.payload
        
        if self.target_object and payload.get("object_class") == self.target_object:
            # Fuse belief
            self.rsn.fuse_belief_from_agent(
                object_class=self.target_object,
                other_belief=payload.get("room_probabilities", {}),
                other_confidence=payload.get("confidence", 0.5),
                other_agent_id=message.sender_id
            )
    
    def _handle_object_detection(self, message: Message) -> None:
        """Handle object detection broadcast from another agent."""
        payload = message.payload
        
        if payload.get("object_class") == self.target_object:
            # Update belief based on detection
            room_id = payload.get("room_id")
            confidence = payload.get("confidence", 0.5)
            
            self.rsn.update_belief_from_observation(
                object_class=self.target_object,
                room_id=room_id,
                detected=True,
                confidence=confidence * 0.9,  # Discount remote observations slightly
                thorough_search=False
            )
            
            # Consider switching to this location if high confidence
            if confidence > 0.8 and self.status.state == AgentState.SEARCHING:
                self._consider_target_switch(room_id, confidence)
    
    def _handle_intention(self, message: Message) -> None:
        """Handle intention broadcast from another agent."""
        payload = message.payload
        sender = message.sender_id
        
        # Update other agent's intention
        if sender in self.other_agents:
            self.other_agents[sender].current_action = GlobalAction(
                action_type=ActionType(payload.get("action_type", "move_to_room")),
                target_room=payload.get("target_room", "")
            )
        
        # Check for conflicts with our intention
        if self.status.current_action:
            if (self.status.current_action.target_room == 
                payload.get("target_room") and
                self.status.current_action.action_type == ActionType.SEARCH_ROOM):
                # Conflict - negotiate or yield
                self._handle_intention_conflict(sender, payload)
    
    def _handle_heartbeat(self, message: Message) -> None:
        """Handle heartbeat from another agent."""
        sender = message.sender_id
        payload = message.payload
        
        # Update agent tracking
        if sender not in self.other_agents:
            self.other_agents[sender] = AgentStatus()
        
        status = payload.get("status", {})
        self.other_agents[sender].state = AgentState(
            status.get("state", "idle")
        )
        if "position" in status:
            self.other_agents[sender].position = np.array(status["position"])
        if "current_room" in status:
            self.other_agents[sender].current_room = status["current_room"]
        
        # Update communication module
        self.comm.connected_agents[sender]["last_heartbeat"] = time.time()
    
    def _handle_dsg_update(self, message: Message) -> None:
        """Handle DSG update from another agent."""
        update = message.payload
        self.dsg.apply_incremental_update(update)
    
    def _handle_rsn_update(self, message: Message) -> None:
        """Handle RSN update from another agent."""
        update = message.payload
        self.rsn.apply_remote_update(update)
    
    def _handle_intention_conflict(
        self,
        other_agent: str,
        other_intention: Dict
    ) -> None:
        """Handle conflict when two agents want to search same room."""
        # Simple resolution: agent with lower ID yields
        if self.agent_id < other_agent:
            # We have priority, continue
            pass
        else:
            # Yield - replan to different room
            if self.planner:
                self.request_replan(reason="conflict")
    
    def _consider_target_switch(
        self,
        new_room: str,
        confidence: float
    ) -> None:
        """Consider switching target based on detection by another agent."""
        if not self.status.current_action:
            return
        
        current_target = self.status.current_action.target_room
        
        # Compare expected value
        current_prob = self.rsn.get_belief(self.target_object).room_probabilities.get(
            current_target, 0
        )
        new_prob = self.rsn.get_belief(self.target_object).room_probabilities.get(
            new_room, 0
        )
        
        # Factor in distance
        current_dist = self.dsg.get_distance(
            self.status.current_room or current_target,
            current_target
        )
        new_dist = self.dsg.get_distance(
            self.status.current_room or new_room,
            new_room
        )
        
        # Switch if significantly better
        current_value = current_prob / (current_dist + 1)
        new_value = new_prob / (new_dist + 1)
        
        if new_value > 1.5 * current_value:
            self.request_replan(reason="better_target_found")
    
    def register_other_agent(
        self,
        agent_id: str,
        position: np.ndarray
    ) -> None:
        """Register another agent for coordination."""
        self.other_agents[agent_id] = AgentStatus(position=position)
        self.comm.register_agent(agent_id, position)
        
        # Update consensus protocol
        self.consensus = ConsensusProtocol(
            num_agents=len(self.other_agents) + 1
        )
    
    def set_planner(self, planner: CollaborativePlanner) -> None:
        """Set the collaborative planner."""
        self.planner = planner
    
    def set_controllers(
        self,
        navigation: Callable = None,
        search: Callable = None,
        inspection: Callable = None
    ) -> None:
        """Set local controllers for execution."""
        if navigation:
            self.navigation_controller = navigation
        if search:
            self.search_controller = search
        if inspection:
            self.inspection_controller = inspection
    
    def start_inspection(
        self,
        target_object: str,
        initial_position: np.ndarray
    ) -> None:
        """
        Start inspection task for a target object.
        
        Args:
            target_object: Object class to search for
            initial_position: Starting position
        """
        self.target_object = target_object
        self.status.position = initial_position
        self.status.current_room = self.dsg.find_room_for_position(initial_position)
        self.status.state = AgentState.IDLE
        
        # Initialize belief
        self.rsn.initialize_belief(target_object)
        
        # Broadcast our status
        self._broadcast_status()
        
        # Start main loop
        self._running = True
        self._update_thread = threading.Thread(target=self._main_loop)
        self._update_thread.start()
    
    def stop_inspection(self) -> None:
        """Stop the inspection task."""
        self._running = False
        if self._update_thread:
            self._update_thread.join(timeout=5.0)
        
        self.status.state = AgentState.IDLE
        self.target_object = None
    
    def _main_loop(self) -> None:
        """Main execution loop."""
        while self._running:
            try:
                # Process incoming messages
                messages = self.comm.receive_messages()
                
                # Execute current action
                self._execute_current_action()
                
                # Send periodic updates
                self._periodic_update()
                
                # Small delay
                time.sleep(0.1)
                
            except Exception as e:
                self.status.state = AgentState.ERROR
                self.status.error_message = str(e)
                print(f"Agent {self.agent_id} error: {e}")
    
    def _execute_current_action(self) -> None:
        """Execute the current action from the plan."""
        if not self.status.current_action:
            # Get action from planner
            if self.planner:
                action = self.planner.get_agent_action(self.agent_id)
                if action:
                    self.status.current_action = action
                    self._broadcast_intention(action)
            return
        
        action = self.status.current_action
        
        if action.action_type == ActionType.MOVE_TO_ROOM:
            self._execute_navigation(action.target_room)
            
        elif action.action_type == ActionType.SEARCH_ROOM:
            self._execute_search(action.target_room)
            
        elif action.action_type == ActionType.INSPECT_OBJECT:
            self._execute_inspection()
            
        elif action.action_type == ActionType.WAIT:
            # Just wait
            pass
    
    def _execute_navigation(self, target_room: str) -> None:
        """Execute navigation to a room."""
        if self.status.state != AgentState.NAVIGATING:
            self.status.state = AgentState.NAVIGATING
        
        if self.navigation_controller:
            # Get target position
            room_node = self.dsg.nodes.get(target_room)
            if room_node:
                reached = self.navigation_controller(
                    current_pos=self.status.position,
                    target_pos=room_node.position
                )
                
                if reached:
                    self.status.current_room = target_room
                    self._action_complete(found_target=False)
        else:
            # Simulate navigation
            self.status.current_room = target_room
            self._action_complete(found_target=False)
    
    def _execute_search(self, room_id: str) -> None:
        """Execute room search."""
        if self.status.state != AgentState.SEARCHING:
            self.status.state = AgentState.SEARCHING
        
        if self.search_controller:
            # Execute search controller
            result = self.search_controller(
                room_id=room_id,
                target_object=self.target_object
            )
            
            if result.get("complete"):
                detected = result.get("detected", False)
                confidence = result.get("confidence", 0.0)
                
                # Update belief
                self.rsn.update_belief_from_observation(
                    object_class=self.target_object,
                    room_id=room_id,
                    detected=detected,
                    confidence=confidence,
                    thorough_search=True
                )
                
                # Broadcast observation
                if detected:
                    self._broadcast_detection(
                        room_id=room_id,
                        confidence=confidence,
                        position=result.get("position")
                    )
                
                # Broadcast belief update
                self._broadcast_belief()
                
                self._action_complete(found_target=detected and confidence > 0.8)
        else:
            # Simulate search
            self.rsn.update_belief_from_observation(
                object_class=self.target_object,
                room_id=room_id,
                detected=False,
                confidence=1.0,
                thorough_search=True
            )
            self._broadcast_belief()
            self._action_complete(found_target=False)
    
    def _execute_inspection(self) -> None:
        """Execute object inspection."""
        self.status.state = AgentState.INSPECTING
        
        if self.inspection_controller:
            result = self.inspection_controller(
                target_object=self.target_object,
                current_pos=self.status.position
            )
            
            if result.get("complete"):
                self._inspection_complete(result)
        else:
            # Simulate inspection
            self._inspection_complete({"success": True})
    
    def _action_complete(self, found_target: bool) -> None:
        """Handle action completion."""
        if self.planner and self.status.current_action:
            done = self.planner.report_action_complete(
                agent_id=self.agent_id,
                room_id=self.status.current_action.target_room,
                found_target=found_target
            )
            
            if done:
                self.status.state = AgentState.INSPECTING
            else:
                self.status.current_action = None
    
    def _inspection_complete(self, result: Dict) -> None:
        """Handle inspection completion."""
        self.status.state = AgentState.IDLE
        self._running = False
        
        # Broadcast completion
        self.comm.send_message(
            MessageType.STATUS_UPDATE,
            {
                "event": "inspection_complete",
                "result": result,
                "target_object": self.target_object
            }
        )
    
    def _periodic_update(self) -> None:
        """Send periodic updates to other agents."""
        # Heartbeat every second
        self._broadcast_status()
        
        # Share DSG updates
        updates = self.dsg.get_incremental_updates(
            since_version=self.dsg.version - 10
        )
        for update in updates[-5:]:  # Limit updates per cycle
            self.comm.send_message(MessageType.DSG_UPDATE, update)
        
        # Share RSN updates if we have new observations
        rsn_updates = self.rsn.get_updates_since(self.rsn.version - 5)
        for update in rsn_updates[-3:]:
            self.comm.send_message(MessageType.RSN_UPDATE, update)
    
    def _broadcast_status(self) -> None:
        """Broadcast current status to other agents."""
        self.comm.send_heartbeat({
            "state": self.status.state.value,
            "position": self.status.position.tolist(),
            "current_room": self.status.current_room,
            "battery_level": self.status.battery_level
        })
    
    def _broadcast_intention(self, action: GlobalAction) -> None:
        """Broadcast intended action."""
        self.comm.send_intention(
            target_room=action.target_room,
            action_type=action.action_type.value,
            estimated_duration=action.estimated_duration
        )
    
    def _broadcast_belief(self) -> None:
        """Broadcast current belief state."""
        if not self.target_object:
            return
        
        belief = self.rsn.get_belief(self.target_object)
        
        self.comm.broadcast_belief_update(BeliefUpdate(
            room_id="",  # Broadcast for all rooms
            object_class=self.target_object,
            probability=0.0,  # Not used for full broadcast
            confidence=belief.confidence,
            evidence_type="observation"
        ))
        
        # Also send detailed room probabilities
        self.comm.send_message(
            MessageType.BELIEF_UPDATE,
            {
                "object_class": self.target_object,
                "room_probabilities": belief.room_probabilities,
                "confidence": belief.confidence
            }
        )
    
    def _broadcast_detection(
        self,
        room_id: str,
        confidence: float,
        position: np.ndarray = None
    ) -> None:
        """Broadcast object detection."""
        self.comm.broadcast_object_detection(
            object_class=self.target_object,
            position=position if position is not None else self.status.position,
            confidence=confidence,
            room_id=room_id
        )
    
    def request_replan(self, reason: str = "periodic") -> None:
        """Request replanning from the coordinator."""
        if self.planner:
            # Get current room probabilities
            belief = self.rsn.get_belief(self.target_object)
            
            # Get all agent positions
            positions = {self.agent_id: self.status.current_room}
            for aid, status in self.other_agents.items():
                if status.current_room:
                    positions[aid] = status.current_room
            
            # Replan
            self.planner.replan(
                target_object=self.target_object,
                room_probabilities=belief.room_probabilities,
                dsg=self.dsg,
                agent_positions=positions,
                trigger_reason=reason
            )
            
            # Clear current action to get new one
            self.status.current_action = None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status as dictionary."""
        return {
            "agent_id": self.agent_id,
            "state": self.status.state.value,
            "position": self.status.position.tolist(),
            "current_room": self.status.current_room,
            "target_object": self.target_object,
            "battery_level": self.status.battery_level,
            "current_action": (
                {
                    "type": self.status.current_action.action_type.value,
                    "target": self.status.current_action.target_room
                } if self.status.current_action else None
            ),
            "num_connected_agents": len(self.other_agents)
        }
