"""
Unit tests for SEEK-Multi components.
"""

import unittest
import numpy as np
from typing import Dict, List

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.shared_dsg import SharedDynamicSceneGraph, NodeType, DSGNode
from src.distributed_rsn import DistributedRSN
from src.collaborative_planner import CollaborativePlanner, ActionType
from src.task_allocator import TaskAllocator, Task, AgentCapabilities, AllocationMethod
from src.communication import CommunicationProtocol, Message, MessageType, ConsensusProtocol


class TestSharedDSG(unittest.TestCase):
    """Tests for SharedDynamicSceneGraph."""
    
    def setUp(self):
        """Set up test DSG."""
        self.rooms = [
            {"id": "room_1", "name": "office", "position": [0, 0, 0]},
            {"id": "room_2", "name": "kitchen", "position": [10, 0, 0]},
            {"id": "room_3", "name": "hallway", "position": [5, 5, 0]},
        ]
        self.connections = [
            ("room_1", "room_2", 10.0),
            ("room_1", "room_3", 7.0),
            ("room_2", "room_3", 7.0),
        ]
        self.dsg = SharedDynamicSceneGraph.from_blueprint(
            agent_id="test",
            rooms=self.rooms,
            connections=self.connections
        )
    
    def test_node_creation(self):
        """Test that nodes are created correctly."""
        self.assertEqual(len(self.dsg.get_room_nodes()), 3)
        
        room_1 = self.dsg.nodes.get("room_1")
        self.assertIsNotNone(room_1)
        self.assertEqual(room_1.semantic_class, "office")
    
    def test_edge_creation(self):
        """Test that edges are created correctly."""
        self.assertIn(("room_1", "room_2"), self.dsg.edges)
        
        edge = self.dsg.edges[("room_1", "room_2")]
        self.assertEqual(edge.weight, 10.0)
    
    def test_distance_computation(self):
        """Test shortest path distance computation."""
        self.dsg.compute_distance_matrix()
        
        dist = self.dsg.get_distance("room_1", "room_2")
        self.assertEqual(dist, 10.0)
        
        # Check triangle inequality
        dist_13 = self.dsg.get_distance("room_1", "room_3")
        dist_23 = self.dsg.get_distance("room_2", "room_3")
        self.assertLessEqual(dist, dist_13 + dist_23)
    
    def test_merge(self):
        """Test DSG merging."""
        dsg2 = SharedDynamicSceneGraph("agent_2")
        
        # Add new node
        dsg2.add_node(
            node_id="room_4",
            node_type=NodeType.ROOM,
            position=np.array([15, 0, 0]),
            semantic_class="storage"
        )
        
        # Merge
        self.dsg.merge(dsg2)
        
        self.assertIn("room_4", self.dsg.nodes)
        self.assertEqual(len(self.dsg.get_room_nodes()), 4)


class TestDistributedRSN(unittest.TestCase):
    """Tests for DistributedRSN."""
    
    def setUp(self):
        """Set up test RSN."""
        self.rsn = DistributedRSN(agent_id="test")
        self.rooms = [
            ("room_1", "kitchen"),
            ("room_2", "office"),
            ("room_3", "storage"),
        ]
        self.rsn.register_rooms(self.rooms)
    
    def test_room_registration(self):
        """Test room registration."""
        self.assertEqual(len(self.rsn.room_id_to_type), 3)
        self.assertEqual(self.rsn.room_id_to_type["room_1"], "kitchen")
    
    def test_belief_initialization(self):
        """Test belief initialization."""
        belief = self.rsn.initialize_belief("coffee_mug")
        
        self.assertEqual(belief.object_class, "coffee_mug")
        self.assertEqual(len(belief.room_probabilities), 3)
        
        # Sum should be approximately 1
        total = sum(belief.room_probabilities.values())
        self.assertAlmostEqual(total, 1.0, places=5)
    
    def test_belief_update(self):
        """Test belief update from observation."""
        self.rsn.initialize_belief("coffee_mug")
        
        # Update: didn't find in room_2
        belief = self.rsn.update_belief_from_observation(
            object_class="coffee_mug",
            room_id="room_2",
            detected=False,
            confidence=1.0,
            thorough_search=True
        )
        
        # Probability should decrease for room_2
        initial_belief = self.rsn.predict_room_probabilities("coffee_mug")
        self.assertLess(
            belief.room_probabilities["room_2"],
            initial_belief.get("room_2", 1.0)
        )
    
    def test_belief_fusion(self):
        """Test belief fusion from another agent."""
        self.rsn.initialize_belief("fire_extinguisher")
        
        # Other agent's belief
        other_belief = {
            "room_1": 0.8,
            "room_2": 0.1,
            "room_3": 0.1
        }
        
        belief = self.rsn.fuse_belief_from_agent(
            object_class="fire_extinguisher",
            other_belief=other_belief,
            other_confidence=0.9,
            other_agent_id="agent_2"
        )
        
        # Fused belief should be influenced by other agent
        self.assertGreater(belief.room_probabilities["room_1"], 0.3)


class TestCollaborativePlanner(unittest.TestCase):
    """Tests for CollaborativePlanner."""
    
    def setUp(self):
        """Set up test planner."""
        self.agent_ids = ["agent_1", "agent_2"]
        self.planner = CollaborativePlanner(agent_ids=self.agent_ids)
        
        # Create simple DSG
        self.rooms = [
            {"id": "room_1", "name": "office", "position": [0, 0, 0]},
            {"id": "room_2", "name": "kitchen", "position": [10, 0, 0]},
            {"id": "room_3", "name": "storage", "position": [5, 5, 0]},
        ]
        self.connections = [
            ("room_1", "room_2", 10.0),
            ("room_1", "room_3", 7.0),
            ("room_2", "room_3", 7.0),
        ]
        self.dsg = SharedDynamicSceneGraph.from_blueprint(
            agent_id="test",
            rooms=self.rooms,
            connections=self.connections
        )
    
    def test_plan_computation(self):
        """Test coordinated plan computation."""
        room_probs = {
            "room_1": 0.2,
            "room_2": 0.5,
            "room_3": 0.3
        }
        
        agent_positions = {
            "agent_1": "room_1",
            "agent_2": "room_3"
        }
        
        plan = self.planner.compute_coordinated_plan(
            target_object="fire_extinguisher",
            room_probabilities=room_probs,
            dsg=self.dsg,
            agent_positions=agent_positions
        )
        
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.agent_plans), 2)
        
        # Each agent should have actions
        for agent_id in self.agent_ids:
            agent_plan = plan.get_agent_plan(agent_id)
            self.assertIsNotNone(agent_plan)
    
    def test_action_retrieval(self):
        """Test getting current action for an agent."""
        room_probs = {"room_1": 0.3, "room_2": 0.4, "room_3": 0.3}
        agent_positions = {"agent_1": "room_1", "agent_2": "room_3"}
        
        self.planner.compute_coordinated_plan(
            target_object="fire_extinguisher",
            room_probabilities=room_probs,
            dsg=self.dsg,
            agent_positions=agent_positions
        )
        
        action = self.planner.get_agent_action("agent_1")
        self.assertIsNotNone(action)
        self.assertIn(action.action_type, [ActionType.MOVE_TO_ROOM, ActionType.SEARCH_ROOM])


class TestTaskAllocator(unittest.TestCase):
    """Tests for TaskAllocator."""
    
    def setUp(self):
        """Set up test allocator."""
        self.agents = [
            AgentCapabilities(
                agent_id="agent_1",
                capabilities={"ground"},
                max_load=10.0,
                current_position="room_1"
            ),
            AgentCapabilities(
                agent_id="agent_2",
                capabilities={"ground", "fly"},
                max_load=5.0,
                current_position="room_2"
            )
        ]
        self.allocator = TaskAllocator(
            agents=self.agents,
            method=AllocationMethod.AUCTION
        )
        
        # Create DSG
        rooms = [
            {"id": "room_1", "name": "office", "position": [0, 0, 0]},
            {"id": "room_2", "name": "kitchen", "position": [10, 0, 0]},
        ]
        connections = [("room_1", "room_2", 10.0)]
        self.dsg = SharedDynamicSceneGraph.from_blueprint(
            agent_id="test",
            rooms=rooms,
            connections=connections
        )
    
    def test_task_generation(self):
        """Test task generation from rooms."""
        room_probs = {"room_1": 0.6, "room_2": 0.4}
        
        tasks = self.allocator.generate_room_tasks(
            rooms=["room_1", "room_2"],
            room_probabilities=room_probs
        )
        
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0].priority, 0.6)
    
    def test_task_allocation(self):
        """Test task allocation."""
        room_probs = {"room_1": 0.6, "room_2": 0.4}
        
        tasks = self.allocator.generate_room_tasks(
            rooms=["room_1", "room_2"],
            room_probabilities=room_probs
        )
        self.allocator.add_tasks(tasks)
        
        allocation = self.allocator.allocate(self.dsg, room_probs)
        
        # All tasks should be assigned
        total_assigned = sum(len(tasks) for tasks in allocation.values())
        self.assertEqual(total_assigned, 2)


class TestCommunication(unittest.TestCase):
    """Tests for CommunicationProtocol."""
    
    def setUp(self):
        """Set up test communication."""
        self.comm1 = CommunicationProtocol(agent_id="agent_1")
        self.comm2 = CommunicationProtocol(agent_id="agent_2")
        
        # Register agents with each other
        self.comm1.register_agent("agent_2", np.array([10, 0, 0]))
        self.comm2.register_agent("agent_1", np.array([0, 0, 0]))
    
    def test_message_creation(self):
        """Test message creation."""
        msg = self.comm1.send_message(
            MessageType.BELIEF_UPDATE,
            {"test": "data"},
            receiver_id="agent_2"
        )
        
        self.assertEqual(msg.sender_id, "agent_1")
        self.assertEqual(msg.receiver_id, "agent_2")
        self.assertEqual(msg.msg_type, MessageType.BELIEF_UPDATE)
    
    def test_message_delivery(self):
        """Test message sending and receiving."""
        # Send message
        msg = self.comm1.send_message(
            MessageType.HEARTBEAT,
            {"status": "ok"}
        )
        
        # Get outgoing messages
        outgoing = self.comm1.get_outgoing_messages(np.array([0, 0, 0]))
        
        self.assertIn("agent_2", outgoing)
        self.assertEqual(len(outgoing["agent_2"]), 1)
    
    def test_consensus_fusion(self):
        """Test consensus belief fusion."""
        consensus = ConsensusProtocol(num_agents=3)
        
        local_belief = np.array([0.3, 0.4, 0.3])
        neighbor_beliefs = [
            ("agent_2", np.array([0.5, 0.3, 0.2]), 0.8),
            ("agent_3", np.array([0.2, 0.5, 0.3]), 0.7)
        ]
        
        fused = consensus.fuse_beliefs(local_belief, neighbor_beliefs)
        
        # Result should be normalized
        self.assertAlmostEqual(fused.sum(), 1.0, places=5)
        
        # Result should be influenced by all beliefs
        self.assertGreater(fused[0], 0.2)
        self.assertGreater(fused[1], 0.2)


class TestIntegration(unittest.TestCase):
    """Integration tests for SEEK-Multi."""
    
    def test_full_workflow(self):
        """Test complete multi-agent workflow."""
        from src.simulation import run_simulation
        
        # Run short simulation
        results = run_simulation(
            num_agents=2,
            target_object="fire_extinguisher",
            max_steps=100,
            verbose=False
        )
        
        self.assertIn("success", results)
        self.assertIn("spl", results)
        self.assertIn("steps", results)
        self.assertGreaterEqual(results["steps"], 1)


if __name__ == "__main__":
    unittest.main()
