"""
Example usage of SEEK-Multi for collaborative inspection.
"""

import numpy as np
from typing import Dict, List

# Import SEEK-Multi components
from src.shared_dsg import SharedDynamicSceneGraph, NodeType
from src.distributed_rsn import DistributedRSN
from src.agent import InspectionAgent
from src.coordinator import MultiAgentCoordinator, CoordinationMode
from src.collaborative_planner import CollaborativePlanner
from src.task_allocator import TaskAllocator, AgentCapabilities, AllocationMethod
from src.simulation import run_simulation, create_office_environment


def basic_example():
    """Basic example of setting up SEEK-Multi."""
    print("=" * 60)
    print("SEEK-Multi: Basic Setup Example")
    print("=" * 60)
    
    # 1. Define environment
    rooms = [
        {"id": "room_1", "name": "office", "position": [0, 0, 0]},
        {"id": "room_2", "name": "kitchen", "position": [10, 0, 0]},
        {"id": "room_3", "name": "hallway", "position": [5, 5, 0]},
        {"id": "room_4", "name": "storage", "position": [10, 5, 0]},
    ]
    
    connections = [
        ("room_1", "room_3", 5.0),
        ("room_2", "room_3", 5.0),
        ("room_3", "room_4", 5.0),
        ("room_2", "room_4", 5.0),
    ]
    
    # 2. Create shared DSG
    dsg = SharedDynamicSceneGraph.from_blueprint(
        agent_id="main",
        rooms=rooms,
        connections=connections
    )
    print(f"Created DSG with {len(dsg.nodes)} nodes")
    
    # 3. Create shared RSN
    rsn = DistributedRSN(agent_id="main")
    rsn.register_rooms([(r["id"], r["name"]) for r in rooms])
    print(f"Created RSN with {len(rsn.room_id_to_type)} rooms")
    
    # 4. Create coordinator
    coordinator = MultiAgentCoordinator(
        coordination_mode=CoordinationMode.HYBRID
    )
    
    # 5. Create agents
    num_agents = 2
    for i in range(num_agents):
        agent = InspectionAgent(
            agent_id=f"robot_{i}",
            dsg=dsg,
            rsn=rsn
        )
        coordinator.register_agent(agent)
    
    print(f"Registered {num_agents} agents")
    
    # 6. Initialize coordinator
    coordinator.set_shared_dsg(dsg)
    coordinator.set_shared_rsn(rsn)
    coordinator.initialize_from_blueprint(rooms, connections)
    
    print("Coordinator initialized")
    
    # 7. Query RSN for object probabilities
    target = "fire_extinguisher"
    rsn.initialize_belief(target)
    belief = rsn.get_belief(target)
    
    print(f"\nProbabilities for finding '{target}':")
    for room_id, prob in sorted(belief.room_probabilities.items(), key=lambda x: -x[1]):
        room_name = rsn.room_id_to_type.get(room_id, room_id)
        print(f"  {room_name} ({room_id}): {prob:.3f}")
    
    return coordinator


def collaborative_planning_example():
    """Example of collaborative planning."""
    print("\n" + "=" * 60)
    print("SEEK-Multi: Collaborative Planning Example")
    print("=" * 60)
    
    # Create environment
    rooms, connections, objects = create_office_environment()
    
    # Create DSG
    dsg = SharedDynamicSceneGraph.from_blueprint(
        agent_id="planning_example",
        rooms=rooms,
        connections=connections
    )
    
    # Create planner for 3 agents
    agent_ids = ["agent_0", "agent_1", "agent_2"]
    planner = CollaborativePlanner(agent_ids=agent_ids)
    
    # Create RSN and get probabilities
    rsn = DistributedRSN(agent_id="planning_example")
    rsn.register_rooms([(r["id"], r["name"]) for r in rooms])
    
    target = "fire_extinguisher"
    rsn.initialize_belief(target)
    belief = rsn.get_belief(target)
    
    # Set initial agent positions
    agent_positions = {
        "agent_0": "entrance",
        "agent_1": "lobby",
        "agent_2": "hallway_1"
    }
    
    # Compute coordinated plan
    plan = planner.compute_coordinated_plan(
        target_object=target,
        room_probabilities=belief.room_probabilities,
        dsg=dsg,
        agent_positions=agent_positions
    )
    
    print(f"\nCoordinated plan for finding '{target}':")
    print(f"Estimated completion time: {plan.estimated_completion_time:.2f}")
    
    for agent_id in agent_ids:
        agent_plan = plan.get_agent_plan(agent_id)
        print(f"\n{agent_id} plan (cost: {agent_plan.estimated_total_cost:.2f}):")
        for i, action in enumerate(agent_plan.actions[:5]):  # Show first 5 actions
            print(f"  {i+1}. {action.action_type.value} -> {action.target_room}")
    
    if plan.coordination_points:
        print(f"\nCoordination points: {len(plan.coordination_points)}")
        for cp in plan.coordination_points[:3]:
            print(f"  Time {cp['time']:.1f}: {cp['agents']} at {cp['room']}")


def belief_fusion_example():
    """Example of distributed belief fusion."""
    print("\n" + "=" * 60)
    print("SEEK-Multi: Belief Fusion Example")
    print("=" * 60)
    
    rooms = [
        {"id": "room_1", "name": "kitchen", "position": [0, 0, 0]},
        {"id": "room_2", "name": "office", "position": [10, 0, 0]},
        {"id": "room_3", "name": "storage", "position": [5, 5, 0]},
    ]
    
    # Create RSN for each agent
    rsn_1 = DistributedRSN(agent_id="agent_1")
    rsn_2 = DistributedRSN(agent_id="agent_2")
    
    for rsn in [rsn_1, rsn_2]:
        rsn.register_rooms([(r["id"], r["name"]) for r in rooms])
    
    target = "coffee_mug"
    
    # Initialize beliefs
    rsn_1.initialize_belief(target)
    rsn_2.initialize_belief(target)
    
    print(f"Initial beliefs for '{target}':")
    print(f"  Agent 1: {rsn_1.get_belief(target).room_probabilities}")
    print(f"  Agent 2: {rsn_2.get_belief(target).room_probabilities}")
    
    # Agent 1 searches room_2 and doesn't find object
    rsn_1.update_belief_from_observation(
        object_class=target,
        room_id="room_2",
        detected=False,
        confidence=1.0,
        thorough_search=True
    )
    
    print(f"\nAfter Agent 1 searches room_2 (not found):")
    print(f"  Agent 1: {rsn_1.get_belief(target).room_probabilities}")
    
    # Agent 2 fuses belief from Agent 1
    belief_1 = rsn_1.get_belief(target)
    rsn_2.fuse_belief_from_agent(
        object_class=target,
        other_belief=belief_1.room_probabilities,
        other_confidence=belief_1.confidence,
        other_agent_id="agent_1"
    )
    
    print(f"\nAfter Agent 2 fuses Agent 1's belief:")
    print(f"  Agent 2: {rsn_2.get_belief(target).room_probabilities}")
    
    # Agent 2 searches room_1 and finds object
    rsn_2.update_belief_from_observation(
        object_class=target,
        room_id="room_1",
        detected=True,
        confidence=0.9,
        thorough_search=True
    )
    
    print(f"\nAfter Agent 2 finds object in room_1:")
    print(f"  Agent 2: {rsn_2.get_belief(target).room_probabilities}")


def task_allocation_example():
    """Example of task allocation."""
    print("\n" + "=" * 60)
    print("SEEK-Multi: Task Allocation Example")
    print("=" * 60)
    
    # Define agents with different capabilities
    agents = [
        AgentCapabilities(
            agent_id="ground_robot",
            capabilities={"ground"},
            max_load=10.0,
            current_position="entrance",
            speed=1.0
        ),
        AgentCapabilities(
            agent_id="drone",
            capabilities={"fly", "ground"},
            max_load=5.0,
            current_position="entrance",
            speed=2.0
        ),
        AgentCapabilities(
            agent_id="arm_robot",
            capabilities={"ground", "manipulate"},
            max_load=8.0,
            current_position="lobby",
            speed=0.5
        )
    ]
    
    # Create task allocator
    allocator = TaskAllocator(
        agents=agents,
        method=AllocationMethod.AUCTION
    )
    
    # Create environment
    rooms, connections, _ = create_office_environment()
    dsg = SharedDynamicSceneGraph.from_blueprint(
        agent_id="allocation_example",
        rooms=rooms,
        connections=connections
    )
    
    # Create RSN for probabilities
    rsn = DistributedRSN(agent_id="allocation_example")
    rsn.register_rooms([(r["id"], r["name"]) for r in rooms])
    rsn.initialize_belief("fire_extinguisher")
    belief = rsn.get_belief("fire_extinguisher")
    
    # Generate tasks from rooms
    room_ids = [r["id"] for r in rooms]
    tasks = allocator.generate_room_tasks(
        rooms=room_ids,
        room_probabilities=belief.room_probabilities
    )
    
    allocator.add_tasks(tasks)
    
    # Allocate tasks
    allocation = allocator.allocate(dsg, belief.room_probabilities)
    
    print("Task allocation results:")
    for agent_id, agent_tasks in allocation.items():
        print(f"\n{agent_id} ({len(agent_tasks)} tasks):")
        for task in agent_tasks[:5]:
            print(f"  - {task.target_room} (priority: {task.priority:.3f})")


def run_comparative_simulation():
    """Run simulation comparing different numbers of agents."""
    print("\n" + "=" * 60)
    print("SEEK-Multi: Comparative Simulation")
    print("=" * 60)
    
    results = []
    
    for n_agents in [1, 2, 3, 4]:
        print(f"\nRunning with {n_agents} agent(s)...")
        result = run_simulation(
            num_agents=n_agents,
            target_object="fire_extinguisher",
            max_steps=500,
            verbose=False
        )
        results.append(result)
        print(f"  SPL: {result['spl']:.3f}, Steps: {result['steps']}, "
              f"Distance: {result['total_distance']:.1f}m")
    
    print("\n" + "-" * 40)
    print("Summary:")
    print(f"{'Agents':<10} {'SPL':<10} {'Steps':<10} {'Distance':<10}")
    print("-" * 40)
    for r in results:
        print(f"{r['num_agents']:<10} {r['spl']:<10.3f} {r['steps']:<10} "
              f"{r['total_distance']:<10.1f}")


if __name__ == "__main__":
    # Run all examples
    basic_example()
    collaborative_planning_example()
    belief_fusion_example()
    task_allocation_example()
    run_comparative_simulation()
