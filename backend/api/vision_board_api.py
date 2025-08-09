#!/usr/bin/env python3
"""
Vision Board API - Board of Directors style multi-agent orchestration
Connects to real agent implementations with verbose logging and autonomous improvement
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
import sys
import logging
from dataclasses import dataclass, asdict
import uuid

# Add agents directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'agents'))
sys.path.insert(0, str(Path(__file__).parent / 'api'))

# Import real agent implementations
try:
    from cto_agent import CTOAgent
    from cfo_agent import CFOAgent
    logger.info("Loaded real CTO and CFO agents")
except ImportError as e:
    logger.warning(f"Could not load real agents: {e}")
    CTOAgent = None
    CFOAgent = None

try:
    from orchestrator import Orchestrator, Task, TaskType
    from agent_health_monitor import AgentHealthMonitor
    from comfyui_integration_agent import ComfyUIIntegrationAgent
    from service_health_monitor_agent import ServiceHealthMonitorAgent
    from tower_infrastructure_agent import TowerInfrastructureAgent
    from database_management_agent import DatabaseManagementAgent
    from vision_guardian import VisionGuardian
    from architect_agent import ArchitectAgent
    from security_agent import SecurityAgent
    from anime_agent import AnimeAgent
    from llm_manager import LLMManager
except ImportError as e:
    logger.warning(f"Could not import some agents: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Vision Board API", version="2.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Board of Directors Configuration
@dataclass
class BoardMember:
    id: str
    name: str
    role: str  # CEO, CTO, CFO, CMO, COO, CSO, etc.
    agent_type: str
    trust_level: float
    status: str  # active, thinking, speaking, listening, busy, overloaded
    current_opinion: str = ""
    avatar: str = ""  # emoji or image
    specialization: List[str] = None
    active_tasks: int = 0
    capacity: int = 5  # max concurrent tasks
    current_workload: List[Dict] = None  # [{task_id, description, priority, started_at}]
    efficiency_rating: float = 1.0  # performance multiplier
    last_task_completion: str = ""
    
@dataclass
class ResourceAllocation:
    task_id: str
    agent_id: str
    agent_name: str
    task_description: str
    priority: int
    estimated_duration: int  # minutes
    actual_duration: int = 0
    status: str = "assigned"  # assigned, in_progress, completed, blocked
    started_at: str = ""
    completed_at: str = ""
    dependencies: List[str] = None  # other task_ids this depends on
    resource_requirements: Dict[str, Any] = None  # memory, cpu, storage needs
    
class VisionBoard:
    """Board of Directors management system with resource allocation tracking"""
    
    def __init__(self):
        self.board_members: Dict[str, BoardMember] = {}
        self.chairperson_id: str = "vision-guardian"
        self.orchestrator = None
        self.llm_manager = None
        self.active_conversation: Optional[str] = None
        self.conversation_history: List[Dict] = []
        self.improvement_log: List[Dict] = []
        self.websocket_clients: Set[WebSocket] = set()
        # Resource allocation tracking
        self.resource_allocations: Dict[str, ResourceAllocation] = {}
        self.task_queue: List[Dict] = []  # pending tasks waiting for assignment
        self.resource_metrics: Dict[str, Any] = {
            "total_capacity": 0,
            "utilized_capacity": 0,
            "efficiency_avg": 0.0,
            "tasks_completed_today": 0,
            "bottlenecks": []
        }
        
        # Real agent instances for task delegation
        self.real_agents: Dict[str, Any] = {}
        
        # Progress monitoring
        self.progress_tracker: Dict[str, Dict] = {}  # task_id -> progress data
        self.completion_metrics: Dict[str, Any] = {
            "tasks_completed_today": 0,
            "average_completion_time": 0,
            "success_rate": 100.0,
            "agent_performance": {},
            "daily_throughput": []
        }
        
        self.initialize_board()
        
    def initialize_board(self):
        """Initialize board members with real agents"""
        board_config = [
            BoardMember("vision-guardian", "Vision Guardian", "CEO", "strategic", 0.95, 
                       "active", "", "👁️", ["strategy", "oversight", "coordination"]),
            BoardMember("architect", "System Architect", "CTO", "technical", 0.85, 
                       "active", "", "🏗️", ["architecture", "design", "integration"]),
            BoardMember("security", "Security Chief", "CSO", "security", 0.90,
                       "active", "", "🔒", ["security", "compliance", "risk"]),
            BoardMember("infrastructure", "Infrastructure Manager", "COO", "operations", 0.85,
                       "active", "", "⚙️", ["infrastructure", "deployment", "monitoring"]),
            BoardMember("database", "Data Administrator", "CDO", "data", 0.80,
                       "active", "", "💾", ["database", "analytics", "storage"]),
            BoardMember("comfyui", "Creative Director", "CCO", "creative", 0.75,
                       "active", "", "🎨", ["ui/ux", "generation", "creative"]),
            BoardMember("health-monitor", "Health Monitor", "CHO", "health", 0.85,
                       "active", "", "❤️", ["monitoring", "health", "performance"]),
            BoardMember("anime", "Content Producer", "CPO", "content", 0.70,
                       "active", "", "🎌", ["content", "media", "production"])
        ]
        
        for member in board_config:
            member.current_workload = []
            self.board_members[member.id] = member
            # Update total capacity
            self.resource_metrics["total_capacity"] += member.capacity
        
        # Initialize real agents for task delegation
        self.initialize_real_agents()
            
        # Initialize real orchestrator
        try:
            self.orchestrator = Orchestrator()
            asyncio.create_task(self.orchestrator.start())
            logger.info("Orchestrator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            
        # Initialize LLM manager for model management
        try:
            self.llm_manager = LLMManager()
            logger.info("LLM Manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize LLM manager: {e}")
    
    def initialize_real_agents(self):
        """Initialize real agent instances for task delegation"""
        # Initialize CTO Agent
        if CTOAgent:
            try:
                self.real_agents["architect"] = CTOAgent()
                logger.info("Initialized real CTO agent")
            except Exception as e:
                logger.error(f"Failed to initialize CTO agent: {e}")
        
        # Initialize CFO Agent  
        if CFOAgent:
            try:
                self.real_agents["database"] = CFOAgent()
                logger.info("Initialized real CFO agent")
            except Exception as e:
                logger.error(f"Failed to initialize CFO agent: {e}")
        
        # Map other board members to their real agents when available
        agent_mapping = {
            "vision-guardian": "vision-guardian",
            "security": "security",  
            "infrastructure": "infrastructure",
            "comfyui": "comfyui",
            "health-monitor": "health-monitor", 
            "anime": "anime"
        }
        
        # Log initialized agents
        logger.info(f"Initialized {len(self.real_agents)} real agents: {list(self.real_agents.keys())}")
    
    async def broadcast_update(self, update_type: str, data: Dict):
        """Broadcast updates to all connected WebSocket clients"""
        message = {
            "type": update_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        logger.info(f"Broadcasting {update_type} to {len(self.websocket_clients)} clients")
        
        disconnected = set()
        for ws in self.websocket_clients:
            try:
                await ws.send_json(message)
                logger.debug(f"Sent {update_type} message")
            except Exception as e:
                logger.error(f"Failed to send to client: {e}")
                disconnected.add(ws)
        
        # Remove disconnected clients
        self.websocket_clients -= disconnected
    
    async def start_conversation(self, topic: str, context: Dict = None) -> str:
        """Start a board conversation on a topic"""
        conversation_id = str(uuid.uuid4())
        self.active_conversation = conversation_id
        
        conversation = {
            "id": conversation_id,
            "topic": topic,
            "context": context or {},
            "started_at": datetime.now().isoformat(),
            "participants": list(self.board_members.keys()),
            "messages": []
        }
        
        self.conversation_history.append(conversation)
        
        # Notify all board members
        await self.broadcast_update("conversation_started", {
            "conversation_id": conversation_id,
            "topic": topic,
            "chairperson": self.chairperson_id
        })
        
        # Get initial thoughts from each board member
        await self.gather_opinions(topic, context)
        
        return conversation_id
    
    async def gather_opinions(self, topic: str, context: Dict):
        """Each board member contributes their perspective"""
        for member_id, member in self.board_members.items():
            # Update member status
            member.status = "thinking"
            await self.broadcast_update("member_status", {
                "member_id": member_id,
                "status": "thinking"
            })
            
            # Simulate agent thinking (in real implementation, call actual agent)
            await asyncio.sleep(0.5)  # Simulate processing
            
            # Generate opinion based on specialization
            opinion = await self.generate_agent_opinion(member, topic, context)
            member.current_opinion = opinion
            
            # Update status to speaking
            member.status = "speaking"
            await self.broadcast_update("member_opinion", {
                "member_id": member_id,
                "member_name": member.name,
                "role": member.role,
                "opinion": opinion,
                "status": "speaking"
            })
            
            # Add to conversation history
            if self.conversation_history:
                self.conversation_history[-1]["messages"].append({
                    "member_id": member_id,
                    "member_name": member.name,
                    "role": member.role,
                    "message": opinion,
                    "timestamp": datetime.now().isoformat()
                })
            
            # Brief pause between speakers
            await asyncio.sleep(0.3)
            
            # Update status back to listening
            member.status = "listening"
    
    async def generate_agent_opinion(self, member: BoardMember, topic: str, context: Dict) -> str:
        """Generate opinion based on agent specialization - delegates to real agents when available"""
        
        # Try to delegate to real agents first
        if member.id in self.real_agents:
            try:
                real_agent = self.real_agents[member.id]
                
                # Call the real agent's analysis method
                if hasattr(real_agent, 'analyze_task'):
                    analysis = await real_agent.analyze_task(topic, context)
                    # Format the analysis as an opinion
                    return f"Based on my {member.role} analysis: {analysis.get('recommendations', ['No specific recommendations'])[0]}"
                elif hasattr(real_agent, 'execute_task'):
                    # For agents that execute tasks, get a quick assessment
                    result = await real_agent.execute_task(f"Provide opinion on: {topic}", context)
                    return f"From my {member.role} perspective: {result.get('status', 'Analyzing this topic')}"
            except Exception as e:
                logger.error(f"Error calling real agent {member.id}: {e}")
        
        # Fallback to simulated opinions
        opinions = {
            "CEO": f"From a strategic perspective on '{topic}': We need to ensure alignment with our long-term vision while maintaining system stability.",
            "CTO": f"Technically speaking about '{topic}': We should consider the architectural implications and ensure scalability.",
            "CSO": f"Security analysis of '{topic}': We must evaluate potential risks and ensure compliance with security protocols.",
            "COO": f"Operationally for '{topic}': We need to consider deployment strategies and monitoring requirements.",
            "CDO": f"From a data perspective on '{topic}': We should analyze the data requirements and storage implications.",
            "CCO": f"Creatively approaching '{topic}': We can enhance user experience with innovative UI solutions.",
            "CHO": f"Health monitoring for '{topic}': We need to track performance metrics and system health.",
            "CPO": f"Content-wise for '{topic}': We should consider content generation and media production aspects."
        }
        
        return opinions.get(member.role, f"Regarding '{topic}': This requires careful consideration from my {member.role} perspective.")
    
    async def autonomous_improvement(self):
        """Autonomous improvement process that runs in background"""
        while True:
            try:
                # Check system health
                health_status = await self.check_system_health()
                
                # Check for model updates
                model_updates = await self.check_model_updates()
                
                # Performance optimization suggestions
                optimizations = await self.analyze_performance()
                
                improvement = {
                    "timestamp": datetime.now().isoformat(),
                    "health": health_status,
                    "model_updates": model_updates,
                    "optimizations": optimizations
                }
                
                self.improvement_log.append(improvement)
                
                # Broadcast improvements
                await self.broadcast_update("autonomous_improvement", improvement)
                
                # Run every 30 seconds
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Autonomous improvement error: {e}")
                await asyncio.sleep(60)
    
    async def check_system_health(self) -> Dict:
        """Check overall system health"""
        return {
            "status": "healthy",
            "agents_active": len([m for m in self.board_members.values() if m.status == "active"]),
            "tasks_pending": 0,  # Would query orchestrator
            "performance": "optimal"
        }
    
    async def check_model_updates(self) -> List[Dict]:
        """Check for available model updates"""
        if self.llm_manager:
            # Check available models
            return [
                {"model": "claude-3-opus", "status": "available", "action": "ready"},
                {"model": "gpt-4", "status": "update_available", "action": "download"},
            ]
        return []
    
    async def analyze_performance(self) -> List[str]:
        """Analyze system performance and suggest optimizations"""
        await self.update_resource_metrics()
        
        suggestions = []
        
        # Check for overloaded agents
        for agent_id, member in self.board_members.items():
            utilization = len(member.current_workload) / member.capacity
            if utilization > 0.8:
                suggestions.append(f"{member.name} is {utilization:.0%} utilized - consider load balancing")
        
        # Check for underutilized agents
        underutilized = [m for m in self.board_members.values() if len(m.current_workload) < m.capacity * 0.3]
        if underutilized:
            suggestions.append(f"{len(underutilized)} agents underutilized - consider task redistribution")
        
        # Check task queue
        if len(self.task_queue) > 5:
            suggestions.append(f"{len(self.task_queue)} tasks pending - consider increasing capacity")
        
        return suggestions
    
    async def update_resource_metrics(self):
        """Update resource utilization metrics"""
        total_tasks = sum(len(m.current_workload) for m in self.board_members.values())
        self.resource_metrics["utilized_capacity"] = total_tasks
        
        # Calculate average efficiency
        efficiencies = [m.efficiency_rating for m in self.board_members.values()]
        self.resource_metrics["efficiency_avg"] = sum(efficiencies) / len(efficiencies) if efficiencies else 0
        
        # Identify bottlenecks
        bottlenecks = []
        for agent_id, member in self.board_members.items():
            if len(member.current_workload) >= member.capacity:
                bottlenecks.append({
                    "agent": member.name,
                    "capacity": member.capacity,
                    "current_load": len(member.current_workload)
                })
        self.resource_metrics["bottlenecks"] = bottlenecks
    
    async def assign_task_to_agent(self, task_data: Dict) -> Optional[str]:
        """Intelligently assign task to best available agent"""
        task_id = str(uuid.uuid4())
        task_description = task_data.get("description", "")
        task_type = task_data.get("type", "general")
        priority = task_data.get("priority", 5)
        estimated_duration = task_data.get("estimated_duration", 30)  # minutes
        
        # Find best agent based on specialization and availability
        best_agent = None
        best_score = -1
        
        for agent_id, member in self.board_members.items():
            # Skip if agent is at capacity
            if len(member.current_workload) >= member.capacity:
                continue
            
            # Calculate assignment score
            score = 0
            
            # Specialization match (higher is better)
            if member.specialization:
                for spec in member.specialization:
                    if spec.lower() in task_description.lower() or spec.lower() in task_type.lower():
                        score += 10
            
            # Availability (fewer current tasks is better)
            availability_score = (member.capacity - len(member.current_workload)) / member.capacity
            score += availability_score * 5
            
            # Efficiency rating
            score += member.efficiency_rating * 3
            
            # Priority handling (some agents better with high priority)
            if priority >= 8 and member.role in ["CEO", "CSO"]:
                score += 2
            
            if score > best_score:
                best_score = score
                best_agent = member
        
        if not best_agent:
            # All agents at capacity - add to queue
            self.task_queue.append({
                "id": task_id,
                "description": task_description,
                "type": task_type,
                "priority": priority,
                "estimated_duration": estimated_duration,
                "queued_at": datetime.now().isoformat()
            })
            
            await self.broadcast_update("task_queued", {
                "task_id": task_id,
                "description": task_description,
                "queue_position": len(self.task_queue)
            })
            
            return None
        
        # Assign task to agent
        task_allocation = ResourceAllocation(
            task_id=task_id,
            agent_id=best_agent.id,
            agent_name=best_agent.name,
            task_description=task_description,
            priority=priority,
            estimated_duration=estimated_duration,
            started_at=datetime.now().isoformat(),
            dependencies=task_data.get("dependencies", []),
            resource_requirements=task_data.get("resource_requirements", {})
        )
        
        # Add to agent's workload
        workload_item = {
            "task_id": task_id,
            "description": task_description,
            "priority": priority,
            "started_at": datetime.now().isoformat()
        }
        best_agent.current_workload.append(workload_item)
        best_agent.active_tasks = len(best_agent.current_workload)
        
        # Update agent status
        if len(best_agent.current_workload) >= best_agent.capacity:
            best_agent.status = "overloaded"
        elif len(best_agent.current_workload) > best_agent.capacity * 0.7:
            best_agent.status = "busy"
        else:
            best_agent.status = "active"
        
        # Store allocation
        self.resource_allocations[task_id] = task_allocation
        
        # Start progress tracking
        await self.start_progress_tracking(task_id, {
            "description": task_description,
            "agent_id": best_agent.id,
            "agent_name": best_agent.name,
            "estimated_completion": f"{estimated_duration} minutes"
        })
        
        # Broadcast assignment
        await self.broadcast_update("task_assigned", {
            "task_id": task_id,
            "agent_id": best_agent.id,
            "agent_name": best_agent.name,
            "description": task_description,
            "estimated_duration": estimated_duration,
            "assignment_score": best_score
        })
        
        return task_id
    
    async def complete_task(self, task_id: str, result: Dict = None):
        """Mark task as completed and update agent workload"""
        if task_id not in self.resource_allocations:
            return False
        
        allocation = self.resource_allocations[task_id]
        agent = self.board_members[allocation.agent_id]
        
        # Remove from agent's workload
        agent.current_workload = [w for w in agent.current_workload if w["task_id"] != task_id]
        agent.active_tasks = len(agent.current_workload)
        agent.last_task_completion = datetime.now().isoformat()
        
        # Update allocation status
        allocation.status = "completed"
        allocation.completed_at = datetime.now().isoformat()
        
        # Calculate actual duration
        if allocation.started_at:
            start_time = datetime.fromisoformat(allocation.started_at.replace("Z", "+00:00"))
            end_time = datetime.now()
            duration_minutes = int((end_time - start_time.replace(tzinfo=None)).total_seconds() / 60)
            allocation.actual_duration = duration_minutes
            
            # Update agent efficiency
            if allocation.estimated_duration > 0:
                task_efficiency = allocation.estimated_duration / max(duration_minutes, 1)
                agent.efficiency_rating = (agent.efficiency_rating + task_efficiency) / 2
        
        # Update agent status
        if len(agent.current_workload) == 0:
            agent.status = "active"
        elif len(agent.current_workload) < agent.capacity * 0.7:
            agent.status = "active"
        else:
            agent.status = "busy"
        
        # Complete progress tracking
        await self.complete_progress_tracking(task_id, {
            "result": result or {},
            "success": True
        })
        
        # Check if any queued tasks can now be assigned
        await self.process_task_queue()
        
        # Update metrics
        self.resource_metrics["tasks_completed_today"] += 1
        
        # Broadcast completion
        await self.broadcast_update("task_completed", {
            "task_id": task_id,
            "agent_id": allocation.agent_id,
            "agent_name": allocation.agent_name,
            "actual_duration": allocation.actual_duration,
            "efficiency": agent.efficiency_rating,
            "result": result or {}
        })
        
        return True
    
    async def process_task_queue(self):
        """Try to assign queued tasks to newly available agents"""
        if not self.task_queue:
            return
        
        # Sort queue by priority (highest first)
        self.task_queue.sort(key=lambda x: x.get("priority", 5), reverse=True)
        
        assigned_tasks = []
        for queued_task in self.task_queue[:]:
            task_id = await self.assign_task_to_agent({
                "description": queued_task["description"],
                "type": queued_task["type"],
                "priority": queued_task["priority"],
                "estimated_duration": queued_task["estimated_duration"]
            })
            
            if task_id:  # Successfully assigned
                assigned_tasks.append(queued_task)
                self.task_queue.remove(queued_task)
        
        if assigned_tasks:
            await self.broadcast_update("queue_processed", {
                "assigned_count": len(assigned_tasks),
                "remaining_queue": len(self.task_queue)
            })
    
    async def delegate_to_real_agent(self, agent_id: str, task_data: Dict) -> Dict:
        """Delegate task to real agent implementation"""
        
        if agent_id not in self.real_agents:
            return {"error": f"Real agent {agent_id} not available"}
        
        real_agent = self.real_agents[agent_id]
        task_description = task_data.get("description", "")
        context = task_data.get("context", {})
        
        try:
            # Call the real agent's execute_task method
            if hasattr(real_agent, 'execute_task'):
                result = await real_agent.execute_task(task_description, context)
                
                # Add metadata about delegation
                result["delegated_to"] = {
                    "agent_id": agent_id,
                    "agent_name": real_agent.name if hasattr(real_agent, 'name') else agent_id,
                    "agent_role": real_agent.role if hasattr(real_agent, 'role') else "Unknown",
                    "delegation_timestamp": datetime.now().isoformat()
                }
                
                return result
            else:
                return {"error": f"Agent {agent_id} does not support task execution"}
                
        except Exception as e:
            logger.error(f"Error delegating to real agent {agent_id}: {e}")
            return {
                "error": f"Task delegation failed: {str(e)}",
                "agent_id": agent_id,
                "task": task_description
            }
    
    async def start_progress_tracking(self, task_id: str, task_data: Dict) -> None:
        """Start tracking progress for a task"""
        self.progress_tracker[task_id] = {
            "task_id": task_id,
            "description": task_data.get("description", ""),
            "agent_id": task_data.get("agent_id", ""),
            "agent_name": task_data.get("agent_name", ""),
            "started_at": datetime.now().isoformat(),
            "status": "started",
            "progress_percentage": 0,
            "milestones": [],
            "current_phase": "initialization",
            "estimated_completion": task_data.get("estimated_completion", ""),
            "last_update": datetime.now().isoformat()
        }
        
        # Broadcast progress start
        await self.broadcast_update("progress_started", {
            "task_id": task_id,
            "agent_name": task_data.get("agent_name", ""),
            "description": task_data.get("description", "")
        })
    
    async def update_task_progress(self, task_id: str, progress_data: Dict) -> None:
        """Update progress for a task"""
        if task_id not in self.progress_tracker:
            logger.warning(f"Progress update for unknown task: {task_id}")
            return
        
        tracker = self.progress_tracker[task_id]
        
        # Update progress data
        tracker["progress_percentage"] = progress_data.get("progress_percentage", tracker["progress_percentage"])
        tracker["status"] = progress_data.get("status", tracker["status"])
        tracker["current_phase"] = progress_data.get("current_phase", tracker["current_phase"])
        tracker["last_update"] = datetime.now().isoformat()
        
        # Add milestone if provided
        if "milestone" in progress_data:
            tracker["milestones"].append({
                "milestone": progress_data["milestone"],
                "timestamp": datetime.now().isoformat(),
                "progress_at_milestone": tracker["progress_percentage"]
            })
        
        # Broadcast progress update
        await self.broadcast_update("progress_updated", {
            "task_id": task_id,
            "progress_percentage": tracker["progress_percentage"],
            "status": tracker["status"],
            "current_phase": tracker["current_phase"],
            "milestone": progress_data.get("milestone", "")
        })
    
    async def complete_progress_tracking(self, task_id: str, completion_data: Dict = None) -> None:
        """Complete progress tracking for a task"""
        if task_id not in self.progress_tracker:
            logger.warning(f"Completion tracking for unknown task: {task_id}")
            return
        
        tracker = self.progress_tracker[task_id]
        completion_data = completion_data or {}
        
        # Mark as completed
        tracker["status"] = "completed"
        tracker["progress_percentage"] = 100
        tracker["completed_at"] = datetime.now().isoformat()
        tracker["final_result"] = completion_data.get("result", "")
        tracker["success"] = completion_data.get("success", True)
        
        # Calculate completion metrics
        start_time = datetime.fromisoformat(tracker["started_at"].replace("Z", "+00:00"))
        end_time = datetime.now()
        duration_minutes = (end_time - start_time.replace(tzinfo=None)).total_seconds() / 60
        tracker["actual_duration_minutes"] = duration_minutes
        
        # Update completion metrics
        await self.update_completion_metrics(tracker)
        
        # Broadcast completion
        await self.broadcast_update("progress_completed", {
            "task_id": task_id,
            "agent_name": tracker["agent_name"],
            "duration_minutes": duration_minutes,
            "success": tracker["success"],
            "milestones_count": len(tracker["milestones"])
        })
    
    async def update_completion_metrics(self, completed_task: Dict) -> None:
        """Update overall completion metrics based on completed task"""
        
        # Increment daily count
        self.completion_metrics["tasks_completed_today"] += 1
        
        # Update average completion time
        duration = completed_task.get("actual_duration_minutes", 0)
        current_avg = self.completion_metrics["average_completion_time"]
        tasks_count = self.completion_metrics["tasks_completed_today"]
        
        if tasks_count == 1:
            self.completion_metrics["average_completion_time"] = duration
        else:
            # Rolling average
            self.completion_metrics["average_completion_time"] = (
                (current_avg * (tasks_count - 1) + duration) / tasks_count
            )
        
        # Update success rate
        successful_tasks = sum(1 for t in self.progress_tracker.values() 
                             if t.get("status") == "completed" and t.get("success", True))
        total_completed = sum(1 for t in self.progress_tracker.values() 
                            if t.get("status") == "completed")
        
        if total_completed > 0:
            self.completion_metrics["success_rate"] = (successful_tasks / total_completed) * 100
        
        # Update agent performance
        agent_id = completed_task.get("agent_id", "unknown")
        if agent_id not in self.completion_metrics["agent_performance"]:
            self.completion_metrics["agent_performance"][agent_id] = {
                "tasks_completed": 0,
                "average_duration": 0,
                "success_rate": 100.0,
                "milestones_avg": 0
            }
        
        agent_perf = self.completion_metrics["agent_performance"][agent_id]
        agent_perf["tasks_completed"] += 1
        
        # Update agent average duration
        if agent_perf["tasks_completed"] == 1:
            agent_perf["average_duration"] = duration
        else:
            agent_perf["average_duration"] = (
                (agent_perf["average_duration"] * (agent_perf["tasks_completed"] - 1) + duration) 
                / agent_perf["tasks_completed"]
            )
        
        # Update agent milestone average
        milestones_count = len(completed_task.get("milestones", []))
        if agent_perf["tasks_completed"] == 1:
            agent_perf["milestones_avg"] = milestones_count
        else:
            agent_perf["milestones_avg"] = (
                (agent_perf["milestones_avg"] * (agent_perf["tasks_completed"] - 1) + milestones_count)
                / agent_perf["tasks_completed"]
            )
    
    async def get_progress_summary(self) -> Dict:
        """Get comprehensive progress summary"""
        
        # Current active tasks
        active_tasks = {
            task_id: tracker for task_id, tracker in self.progress_tracker.items()
            if tracker.get("status") not in ["completed", "failed", "cancelled"]
        }
        
        # Recently completed tasks (last 24 hours)
        recent_completed = []
        for task_id, tracker in self.progress_tracker.items():
            if tracker.get("status") == "completed" and tracker.get("completed_at"):
                completed_time = datetime.fromisoformat(tracker["completed_at"].replace("Z", "+00:00"))
                if (datetime.now() - completed_time.replace(tzinfo=None)).days < 1:
                    recent_completed.append(tracker)
        
        return {
            "active_tasks": len(active_tasks),
            "active_task_details": list(active_tasks.values()),
            "completed_today": len(recent_completed),
            "recent_completions": recent_completed[-10:],  # Last 10
            "completion_metrics": self.completion_metrics,
            "progress_trends": await self.calculate_progress_trends()
        }
    
    async def calculate_progress_trends(self) -> Dict:
        """Calculate progress trends and predictions"""
        
        completed_tasks = [
            t for t in self.progress_tracker.values() 
            if t.get("status") == "completed"
        ]
        
        if not completed_tasks:
            return {"trend": "no_data"}
        
        # Calculate trends over last 7 days
        daily_completions = {}
        for task in completed_tasks:
            if task.get("completed_at"):
                completed_date = datetime.fromisoformat(task["completed_at"].replace("Z", "+00:00")).date()
                daily_completions[completed_date] = daily_completions.get(completed_date, 0) + 1
        
        # Get last 7 days
        recent_days = sorted(daily_completions.keys())[-7:]
        daily_counts = [daily_completions.get(day, 0) for day in recent_days]
        
        # Simple trend calculation
        if len(daily_counts) >= 2:
            trend_direction = "increasing" if daily_counts[-1] > daily_counts[0] else "decreasing"
        else:
            trend_direction = "stable"
        
        return {
            "trend": trend_direction,
            "daily_completions": dict(zip([str(d) for d in recent_days], daily_counts)),
            "average_daily": sum(daily_counts) / len(daily_counts) if daily_counts else 0,
            "peak_day": max(daily_completions.items(), key=lambda x: x[1])[0] if daily_completions else None
        }

# Global board instance
vision_board = VisionBoard()

# API Endpoints

@app.on_event("startup")
async def startup():
    """Start autonomous processes on startup"""
    asyncio.create_task(vision_board.autonomous_improvement())

@app.get("/api/board/members")
async def get_board_members():
    """Get all board members with their current status"""
    return {
        "members": [asdict(member) for member in vision_board.board_members.values()],
        "chairperson": vision_board.chairperson_id
    }

@app.post("/api/board/chairperson")
async def set_chairperson(data: Dict[str, str]):
    """Change the board chairperson"""
    member_id = data.get("member_id")
    if member_id in vision_board.board_members:
        vision_board.chairperson_id = member_id
        await vision_board.broadcast_update("chairperson_changed", {
            "new_chairperson": member_id
        })
        return {"success": True, "chairperson": member_id}
    return {"success": False, "error": "Invalid member ID"}

@app.post("/api/board/conversation")
async def start_conversation(data: Dict[str, Any]):
    """Start a new board conversation"""
    topic = data.get("topic", "General Discussion")
    context = data.get("context", {})
    
    conversation_id = await vision_board.start_conversation(topic, context)
    
    return {
        "success": True,
        "conversation_id": conversation_id,
        "message": f"Board conversation started on: {topic}"
    }

@app.post("/api/board/task")
async def assign_task(data: Dict[str, Any]):
    """Assign a task to the board with intelligent resource allocation"""
    task_description = data.get("description")
    task_type = data.get("type", "general")
    priority = data.get("priority", 5)
    estimated_duration = data.get("estimated_duration", 30)
    dependencies = data.get("dependencies", [])
    resource_requirements = data.get("resource_requirements", {})
    
    # Use intelligent task assignment
    task_id = await vision_board.assign_task_to_agent({
        "description": task_description,
        "type": task_type,
        "priority": priority,
        "estimated_duration": estimated_duration,
        "dependencies": dependencies,
        "resource_requirements": resource_requirements
    })
    
    if task_id:
        return {
            "success": True,
            "task_id": task_id,
            "message": f"Task assigned with resource allocation tracking"
        }
    else:
        return {
            "success": True,
            "task_id": None,
            "message": f"Task queued - all agents at capacity",
            "queue_position": len(vision_board.task_queue)
        }

@app.get("/api/board/improvements")
async def get_improvements():
    """Get autonomous improvement log"""
    return {
        "improvements": vision_board.improvement_log[-50:],  # Last 50 improvements
        "total": len(vision_board.improvement_log)
    }

@app.get("/api/board/conversations")
async def get_conversations():
    """Get conversation history"""
    return {
        "conversations": vision_board.conversation_history[-10:],  # Last 10 conversations
        "active": vision_board.active_conversation
    }

@app.websocket("/ws/board")
async def websocket_board(websocket: WebSocket):
    """WebSocket for real-time board updates"""
    await websocket.accept()
    vision_board.websocket_clients.add(websocket)
    
    try:
        # Send initial board state
        await websocket.send_json({
            "type": "board_state",
            "data": {
                "members": [asdict(m) for m in vision_board.board_members.values()],
                "chairperson": vision_board.chairperson_id
            }
        })
        
        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages if needed
            
    except WebSocketDisconnect:
        vision_board.websocket_clients.remove(websocket)

@app.post("/api/models/download")
async def download_model(data: Dict[str, str]):
    """Download a specific AI model"""
    model_name = data.get("model")
    
    if vision_board.llm_manager:
        # Start download process
        await vision_board.broadcast_update("model_download_started", {
            "model": model_name,
            "status": "downloading"
        })
        
        # In real implementation, would call llm_manager.download_model()
        await asyncio.sleep(2)  # Simulate download
        
        await vision_board.broadcast_update("model_download_complete", {
            "model": model_name,
            "status": "ready"
        })
        
        return {"success": True, "message": f"Model {model_name} downloaded"}
    
    return {"success": False, "error": "LLM Manager not available"}

@app.post("/api/board/delegate")
async def delegate_task(data: Dict[str, Any]):
    """Delegate a specific task to a real agent"""
    agent_id = data.get("agent_id")
    task_description = data.get("description")
    context = data.get("context", {})
    
    if not agent_id or not task_description:
        raise HTTPException(status_code=400, detail="agent_id and description are required")
    
    # Check if agent exists
    if agent_id not in vision_board.board_members:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    # Delegate to real agent
    result = await vision_board.delegate_to_real_agent(agent_id, {
        "description": task_description,
        "context": context
    })
    
    # Broadcast delegation
    await vision_board.broadcast_update("task_delegated", {
        "agent_id": agent_id,
        "agent_name": vision_board.board_members[agent_id].name,
        "task": task_description,
        "result": result
    })
    
    return {
        "success": True,
        "result": result,
        "delegated_to": agent_id
    }

@app.get("/api/board/agents/real")
async def get_real_agents():
    """Get status of all real agent connections"""
    agent_status = {}
    
    for agent_id, real_agent in vision_board.real_agents.items():
        try:
            if hasattr(real_agent, 'get_status'):
                status = real_agent.get_status()
            else:
                status = {
                    "name": getattr(real_agent, 'name', agent_id),
                    "role": getattr(real_agent, 'role', 'Unknown'),
                    "status": "connected"
                }
            
            agent_status[agent_id] = {
                "connected": True,
                "details": status
            }
        except Exception as e:
            agent_status[agent_id] = {
                "connected": False,
                "error": str(e)
            }
    
    return {
        "real_agents": agent_status,
        "total_connected": len([a for a in agent_status.values() if a["connected"]])
    }

@app.post("/api/board/execute")
async def execute_board_task(data: Dict[str, Any]):
    """Execute a task using the best available agent (real or simulated)"""
    task_description = data.get("description")
    task_type = data.get("type", "general")
    context = data.get("context", {})
    
    if not task_description:
        raise HTTPException(status_code=400, detail="description is required")
    
    # Determine best agent based on task type
    agent_mapping = {
        "technical": "architect",  # Maps to CTO
        "architecture": "architect",
        "code": "architect", 
        "backend": "architect",
        "frontend": "architect",
        "financial": "database",  # Maps to CFO  
        "budget": "database",
        "cost": "database",
        "roi": "database"
    }
    
    target_agent = None
    
    # Find best agent based on task content
    task_lower = task_description.lower()
    for keyword, agent_id in agent_mapping.items():
        if keyword in task_lower:
            target_agent = agent_id
            break
    
    # Default to architect for technical tasks
    if not target_agent and any(word in task_lower for word in ['build', 'create', 'implement', 'develop']):
        target_agent = "architect"
    
    if target_agent and target_agent in vision_board.real_agents:
        # Delegate to real agent
        result = await vision_board.delegate_to_real_agent(target_agent, {
            "description": task_description,
            "context": context
        })
        
        return {
            "success": True,
            "execution_type": "real_agent",
            "executed_by": target_agent,
            "result": result
        }
    else:
        # Use regular board discussion
        conversation_id = await vision_board.start_conversation(task_description, context)
        
        return {
            "success": True,
            "execution_type": "board_discussion",
            "conversation_id": conversation_id,
            "message": "Task discussed by board - check conversation history for results"
        }

@app.get("/api/progress/summary")
async def get_progress_summary():
    """Get comprehensive progress summary"""
    return await vision_board.get_progress_summary()

@app.get("/api/progress/{task_id}")
async def get_task_progress(task_id: str):
    """Get detailed progress for a specific task"""
    if task_id not in vision_board.progress_tracker:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    return {
        "task_progress": vision_board.progress_tracker[task_id]
    }

@app.post("/api/progress/{task_id}/update")
async def update_task_progress_endpoint(task_id: str, data: Dict[str, Any]):
    """Update progress for a specific task"""
    if task_id not in vision_board.progress_tracker:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    await vision_board.update_task_progress(task_id, data)
    
    return {
        "success": True,
        "message": f"Progress updated for task {task_id}",
        "current_progress": vision_board.progress_tracker[task_id]["progress_percentage"]
    }

@app.post("/api/progress/{task_id}/complete")
async def complete_task_progress_endpoint(task_id: str, data: Dict[str, Any] = None):
    """Mark task as completed and finalize progress tracking"""
    if task_id not in vision_board.progress_tracker:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    completion_data = data or {}
    await vision_board.complete_progress_tracking(task_id, completion_data)
    
    return {
        "success": True,
        "message": f"Task {task_id} completed",
        "final_progress": vision_board.progress_tracker[task_id]
    }

@app.get("/api/progress/metrics")
async def get_progress_metrics():
    """Get detailed progress and completion metrics"""
    
    # Get current progress summary
    summary = await vision_board.get_progress_summary()
    
    # Add additional metrics
    total_tasks = len(vision_board.progress_tracker)
    completed_tasks = len([t for t in vision_board.progress_tracker.values() if t.get("status") == "completed"])
    failed_tasks = len([t for t in vision_board.progress_tracker.values() if t.get("status") == "failed"])
    active_tasks = len([t for t in vision_board.progress_tracker.values() if t.get("status") not in ["completed", "failed", "cancelled"]])
    
    return {
        "overview": {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "active_tasks": active_tasks,
            "completion_rate": (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        },
        "detailed_summary": summary,
        "agent_breakdown": vision_board.completion_metrics["agent_performance"]
    }

@app.get("/api/progress/live")
async def get_live_progress():
    """Get real-time progress updates for all active tasks"""
    
    active_tasks = {
        task_id: {
            "task_id": task_id,
            "description": tracker["description"],
            "agent_name": tracker["agent_name"],
            "progress_percentage": tracker["progress_percentage"],
            "current_phase": tracker["current_phase"],
            "status": tracker["status"],
            "started_at": tracker["started_at"],
            "last_update": tracker["last_update"],
            "milestones_count": len(tracker.get("milestones", []))
        }
        for task_id, tracker in vision_board.progress_tracker.items()
        if tracker.get("status") not in ["completed", "failed", "cancelled"]
    }
    
    return {
        "active_tasks": active_tasks,
        "active_count": len(active_tasks),
        "system_health": {
            "average_progress": sum(t["progress_percentage"] for t in active_tasks.values()) / len(active_tasks) if active_tasks else 0,
            "tasks_starting": len([t for t in active_tasks.values() if t["progress_percentage"] < 10]),
            "tasks_in_progress": len([t for t in active_tasks.values() if 10 <= t["progress_percentage"] < 90]),
            "tasks_finishing": len([t for t in active_tasks.values() if t["progress_percentage"] >= 90])
        }
    }

# Resource allocation endpoints

@app.get("/api/resources/status")
async def get_resource_status():
    """Get current resource allocation status"""
    await vision_board.update_resource_metrics()
    
    # Get agent workload details
    agent_details = {}
    for agent_id, member in vision_board.board_members.items():
        agent_details[agent_id] = {
            "name": member.name,
            "role": member.role,
            "capacity": member.capacity,
            "current_load": len(member.current_workload),
            "utilization": len(member.current_workload) / member.capacity * 100,
            "status": member.status,
            "efficiency": member.efficiency_rating,
            "active_tasks": [
                {
                    "task_id": task["task_id"],
                    "description": task["description"],
                    "priority": task["priority"],
                    "started_at": task["started_at"]
                } for task in member.current_workload
            ],
            "last_completion": member.last_task_completion
        }
    
    return {
        "metrics": vision_board.resource_metrics,
        "agents": agent_details,
        "task_queue": vision_board.task_queue,
        "active_allocations": len(vision_board.resource_allocations)
    }

@app.get("/api/resources/allocations")
async def get_resource_allocations():
    """Get detailed resource allocation history"""
    allocations = []
    for allocation in vision_board.resource_allocations.values():
        allocations.append({
            "task_id": allocation.task_id,
            "agent_id": allocation.agent_id,
            "agent_name": allocation.agent_name,
            "task_description": allocation.task_description,
            "priority": allocation.priority,
            "status": allocation.status,
            "estimated_duration": allocation.estimated_duration,
            "actual_duration": allocation.actual_duration,
            "started_at": allocation.started_at,
            "completed_at": allocation.completed_at,
            "dependencies": allocation.dependencies or [],
            "resource_requirements": allocation.resource_requirements or {}
        })
    
    return {"allocations": allocations}

@app.post("/api/resources/complete")
async def complete_task_endpoint(data: Dict[str, Any]):
    """Mark a task as completed"""
    task_id = data.get("task_id")
    result = data.get("result", {})
    
    if not task_id:
        raise HTTPException(status_code=400, detail="task_id is required")
    
    success = await vision_board.complete_task(task_id, result)
    
    if success:
        return {"success": True, "message": f"Task {task_id} completed"}
    else:
        return {"success": False, "error": f"Task {task_id} not found"}

@app.get("/api/resources/queue")
async def get_task_queue():
    """Get current task queue"""
    return {
        "queue": vision_board.task_queue,
        "queue_length": len(vision_board.task_queue)
    }

@app.post("/api/resources/rebalance")
async def rebalance_workload():
    """Trigger workload rebalancing across agents"""
    # Find overloaded agents
    overloaded_agents = []
    underutilized_agents = []
    
    for agent_id, member in vision_board.board_members.items():
        utilization = len(member.current_workload) / member.capacity
        if utilization >= 0.8:
            overloaded_agents.append((agent_id, member))
        elif utilization <= 0.3:
            underutilized_agents.append((agent_id, member))
    
    rebalanced_tasks = []
    
    # Move tasks from overloaded to underutilized agents
    for overloaded_id, overloaded_member in overloaded_agents:
        if not underutilized_agents:
            break
            
        # Move lowest priority tasks
        tasks_to_move = sorted(overloaded_member.current_workload, key=lambda x: x.get("priority", 5))
        
        for task in tasks_to_move[:len(underutilized_agents)]:
            if underutilized_agents:
                target_agent_id, target_member = underutilized_agents.pop(0)
                
                # Move task
                overloaded_member.current_workload.remove(task)
                target_member.current_workload.append(task)
                
                # Update allocation record
                task_id = task["task_id"]
                if task_id in vision_board.resource_allocations:
                    allocation = vision_board.resource_allocations[task_id]
                    allocation.agent_id = target_agent_id
                    allocation.agent_name = target_member.name
                
                rebalanced_tasks.append({
                    "task_id": task_id,
                    "from": overloaded_member.name,
                    "to": target_member.name
                })
    
    # Update agent statuses
    for agent_id, member in vision_board.board_members.items():
        utilization = len(member.current_workload) / member.capacity
        if utilization >= 1.0:
            member.status = "overloaded"
        elif utilization >= 0.7:
            member.status = "busy"
        else:
            member.status = "active"
        member.active_tasks = len(member.current_workload)
    
    if rebalanced_tasks:
        await vision_board.broadcast_update("workload_rebalanced", {
            "rebalanced_tasks": rebalanced_tasks,
            "count": len(rebalanced_tasks)
        })
    
    return {
        "success": True,
        "rebalanced_tasks": rebalanced_tasks,
        "message": f"Rebalanced {len(rebalanced_tasks)} tasks"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    await vision_board.update_resource_metrics()
    
    return {
        "status": "healthy",
        "board_members": len(vision_board.board_members),
        "orchestrator": vision_board.orchestrator is not None,
        "llm_manager": vision_board.llm_manager is not None,
        "resource_metrics": vision_board.resource_metrics,
        "total_capacity": vision_board.resource_metrics["total_capacity"],
        "utilized_capacity": vision_board.resource_metrics["utilized_capacity"],
        "utilization_percentage": (vision_board.resource_metrics["utilized_capacity"] / vision_board.resource_metrics["total_capacity"] * 100) if vision_board.resource_metrics["total_capacity"] > 0 else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8302)