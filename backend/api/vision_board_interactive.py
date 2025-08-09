#!/usr/bin/env python3
"""
Interactive Vision Board API - Simplified but functional version
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Set, Any
from datetime import datetime
import asyncio
import json
import uuid
import logging
import sys
from pathlib import Path

# Add agents directory to path
sys.path.insert(0, str(Path(__file__).parent / ".." / "agents"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Interactive Vision Board", version="2.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InteractiveVisionBoard:
    def __init__(self):
        self.board_members = self._init_board_members()
        self.websocket_clients: Set[WebSocket] = set()
        self.active_conversations = {}
        self.user_decisions = {}
        self.real_agents = {}
        self._initialize_real_agents()
        
    def _init_board_members(self):
        """Initialize board members"""
        return {
            "vision-ceo": {
                "id": "vision-ceo",
                "name": "Alexandra Vision",
                "role": "CEO",
                "avatar": "👁️",
                "status": "active",
                "expertise": ["strategy", "leadership"]
            },
            "tech-cto": {
                "id": "tech-cto",
                "name": "Marcus Tech",
                "role": "CTO",
                "avatar": "🏗️",
                "status": "active",
                "expertise": ["architecture", "development"]
            },
            "security-cso": {
                "id": "security-cso",
                "name": "David Security",
                "role": "CSO",
                "avatar": "🔒",
                "status": "active",
                "expertise": ["security", "compliance"]
            },
            "ops-coo": {
                "id": "ops-coo",
                "name": "Lisa Operations",
                "role": "COO",
                "avatar": "⚙️",
                "status": "active",
                "expertise": ["deployment", "monitoring"]
            },
            "data-cdo": {
                "id": "data-cdo",
                "name": "James Data",
                "role": "CDO",
                "avatar": "💾",
                "status": "active",
                "expertise": ["database", "analytics"]
            },
            "creative-cco": {
                "id": "creative-cco",
                "name": "Emma Creative",
                "role": "CCO",
                "avatar": "🎨",
                "status": "active",
                "expertise": ["design", "ux"]
            }
        }
    
    def _initialize_real_agents(self):
        """Initialize real agent connections"""
        try:
            # Import and initialize CTO agent
            from cto_agent import CTOAgent
            self.real_agents["tech-cto"] = CTOAgent()
            logger.info("CTO Agent initialized")
        except Exception as e:
            logger.warning(f"Could not initialize CTO Agent: {e}")
        
        try:
            # Import and initialize CFO agent
            from cfo_agent import CFOAgent
            self.real_agents["finance-cfo"] = CFOAgent()
            # Update board members to include CFO
            self.board_members["finance-cfo"] = {
                "id": "finance-cfo",
                "name": "Sarah Finance",
                "role": "CFO", 
                "avatar": "💰",
                "status": "active",
                "expertise": ["budget", "roi", "financial planning"]
            }
            logger.info("CFO Agent initialized")
        except Exception as e:
            logger.warning(f"Could not initialize CFO Agent: {e}")
    
    def analyze_task(self, topic: str) -> Dict:
        """Simple task analysis"""
        topic_lower = topic.lower()
        
        # Determine domain and complexity
        if 'budget' in topic_lower or 'cost' in topic_lower or 'financial' in topic_lower or 'roi' in topic_lower:
            return {"domain": "financial", "complexity": "medium", "chair": "finance-cfo"}
        elif 'security' in topic_lower or 'auth' in topic_lower:
            return {"domain": "security", "complexity": "high", "chair": "security-cso"}
        elif 'deploy' in topic_lower or 'infrastructure' in topic_lower:
            return {"domain": "infrastructure", "complexity": "medium", "chair": "ops-coo"}
        elif 'design' in topic_lower or 'ui' in topic_lower:
            return {"domain": "creative", "complexity": "medium", "chair": "creative-cco"}
        elif 'data' in topic_lower or 'database' in topic_lower:
            return {"domain": "data", "complexity": "medium", "chair": "data-cdo"}
        elif 'build' in topic_lower or 'develop' in topic_lower:
            return {"domain": "technical", "complexity": "high", "chair": "tech-cto"}
        else:
            return {"domain": "strategic", "complexity": "medium", "chair": "vision-ceo"}
    
    def identify_resources(self, task_analysis: Dict) -> Dict:
        """Identify required resources"""
        domain = task_analysis["domain"]
        
        resources = {
            "team": [],
            "tools": [],
            "timeline": "",
            "budget": ""
        }
        
        # Add team members based on domain
        if domain == "financial":
            resources["team"] = ["finance-cfo", "vision-ceo", "ops-coo"]
            resources["tools"] = ["Financial Modeling", "Budget Tracking", "ROI Calculator"]
        elif domain == "technical":
            resources["team"] = ["tech-cto", "ops-coo", "data-cdo"]
            resources["tools"] = ["Development Environment", "CI/CD Pipeline"]
        elif domain == "security":
            resources["team"] = ["security-cso", "ops-coo"]
            resources["tools"] = ["Security Scanner", "Compliance Checker"]
        elif domain == "creative":
            resources["team"] = ["creative-cco", "tech-cto"]
            resources["tools"] = ["Design Tools", "User Testing"]
        else:
            resources["team"] = ["vision-ceo", "tech-cto", "ops-coo"]
            resources["tools"] = ["Planning Tools", "Analytics"]
        
        # Set timeline based on complexity
        if task_analysis["complexity"] == "high":
            resources["timeline"] = "1-2 weeks"
            resources["budget"] = "$10,000+"
        elif task_analysis["complexity"] == "medium":
            resources["timeline"] = "3-5 days"
            resources["budget"] = "$2,000-5,000"
        else:
            resources["timeline"] = "1-2 days"
            resources["budget"] = "$500-1,000"
        
        return resources
    
    async def broadcast(self, message: Dict):
        """Send message to all WebSocket clients"""
        msg_str = json.dumps(message)
        disconnected = set()
        
        for ws in self.websocket_clients:
            try:
                await ws.send_text(msg_str)
            except:
                disconnected.add(ws)
        
        self.websocket_clients -= disconnected
    
    async def start_conversation(self, topic: str) -> str:
        """Start interactive board conversation"""
        conversation_id = str(uuid.uuid4())
        
        # Phase 1: Task Analysis
        await self.broadcast({
            "type": "conversation_started",
            "data": {
                "conversation_id": conversation_id,
                "topic": topic,
                "timestamp": datetime.now().isoformat()
            }
        })
        
        await asyncio.sleep(1)
        
        task_analysis = self.analyze_task(topic)
        chair = self.board_members[task_analysis["chair"]]
        
        # Announce chair selection
        await self.broadcast({
            "type": "member_opinion",
            "data": {
                "member_id": "system",
                "member_name": "Board System",
                "role": "Moderator",
                "opinion": f"Based on the topic '{topic}', I'm selecting {chair['name']} ({chair['role']}) as chair. Domain: {task_analysis['domain']}, Complexity: {task_analysis['complexity']}",
                "timestamp": datetime.now().isoformat()
            }
        })
        
        await asyncio.sleep(2)
        
        # Phase 2: Chair presents initial plan
        chair_plan = f"As {chair['role']}, I'll lead this {task_analysis['complexity']} complexity {task_analysis['domain']} initiative. Let me outline our approach."
        
        await self.broadcast({
            "type": "member_opinion",
            "data": {
                "member_id": chair["id"],
                "member_name": chair["name"],
                "role": chair["role"],
                "opinion": chair_plan,
                "timestamp": datetime.now().isoformat()
            }
        })
        
        await asyncio.sleep(2)
        
        # Phase 3: Resource identification
        resources = self.identify_resources(task_analysis)
        
        await self.broadcast({
            "type": "member_opinion",
            "data": {
                "member_id": chair["id"],
                "member_name": chair["name"],
                "role": chair["role"],
                "opinion": f"Required resources: Team of {len(resources['team'])} members, Timeline: {resources['timeline']}, Budget: {resources['budget']}",
                "timestamp": datetime.now().isoformat()
            }
        })
        
        await asyncio.sleep(2)
        
        # Phase 4: Get input from relevant team members
        for member_id in resources["team"]:
            if member_id != task_analysis["chair"]:
                member = self.board_members[member_id]
                
                # Update status
                await self.broadcast({
                    "type": "member_status",
                    "data": {
                        "member_id": member_id,
                        "status": "speaking"
                    }
                })
                
                # Generate department-specific input using real agents when available
                opinion = await self._generate_agent_response(member_id, topic, task_analysis, resources)
                
                await self.broadcast({
                    "type": "member_opinion",
                    "data": {
                        "member_id": member_id,
                        "member_name": member["name"],
                        "role": member["role"],
                        "opinion": opinion,
                        "timestamp": datetime.now().isoformat()
                    }
                })
                
                await asyncio.sleep(2)
        
        # Phase 5: Present decision points
        decisions = [
            {
                "id": "approach",
                "question": "Select implementation approach:",
                "options": [
                    {"value": "agile", "label": "Agile - Iterative development", "icon": "🔄"},
                    {"value": "waterfall", "label": "Waterfall - Sequential phases", "icon": "📊"},
                    {"value": "hybrid", "label": "Hybrid - Best of both", "icon": "🔀"}
                ]
            },
            {
                "id": "priority",
                "question": "Set priority level:",
                "options": [
                    {"value": "urgent", "label": "Urgent - Start immediately", "icon": "🔴"},
                    {"value": "high", "label": "High - Start this week", "icon": "🟡"},
                    {"value": "normal", "label": "Normal - Scheduled", "icon": "🟢"}
                ]
            },
            {
                "id": "resources",
                "question": "Resource allocation:",
                "options": [
                    {"value": "full", "label": "Full team + budget", "icon": "💯"},
                    {"value": "standard", "label": "Standard allocation", "icon": "✅"},
                    {"value": "minimal", "label": "Minimal viable", "icon": "⚡"}
                ]
            }
        ]
        
        await self.broadcast({
            "type": "member_opinion",
            "data": {
                "member_id": chair["id"],
                "member_name": chair["name"],
                "role": chair["role"],
                "opinion": "Now I need your input on key decisions:",
                "decisions": decisions,
                "timestamp": datetime.now().isoformat()
            }
        })
        
        # Store conversation state
        self.active_conversations[conversation_id] = {
            "topic": topic,
            "task_analysis": task_analysis,
            "resources": resources,
            "chair": chair["id"],
            "decisions_pending": True
        }
        
        return conversation_id
    
    async def _generate_agent_response(self, member_id: str, topic: str, task_analysis: Dict, resources: Dict) -> str:
        """Generate response using real agent or fallback to simulated response"""
        
        # Check if we have a real agent for this member
        if member_id in self.real_agents:
            try:
                agent = self.real_agents[member_id]
                logger.info(f"Using real agent for {member_id}: {topic}")
                
                # Execute task with the agent
                context = {
                    'task_analysis': task_analysis,
                    'resources': resources,
                    'domain': task_analysis.get('domain'),
                    'complexity': task_analysis.get('complexity')
                }
                
                result = await agent.execute_task(topic, context)
                
                # Extract a concise opinion from the result for the chat
                if member_id == "tech-cto" and result.get('status') == 'completed':
                    analysis = result.get('analysis', {})
                    return f"Technical Analysis: {analysis.get('complexity', 'medium')} complexity, estimated {analysis.get('estimated_effort', '3-5 days')}. I'll handle architecture design and implementation. {analysis.get('recommendations', [''])[0] if analysis.get('recommendations') else ''}"
                
                elif member_id == "finance-cfo" and result.get('status') == 'completed':
                    budget_analysis = result.get('budget_analysis', {})
                    if budget_analysis:
                        cost = budget_analysis.get('estimated_cost', 0)
                        roi = budget_analysis.get('roi_analysis', {})
                        approval = budget_analysis.get('approval_status', 'pending')
                        return f"Financial Analysis: Estimated cost ${cost:,.0f}, ROI: {roi.get('three_year_roi_percentage', 0):.1f}% over 3 years. Status: {approval.upper()}. {budget_analysis.get('recommendation', '')}"
                
                # Fallback for other agents or failed execution
                if result.get('status') == 'failed':
                    return f"I encountered an issue analyzing this task: {result.get('error', 'Unknown error')}. Will provide manual assessment."
                else:
                    return f"I've analyzed the task and will execute my role-specific responsibilities."
                    
            except Exception as e:
                logger.error(f"Error using real agent {member_id}: {e}")
                # Fall through to simulated response
        
        # Fallback to simulated responses
        member = self.board_members[member_id]
        role = member["role"]
        
        if role == "CTO":
            return f"From a technical perspective: We'll need proper architecture design, API development, and thorough testing. I recommend microservices approach."
        elif role == "CFO":
            return f"Financial assessment: I'll analyze the budget requirements and ROI projections. Need to ensure this aligns with our financial goals."
        elif role == "CSO":
            return f"Security requirements: We must implement authentication, encryption, and compliance checks. I'll conduct threat modeling."
        elif role == "COO":
            return f"Operations plan: I'll prepare deployment pipelines, monitoring dashboards, and ensure 99.9% uptime SLA."
        elif role == "CDO":
            return f"Data strategy: I'll design the schema, set up analytics, and ensure data governance compliance."
        elif role == "CCO":
            return f"UX approach: I'll create wireframes, ensure dark theme by default (per user preferences), and conduct user testing."
        else:
            return f"I'll contribute my expertise in {', '.join(member['expertise'])} to ensure success."
    
    async def process_decision(self, conversation_id: str, decision_id: str, value: str):
        """Process user decision"""
        if conversation_id not in self.active_conversations:
            return {"error": "Invalid conversation"}
        
        if conversation_id not in self.user_decisions:
            self.user_decisions[conversation_id] = {}
        
        self.user_decisions[conversation_id][decision_id] = value
        
        # Get chair for this conversation
        chair_id = self.active_conversations[conversation_id]["chair"]
        chair = self.board_members[chair_id]
        
        # Generate response based on decision
        responses = {
            "agile": "Excellent! We'll proceed with 2-week sprints and regular demos.",
            "waterfall": "Understood. We'll complete each phase before moving to the next.",
            "hybrid": "Good choice. We'll use sprints for development but maintain phase gates.",
            "urgent": "Mobilizing the team immediately. All hands on deck!",
            "high": "Scheduling kickoff for this week. Team calendars being cleared.",
            "normal": "Added to the roadmap. We'll start as scheduled.",
            "full": "Full resources approved. We have everything we need.",
            "standard": "Standard allocation confirmed. We'll work within constraints.",
            "minimal": "Going lean. We'll focus on core requirements only."
        }
        
        response = responses.get(value, "Decision recorded.")
        
        await self.broadcast({
            "type": "member_opinion",
            "data": {
                "member_id": chair["id"],
                "member_name": chair["name"],
                "role": chair["role"],
                "opinion": f"Decision on {decision_id}: {response}",
                "timestamp": datetime.now().isoformat()
            }
        })
        
        # Check if all decisions are made
        expected_decisions = ["approach", "priority", "resources"]
        if all(d in self.user_decisions.get(conversation_id, {}) for d in expected_decisions):
            await asyncio.sleep(1)
            await self.broadcast({
                "type": "member_opinion",
                "data": {
                    "member_id": chair["id"],
                    "member_name": chair["name"],
                    "role": chair["role"],
                    "opinion": "All decisions received! We're ready to begin execution. I'll coordinate with all departments and provide regular updates.",
                    "timestamp": datetime.now().isoformat()
                }
            })
            
            # Mark conversation as ready for execution
            self.active_conversations[conversation_id]["decisions_pending"] = False
        
        return {"status": "processed", "response": response}

# Initialize board
board = InteractiveVisionBoard()

@app.get("/")
async def root():
    return {"status": "running", "version": "2.0"}

@app.get("/api/board/members")
async def get_members():
    return {
        "members": list(board.board_members.values()),
        "chairperson": "vision-ceo"
    }

@app.post("/api/board/conversation")
async def start_conversation(data: Dict[str, Any]):
    topic = data.get("topic", "General Discussion")
    conversation_id = await board.start_conversation(topic)
    return {
        "success": True,
        "conversation_id": conversation_id
    }

@app.post("/api/board/decision")
async def make_decision(data: Dict[str, Any]):
    conversation_id = data.get("conversation_id")
    decision_id = data.get("decision_id")
    value = data.get("value")
    
    result = await board.process_decision(conversation_id, decision_id, value)
    return {"success": True, "result": result}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    board.websocket_clients.add(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            # Process commands if needed
            message = json.loads(data)
            if message.get("type") == "decision":
                await board.process_decision(
                    message["conversation_id"],
                    message["decision_id"],
                    message["value"]
                )
    except WebSocketDisconnect:
        board.websocket_clients.discard(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8302)