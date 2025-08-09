#!/usr/bin/env python3
"""
Enhanced Vision Board API - Interactive Board of Directors with decision points
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from dataclasses import dataclass, asdict, field
import asyncio
import json
import uuid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Vision Board Enhanced API", version="3.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@dataclass
class BoardMember:
    id: str
    name: str
    role: str  # CEO, CTO, CFO, CMO, COO, CSO, etc.
    department: str
    trust_level: float
    status: str = "listening"  # active, thinking, speaking, listening, leading, analyzing
    current_opinion: str = ""
    avatar: str = ""
    specialization: List[str] = field(default_factory=list)
    active_tasks: int = 0
    expertise_areas: List[str] = field(default_factory=list)

@dataclass
class Decision:
    id: str
    type: str  # approval, resource, priority, approach, technical, strategic
    question: str
    options: List[Dict]
    context: Dict
    required: bool = True
    depends_on: Optional[str] = None

@dataclass
class ConversationState:
    id: str
    topic: str
    phase: str  # analysis, planning, discussion, decision, execution, monitoring
    task_analysis: Dict
    chair: Dict
    resources: Dict
    initial_plan: Dict
    department_inputs: List[Dict]
    pending_decisions: List[Decision]
    user_decisions: Dict = field(default_factory=dict)
    execution_status: Dict = field(default_factory=dict)

class EnhancedVisionBoard:
    """Enhanced Board of Directors with full interactivity"""
    
    def __init__(self):
        self.board_members: Dict[str, BoardMember] = {}
        self.conversation_states: Dict[str, ConversationState] = {}
        self.websocket_clients: Set[WebSocket] = set()
        self.active_tasks: Dict[str, Dict] = {}
        self.initialize_board()
        
    def initialize_board(self):
        """Initialize board members with detailed roles"""
        board_config = [
            BoardMember("vision-ceo", "Alexandra Vision", "CEO", "Executive", 0.95,
                       avatar="👁️", 
                       specialization=["strategy", "leadership", "vision"],
                       expertise_areas=["long-term planning", "stakeholder management", "decision making"]),
            
            BoardMember("tech-cto", "Marcus Tech", "CTO", "Technology", 0.90,
                       avatar="🏗️",
                       specialization=["architecture", "technology", "innovation"],
                       expertise_areas=["system design", "tech stack", "scalability"]),
            
            BoardMember("finance-cfo", "Sarah Finance", "CFO", "Finance", 0.85,
                       avatar="💰",
                       specialization=["budget", "roi", "financial planning"],
                       expertise_areas=["cost analysis", "resource allocation", "financial risk"]),
            
            BoardMember("security-cso", "David Security", "CSO", "Security", 0.92,
                       avatar="🔒",
                       specialization=["security", "compliance", "risk"],
                       expertise_areas=["threat assessment", "compliance", "incident response"]),
            
            BoardMember("ops-coo", "Lisa Operations", "COO", "Operations", 0.88,
                       avatar="⚙️",
                       specialization=["operations", "efficiency", "process"],
                       expertise_areas=["deployment", "monitoring", "optimization"]),
            
            BoardMember("data-cdo", "James Data", "CDO", "Data", 0.82,
                       avatar="💾",
                       specialization=["data", "analytics", "storage"],
                       expertise_areas=["data architecture", "analytics", "governance"]),
            
            BoardMember("creative-cco", "Emma Creative", "CCO", "Creative", 0.78,
                       avatar="🎨",
                       specialization=["design", "ux", "branding"],
                       expertise_areas=["user experience", "visual design", "brand identity"]),
            
            BoardMember("product-cpo", "Michael Product", "CPO", "Product", 0.86,
                       avatar="📦",
                       specialization=["product", "features", "roadmap"],
                       expertise_areas=["product strategy", "feature prioritization", "user needs"])
        ]
        
        for member in board_config:
            self.board_members[member.id] = member
    
    async def analyze_task(self, topic: str) -> Dict:
        """Comprehensive task analysis"""
        topic_lower = topic.lower()
        
        # Domain identification with scoring
        domain_scores = {}
        domains = {
            'technical': ['build', 'deploy', 'code', 'api', 'backend', 'frontend', 'database', 'server'],
            'security': ['security', 'vulnerability', 'breach', 'auth', 'encrypt', 'compliance', 'audit'],
            'financial': ['budget', 'cost', 'roi', 'investment', 'revenue', 'expense', 'funding'],
            'strategic': ['plan', 'strategy', 'roadmap', 'vision', 'goal', 'objective', 'milestone'],
            'operational': ['deploy', 'monitor', 'scale', 'maintain', 'optimize', 'performance'],
            'creative': ['design', 'ui', 'ux', 'interface', 'user experience', 'visual', 'brand'],
            'data': ['data', 'analytics', 'metrics', 'reporting', 'dashboard', 'insights'],
            'product': ['feature', 'product', 'release', 'mvp', 'user story', 'requirement']
        }
        
        for domain, keywords in domains.items():
            score = sum(1 for keyword in keywords if keyword in topic_lower)
            if score > 0:
                domain_scores[domain] = score
        
        primary_domain = max(domain_scores, key=domain_scores.get) if domain_scores else 'strategic'
        
        # Task complexity analysis
        complexity_indicators = {
            'high': ['enterprise', 'complex', 'large-scale', 'critical', 'multi-system', 'integration'],
            'medium': ['standard', 'typical', 'moderate', 'regular'],
            'low': ['simple', 'basic', 'quick', 'minor', 'small']
        }
        
        complexity = 'medium'
        for level, indicators in complexity_indicators.items():
            if any(ind in topic_lower for ind in indicators):
                complexity = level
                break
        
        # Extract specific requirements
        requirements = []
        action_words = {
            'create': 'Design and implement new solution',
            'build': 'Construct complete system',
            'fix': 'Identify and resolve issues',
            'improve': 'Enhance existing functionality',
            'optimize': 'Improve performance and efficiency',
            'secure': 'Implement security measures',
            'deploy': 'Set up production environment',
            'integrate': 'Connect systems and services',
            'analyze': 'Investigate and provide insights',
            'design': 'Create architectural plans',
            'implement': 'Build and deploy solution'
        }
        
        for action, description in action_words.items():
            if action in topic_lower:
                requirements.append(description)
        
        # Identify stakeholders
        stakeholders = []
        if 'user' in topic_lower or 'customer' in topic_lower:
            stakeholders.append('End Users')
        if 'team' in topic_lower or 'developer' in topic_lower:
            stakeholders.append('Development Team')
        if 'business' in topic_lower or 'executive' in topic_lower:
            stakeholders.append('Business Leadership')
        
        return {
            'topic': topic,
            'primary_domain': primary_domain,
            'domain_scores': domain_scores,
            'complexity': complexity,
            'urgency': 'high' if any(u in topic_lower for u in ['urgent', 'asap', 'critical']) else 'normal',
            'requirements': requirements or ['Analyze and provide comprehensive solution'],
            'stakeholders': stakeholders or ['Internal Team'],
            'estimated_impact': 'high' if complexity == 'high' or 'critical' in topic_lower else 'medium',
            'risk_level': 'high' if 'security' in primary_domain or 'critical' in topic_lower else 'medium'
        }
    
    def select_chair(self, task_analysis: Dict) -> BoardMember:
        """Select optimal chair based on task analysis"""
        domain = task_analysis['primary_domain']
        
        # Strategic chair selection based on primary domain
        chair_mapping = {
            'strategic': 'vision-ceo',
            'technical': 'tech-cto',
            'financial': 'finance-cfo',
            'security': 'security-cso',
            'operational': 'ops-coo',
            'data': 'data-cdo',
            'creative': 'creative-cco',
            'product': 'product-cpo'
        }
        
        chair_id = chair_mapping.get(domain, 'vision-ceo')
        chair = self.board_members[chair_id]
        
        # Validate chair selection based on trust level and expertise
        if chair.trust_level < 0.7 and task_analysis['complexity'] == 'high':
            # Fall back to CEO for high complexity tasks if trust is low
            chair = self.board_members['vision-ceo']
        
        return chair
    
    async def identify_resources(self, task_analysis: Dict) -> Dict:
        """Identify all required resources"""
        resources = {
            'required_members': [],
            'required_services': [],
            'required_tools': [],
            'estimated_time': '',
            'estimated_budget': '',
            'dependencies': [],
            'risks': [],
            'success_factors': []
        }
        
        # Determine required board members
        domain_scores = task_analysis.get('domain_scores', {})
        for domain, score in domain_scores.items():
            if score > 0:
                if domain == 'technical':
                    resources['required_members'].extend(['tech-cto', 'ops-coo'])
                    resources['required_services'].extend(['Development Environment', 'CI/CD Pipeline'])
                elif domain == 'security':
                    resources['required_members'].append('security-cso')
                    resources['required_tools'].append('Security Scanning Tools')
                elif domain == 'financial':
                    resources['required_members'].append('finance-cfo')
                    resources['required_tools'].append('Financial Analysis Tools')
                elif domain == 'data':
                    resources['required_members'].append('data-cdo')
                    resources['required_services'].extend(['PostgreSQL', 'Analytics Platform'])
        
        # Time estimation
        complexity_time = {
            'low': {'time': '2-4 hours', 'budget': '$500-1000'},
            'medium': {'time': '2-5 days', 'budget': '$2000-5000'},
            'high': {'time': '1-3 weeks', 'budget': '$10000+'}
        }
        
        estimates = complexity_time[task_analysis['complexity']]
        resources['estimated_time'] = estimates['time']
        resources['estimated_budget'] = estimates['budget']
        
        # Dependencies and risks
        if task_analysis['complexity'] == 'high':
            resources['dependencies'].extend(['Executive approval', 'Resource allocation', 'Team availability'])
            resources['risks'].extend(['Scope creep', 'Technical debt', 'Timeline delays'])
        
        resources['success_factors'] = [
            'Clear requirements',
            'Stakeholder alignment',
            'Adequate resources',
            'Regular monitoring'
        ]
        
        return resources
    
    async def generate_initial_plan(self, chair: BoardMember, task_analysis: Dict, resources: Dict) -> Dict:
        """Chair generates comprehensive initial plan"""
        plan = {
            'executive_summary': f"As {chair.role}, I propose a {task_analysis['complexity']} complexity solution for: {task_analysis['topic']}",
            'objectives': task_analysis['requirements'],
            'approach': '',
            'phases': [],
            'milestones': [],
            'deliverables': [],
            'success_criteria': [],
            'risk_mitigation': [],
            'communication_plan': []
        }
        
        # Determine approach based on complexity
        if task_analysis['complexity'] == 'low':
            plan['approach'] = 'Rapid implementation with minimal overhead'
            plan['phases'] = [
                {'id': 1, 'name': 'Quick Implementation', 'duration': '2 hours', 'owner': chair.id},
                {'id': 2, 'name': 'Testing & Validation', 'duration': '1 hour', 'owner': 'ops-coo'},
                {'id': 3, 'name': 'Deployment', 'duration': '1 hour', 'owner': 'ops-coo'}
            ]
        elif task_analysis['complexity'] == 'medium':
            plan['approach'] = 'Structured approach with proper planning and testing'
            plan['phases'] = [
                {'id': 1, 'name': 'Planning & Design', 'duration': '1 day', 'owner': chair.id},
                {'id': 2, 'name': 'Development', 'duration': '2-3 days', 'owner': 'tech-cto'},
                {'id': 3, 'name': 'Testing & QA', 'duration': '1 day', 'owner': 'ops-coo'},
                {'id': 4, 'name': 'Deployment & Monitoring', 'duration': '1 day', 'owner': 'ops-coo'}
            ]
        else:  # high complexity
            plan['approach'] = 'Enterprise approach with comprehensive planning and risk management'
            plan['phases'] = [
                {'id': 1, 'name': 'Discovery & Research', 'duration': '3 days', 'owner': chair.id},
                {'id': 2, 'name': 'Architecture & Design', 'duration': '3 days', 'owner': 'tech-cto'},
                {'id': 3, 'name': 'Prototype Development', 'duration': '5 days', 'owner': 'tech-cto'},
                {'id': 4, 'name': 'Full Implementation', 'duration': '7 days', 'owner': 'tech-cto'},
                {'id': 5, 'name': 'Testing & Security Audit', 'duration': '3 days', 'owner': 'security-cso'},
                {'id': 6, 'name': 'Deployment & Training', 'duration': '2 days', 'owner': 'ops-coo'},
                {'id': 7, 'name': 'Monitoring & Optimization', 'duration': 'Ongoing', 'owner': 'ops-coo'}
            ]
        
        # Define deliverables
        plan['deliverables'] = [f"Completion of {phase['name']}" for phase in plan['phases']]
        
        # Success criteria
        if 'security' in task_analysis['primary_domain']:
            plan['success_criteria'].extend([
                'Pass security audit',
                'Zero critical vulnerabilities',
                'Compliance certification'
            ])
        
        plan['success_criteria'].extend([
            'Meet all functional requirements',
            'Performance benchmarks achieved',
            'User acceptance obtained',
            'Documentation complete'
        ])
        
        # Risk mitigation
        plan['risk_mitigation'] = [
            {'risk': 'Timeline delay', 'mitigation': 'Buffer time in each phase'},
            {'risk': 'Resource unavailability', 'mitigation': 'Identify backup resources'},
            {'risk': 'Scope creep', 'mitigation': 'Strict change control process'},
            {'risk': 'Technical issues', 'mitigation': 'Prototype and test early'}
        ]
        
        # Communication plan
        plan['communication_plan'] = [
            'Daily standup meetings',
            'Weekly progress reports to stakeholders',
            'Immediate escalation for blockers',
            'Final presentation upon completion'
        ]
        
        return plan
    
    async def generate_department_input(self, member: BoardMember, task_analysis: Dict, initial_plan: Dict) -> Dict:
        """Generate detailed department-specific input"""
        dept_input = {
            'member_id': member.id,
            'member_name': member.name,
            'department': member.department,
            'role': member.role,
            'assessment': '',
            'concerns': [],
            'recommendations': [],
            'resource_requirements': [],
            'estimated_effort': '',
            'dependencies': [],
            'success_metrics': [],
            'action_items': []
        }
        
        # Role-specific detailed input
        if member.role == 'CTO':
            dept_input['assessment'] = f"Technical feasibility: HIGH. The {task_analysis['topic']} aligns with our architecture."
            dept_input['concerns'] = [
                'Integration complexity with existing systems',
                'Technical debt accumulation',
                'Scalability requirements'
            ]
            dept_input['recommendations'] = [
                'Use microservices architecture',
                'Implement comprehensive testing',
                'Follow SOLID principles',
                'Set up CI/CD pipeline'
            ]
            dept_input['resource_requirements'] = [
                '2 senior developers',
                '1 DevOps engineer',
                'Development and staging environments'
            ]
            dept_input['action_items'] = [
                'Create technical specification',
                'Set up development environment',
                'Define API contracts',
                'Establish code review process'
            ]
            
        elif member.role == 'CFO':
            budget = resources.get('estimated_budget', '$5000')
            dept_input['assessment'] = f"Financial impact: {budget}. ROI expected within 6 months."
            dept_input['concerns'] = [
                'Budget allocation',
                'Cost overruns',
                'ROI timeline'
            ]
            dept_input['recommendations'] = [
                'Phase budget release',
                'Track expenses weekly',
                'Identify cost savings',
                'Plan for contingency (20%)'
            ]
            dept_input['resource_requirements'] = [
                'Budget approval',
                'Financial tracking tools',
                'Cost-benefit analysis'
            ]
            dept_input['action_items'] = [
                'Prepare budget proposal',
                'Set up cost tracking',
                'Define ROI metrics',
                'Schedule financial reviews'
            ]
            
        elif member.role == 'CSO':
            dept_input['assessment'] = f"Security risk: {task_analysis.get('risk_level', 'MEDIUM')}. Requires security review."
            dept_input['concerns'] = [
                'Data protection',
                'Access control',
                'Compliance requirements',
                'Vulnerability exposure'
            ]
            dept_input['recommendations'] = [
                'Implement zero-trust architecture',
                'Regular security audits',
                'Encryption at rest and in transit',
                'Multi-factor authentication'
            ]
            dept_input['resource_requirements'] = [
                'Security scanning tools',
                'Penetration testing',
                'Compliance checklist'
            ]
            dept_input['action_items'] = [
                'Conduct threat modeling',
                'Define security requirements',
                'Set up security monitoring',
                'Plan incident response'
            ]
            
        elif member.role == 'COO':
            dept_input['assessment'] = "Operational readiness: GOOD. Infrastructure can support deployment."
            dept_input['concerns'] = [
                'System availability',
                'Performance impact',
                'Maintenance windows',
                'Team capacity'
            ]
            dept_input['recommendations'] = [
                'Blue-green deployment',
                'Comprehensive monitoring',
                'Automated rollback',
                'Load testing'
            ]
            dept_input['resource_requirements'] = [
                'Production environment',
                'Monitoring tools',
                'On-call rotation'
            ]
            dept_input['action_items'] = [
                'Prepare deployment plan',
                'Set up monitoring',
                'Define SLAs',
                'Create runbooks'
            ]
            
        elif member.role == 'CDO':
            dept_input['assessment'] = "Data requirements: MODERATE. Need proper data architecture."
            dept_input['concerns'] = [
                'Data integrity',
                'Storage capacity',
                'Query performance',
                'Data governance'
            ]
            dept_input['recommendations'] = [
                'Design proper schema',
                'Implement data validation',
                'Set up backups',
                'Data lifecycle management'
            ]
            dept_input['resource_requirements'] = [
                'Database infrastructure',
                'ETL tools',
                'Analytics platform'
            ]
            dept_input['action_items'] = [
                'Design data model',
                'Set up database',
                'Create data pipelines',
                'Implement analytics'
            ]
            
        elif member.role == 'CCO':
            dept_input['assessment'] = "User experience impact: HIGH. Requires thoughtful design."
            dept_input['concerns'] = [
                'User adoption',
                'Learning curve',
                'Visual consistency',
                'Accessibility'
            ]
            dept_input['recommendations'] = [
                'User research',
                'Iterative design',
                'A/B testing',
                'Dark theme by default'
            ]
            dept_input['resource_requirements'] = [
                'Design tools',
                'User testing',
                'Style guide'
            ]
            dept_input['action_items'] = [
                'Create wireframes',
                'Design mockups',
                'Conduct user testing',
                'Implement feedback'
            ]
            
        elif member.role == 'CPO':
            dept_input['assessment'] = "Product alignment: STRONG. Fits product roadmap."
            dept_input['concerns'] = [
                'Feature scope',
                'User value',
                'Market timing',
                'Competition'
            ]
            dept_input['recommendations'] = [
                'MVP approach',
                'User feedback loops',
                'Feature flags',
                'Analytics tracking'
            ]
            dept_input['resource_requirements'] = [
                'Product analytics',
                'User feedback tools',
                'Feature management'
            ]
            dept_input['action_items'] = [
                'Define user stories',
                'Prioritize features',
                'Set success metrics',
                'Plan releases'
            ]
        
        # Common elements
        dept_input['estimated_effort'] = resources.get('estimated_time', '1 week')
        dept_input['dependencies'] = [f"Coordination with {m}" for m in resources.get('required_members', [])]
        dept_input['success_metrics'] = [
            f"{member.department} KPIs met",
            'No critical issues',
            'Stakeholder satisfaction'
        ]
        
        return dept_input
    
    def generate_decision_points(self, task_analysis: Dict, initial_plan: Dict, 
                                 department_inputs: List[Dict], resources: Dict) -> List[Decision]:
        """Generate interactive decision points for user"""
        decisions = []
        
        # Decision 1: Approve overall approach
        decisions.append(Decision(
            id=str(uuid.uuid4()),
            type='strategic',
            question='Do you approve the strategic approach?',
            options=[
                {'value': 'approve', 'label': 'Approve - Proceed as planned', 'style': 'success', 'icon': '✅'},
                {'value': 'modify', 'label': 'Modify - Request changes', 'style': 'warning', 'icon': '✏️'},
                {'value': 'escalate', 'label': 'Escalate - Need executive review', 'style': 'danger', 'icon': '⚠️'},
                {'value': 'defer', 'label': 'Defer - Not ready to proceed', 'style': 'secondary', 'icon': '⏸️'}
            ],
            context={
                'plan_summary': initial_plan['executive_summary'],
                'approach': initial_plan['approach'],
                'timeline': resources['estimated_time'],
                'budget': resources['estimated_budget']
            },
            required=True
        ))
        
        # Decision 2: Budget approval (if CFO raised concerns)
        cfo_input = next((d for d in department_inputs if d['role'] == 'CFO'), None)
        if cfo_input and cfo_input['concerns']:
            decisions.append(Decision(
                id=str(uuid.uuid4()),
                type='financial',
                question='Approve budget allocation?',
                options=[
                    {'value': 'full', 'label': f"Full budget: {resources['estimated_budget']}", 'style': 'success', 'icon': '💰'},
                    {'value': 'phased', 'label': 'Phased release (50% upfront)', 'style': 'warning', 'icon': '📊'},
                    {'value': 'reduced', 'label': 'Reduced scope (75% budget)', 'style': 'info', 'icon': '📉'},
                    {'value': 'review', 'label': 'Request detailed breakdown', 'style': 'secondary', 'icon': '🔍'}
                ],
                context={
                    'budget': resources['estimated_budget'],
                    'cfo_concerns': cfo_input['concerns'],
                    'roi_timeline': '6 months'
                },
                required=True,
                depends_on='strategic'
            ))
        
        # Decision 3: Technical approach (if CTO provided options)
        cto_input = next((d for d in department_inputs if d['role'] == 'CTO'), None)
        if cto_input:
            decisions.append(Decision(
                id=str(uuid.uuid4()),
                type='technical',
                question='Select technical implementation approach',
                options=[
                    {'value': 'microservices', 'label': 'Microservices (scalable, complex)', 'style': 'primary', 'icon': '🔧'},
                    {'value': 'monolithic', 'label': 'Monolithic (simpler, faster)', 'style': 'info', 'icon': '📦'},
                    {'value': 'serverless', 'label': 'Serverless (cost-effective)', 'style': 'success', 'icon': '☁️'},
                    {'value': 'hybrid', 'label': 'Hybrid approach', 'style': 'warning', 'icon': '🔀'}
                ],
                context={
                    'cto_recommendation': cto_input['recommendations'][0] if cto_input['recommendations'] else '',
                    'technical_concerns': cto_input['concerns']
                },
                required=False
            ))
        
        # Decision 4: Risk acceptance
        if task_analysis['risk_level'] == 'high':
            decisions.append(Decision(
                id=str(uuid.uuid4()),
                type='risk',
                question='Accept identified risks?',
                options=[
                    {'value': 'accept', 'label': 'Accept risks and proceed', 'style': 'warning', 'icon': '⚠️'},
                    {'value': 'mitigate', 'label': 'Implement additional safeguards', 'style': 'success', 'icon': '🛡️'},
                    {'value': 'transfer', 'label': 'Transfer risk (insurance/outsource)', 'style': 'info', 'icon': '🔄'},
                    {'value': 'avoid', 'label': 'Modify plan to avoid risks', 'style': 'primary', 'icon': '🚫'}
                ],
                context={
                    'risks': resources['risks'],
                    'mitigation_plan': initial_plan['risk_mitigation']
                },
                required=True
            ))
        
        # Decision 5: Timeline and priority
        decisions.append(Decision(
            id=str(uuid.uuid4()),
            type='priority',
            question='Set execution priority and timeline',
            options=[
                {'value': 'urgent', 'label': f"Urgent - Complete in {initial_plan['phases'][0]['duration']}", 
                 'style': 'danger', 'icon': '🚨'},
                {'value': 'high', 'label': f"High - Follow standard timeline ({resources['estimated_time']})", 
                 'style': 'warning', 'icon': '⏰'},
                {'value': 'normal', 'label': 'Normal - Flexible timeline', 'style': 'primary', 'icon': '📅'},
                {'value': 'low', 'label': 'Low - As resources permit', 'style': 'secondary', 'icon': '🎯'}
            ],
            context={
                'recommended_timeline': resources['estimated_time'],
                'phases': len(initial_plan['phases']),
                'dependencies': resources['dependencies']
            },
            required=True
        ))
        
        # Decision 6: Team composition
        if len(resources['required_members']) > 3:
            decisions.append(Decision(
                id=str(uuid.uuid4()),
                type='team',
                question='Approve team composition?',
                options=[
                    {'value': 'full', 'label': 'Full team as recommended', 'style': 'success', 'icon': '👥'},
                    {'value': 'core', 'label': 'Core team only', 'style': 'warning', 'icon': '👤'},
                    {'value': 'augment', 'label': 'Add external resources', 'style': 'info', 'icon': '➕'},
                    {'value': 'custom', 'label': 'Custom team selection', 'style': 'primary', 'icon': '⚙️'}
                ],
                context={
                    'required_members': resources['required_members'],
                    'available_members': list(self.board_members.keys())
                },
                required=False
            ))
        
        return decisions
    
    async def broadcast(self, message: Dict, session_id: str = None):
        """Broadcast message to WebSocket clients"""
        message_str = json.dumps(message)
        disconnected = set()
        
        for ws in self.websocket_clients:
            try:
                await ws.send_text(message_str)
            except:
                disconnected.add(ws)
        
        self.websocket_clients -= disconnected
    
    async def start_conversation(self, topic: str, context: Dict = None) -> str:
        """Start an interactive board conversation"""
        conversation_id = str(uuid.uuid4())
        
        # Phase 1: Task Analysis
        await self.broadcast({
            'type': 'phase_change',
            'phase': 'analysis',
            'message': 'Analyzing task requirements...'
        })
        
        task_analysis = await self.analyze_task(topic)
        
        # Phase 2: Chair Selection
        chair = self.select_chair(task_analysis)
        chair.status = 'leading'
        
        await self.broadcast({
            'type': 'chair_selected',
            'chair': asdict(chair),
            'reason': f"Selected for expertise in {task_analysis['primary_domain']}"
        })
        
        # Phase 3: Resource Identification
        await self.broadcast({
            'type': 'phase_change',
            'phase': 'resource_identification',
            'message': 'Identifying required resources...'
        })
        
        resources = await self.identify_resources(task_analysis)
        
        await self.broadcast({
            'type': 'resources_identified',
            'resources': resources
        })
        
        # Phase 4: Initial Planning
        await self.broadcast({
            'type': 'phase_change',
            'phase': 'planning',
            'message': f"{chair.name} is developing initial plan..."
        })
        
        initial_plan = await self.generate_initial_plan(chair, task_analysis, resources)
        
        await self.broadcast({
            'type': 'initial_plan',
            'plan': initial_plan,
            'presenter': asdict(chair)
        })
        
        # Phase 5: Department Inputs
        await self.broadcast({
            'type': 'phase_change',
            'phase': 'department_input',
            'message': 'Gathering department perspectives...'
        })
        
        department_inputs = []
        for member_id, member in self.board_members.items():
            if member_id != chair.id and member_id in resources['required_members']:
                member.status = 'analyzing'
                await self.broadcast({
                    'type': 'member_status',
                    'member': asdict(member),
                    'action': 'analyzing'
                })
                
                dept_input = await self.generate_department_input(member, task_analysis, initial_plan)
                department_inputs.append(dept_input)
                
                await self.broadcast({
                    'type': 'department_input',
                    'input': dept_input
                })
                
                member.status = 'listening'
                await asyncio.sleep(1)
        
        # Phase 6: Generate Decision Points
        await self.broadcast({
            'type': 'phase_change',
            'phase': 'decision',
            'message': 'Preparing decision points for user input...'
        })
        
        decisions = self.generate_decision_points(task_analysis, initial_plan, department_inputs, resources)
        
        # Create conversation state
        state = ConversationState(
            id=conversation_id,
            topic=topic,
            phase='decision',
            task_analysis=task_analysis,
            chair=asdict(chair),
            resources=resources,
            initial_plan=initial_plan,
            department_inputs=department_inputs,
            pending_decisions=decisions
        )
        
        self.conversation_states[conversation_id] = state
        
        # Send decision points to user
        await self.broadcast({
            'type': 'decisions_required',
            'conversation_id': conversation_id,
            'decisions': [asdict(d) for d in decisions],
            'summary': {
                'topic': topic,
                'complexity': task_analysis['complexity'],
                'timeline': resources['estimated_time'],
                'budget': resources['estimated_budget'],
                'team_size': len(resources['required_members'])
            }
        })
        
        return conversation_id
    
    async def process_user_decision(self, conversation_id: str, decision_id: str, value: str) -> Dict:
        """Process user's decision and update plan accordingly"""
        if conversation_id not in self.conversation_states:
            return {'error': 'Invalid conversation ID'}
        
        state = self.conversation_states[conversation_id]
        state.user_decisions[decision_id] = value
        
        # Find the decision
        decision = next((d for d in state.pending_decisions if d.id == decision_id), None)
        if not decision:
            return {'error': 'Invalid decision ID'}
        
        # Process based on decision type
        response = {'status': 'processed', 'next_steps': []}
        
        if decision.type == 'strategic' and value == 'approve':
            response['next_steps'].append('Proceeding with implementation')
            state.phase = 'execution'
            
            # Start execution
            await self.start_execution(conversation_id)
            
        elif decision.type == 'financial':
            if value == 'full':
                response['next_steps'].append('Full budget allocated')
            elif value == 'phased':
                response['next_steps'].append('Budget will be released in phases')
            elif value == 'reduced':
                response['next_steps'].append('Adjusting scope for reduced budget')
        
        elif decision.type == 'priority':
            state.execution_status['priority'] = value
            response['next_steps'].append(f'Priority set to {value}')
        
        # Broadcast decision result
        await self.broadcast({
            'type': 'decision_made',
            'decision_id': decision_id,
            'decision_type': decision.type,
            'value': value,
            'impact': response['next_steps']
        })
        
        return response
    
    async def start_execution(self, conversation_id: str):
        """Start executing the approved plan"""
        state = self.conversation_states[conversation_id]
        
        await self.broadcast({
            'type': 'phase_change',
            'phase': 'execution',
            'message': 'Starting execution of approved plan...'
        })
        
        # Simulate execution of each phase
        for phase in state.initial_plan['phases']:
            owner = self.board_members.get(phase['owner'])
            if owner:
                owner.status = 'executing'
                owner.active_tasks += 1
                
                await self.broadcast({
                    'type': 'execution_update',
                    'phase': phase['name'],
                    'owner': asdict(owner),
                    'status': 'started',
                    'duration': phase['duration']
                })
                
                # Simulate work being done
                await asyncio.sleep(2)
                
                await self.broadcast({
                    'type': 'execution_update',
                    'phase': phase['name'],
                    'status': 'completed',
                    'message': f"{owner.name} completed {phase['name']}"
                })
                
                owner.status = 'active'
                owner.active_tasks -= 1
        
        # Final summary
        await self.broadcast({
            'type': 'execution_complete',
            'conversation_id': conversation_id,
            'summary': 'All phases completed successfully',
            'next_steps': ['Monitor results', 'Gather feedback', 'Document learnings']
        })

# Initialize the enhanced board
vision_board = EnhancedVisionBoard()

@app.get("/")
async def root():
    return {
        "name": "Enhanced Vision Board API",
        "version": "3.0.0",
        "status": "running",
        "features": [
            "Interactive decision points",
            "Role-based planning",
            "Real-time execution tracking",
            "Comprehensive task analysis"
        ]
    }

@app.get("/api/board/members")
async def get_board_members():
    """Get all board members"""
    return {
        "members": [asdict(m) for m in vision_board.board_members.values()]
    }

@app.post("/api/board/conversation")
async def start_conversation(data: Dict[str, Any]):
    """Start a new board conversation"""
    topic = data.get("topic", "General Discussion")
    context = data.get("context", {})
    
    conversation_id = await vision_board.start_conversation(topic, context)
    
    return {
        "success": True,
        "conversation_id": conversation_id,
        "message": f"Interactive board session started for: {topic}"
    }

@app.post("/api/board/decision")
async def make_decision(data: Dict[str, Any]):
    """Process user decision"""
    conversation_id = data.get("conversation_id")
    decision_id = data.get("decision_id")
    value = data.get("value")
    
    result = await vision_board.process_user_decision(conversation_id, decision_id, value)
    
    return {
        "success": True,
        "result": result
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket connection for real-time updates"""
    await websocket.accept()
    vision_board.websocket_clients.add(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages if needed
    except WebSocketDisconnect:
        vision_board.websocket_clients.discard(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8326)