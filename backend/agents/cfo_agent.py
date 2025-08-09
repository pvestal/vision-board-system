#!/usr/bin/env python3
"""
CFO Agent - Financial analysis, budgeting, and ROI calculations
Performs real financial work and generates actionable reports
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from decimal import Decimal, ROUND_HALF_UP

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

logger = logging.getLogger(__name__)

class CFOAgent:
    """Chief Financial Officer - Handles all financial decisions and analysis"""
    
    def __init__(self):
        self.id = "finance-cfo"
        self.name = "Sarah Finance" 
        self.role = "CFO"
        self.department = "Finance"
        self.status = "active"
        self.current_tasks = []
        self.trust_level = 0.85
        
        # Financial parameters
        self.company_params = {
            'annual_revenue': 2_500_000,  # $2.5M
            'annual_expenses': 2_100_000,  # $2.1M
            'cash_reserves': 500_000,     # $500K
            'employee_cost_per_hour': 75,
            'overhead_rate': 0.3,         # 30% overhead
            'profit_margin_target': 0.15, # 15% profit margin
            'discount_rate': 0.08,        # 8% discount rate for NPV
            'tax_rate': 0.25              # 25% corporate tax rate
        }
        
        logger.info("CFO Agent initialized with financial modeling capabilities")
    
    async def analyze_budget_request(self, request: str, amount: float = None, timeline: str = None) -> Dict:
        """Analyze a budget request and provide financial assessment"""
        
        analysis = {
            'request': request,
            'estimated_cost': 0,
            'cost_breakdown': {},
            'roi_analysis': {},
            'risk_assessment': {},
            'recommendation': '',
            'approval_status': 'pending',
            'conditions': []
        }
        
        # Estimate cost if not provided
        if amount is None:
            analysis['estimated_cost'] = await self._estimate_project_cost(request, timeline)
        else:
            analysis['estimated_cost'] = amount
        
        # Cost breakdown
        analysis['cost_breakdown'] = self._generate_cost_breakdown(request, analysis['estimated_cost'])
        
        # ROI Analysis
        analysis['roi_analysis'] = await self._calculate_roi(request, analysis['estimated_cost'], timeline)
        
        # Risk assessment
        analysis['risk_assessment'] = self._assess_financial_risk(request, analysis['estimated_cost'])
        
        # Generate recommendation
        analysis['recommendation'] = self._generate_recommendation(analysis)
        
        # Approval decision
        analysis['approval_status'] = self._make_approval_decision(analysis)
        
        return analysis
    
    async def _estimate_project_cost(self, request: str, timeline: str = None) -> float:
        """Estimate project cost based on description and timeline"""
        
        request_lower = request.lower()
        base_cost = 5000  # Base cost for small projects
        
        # Scale by complexity keywords
        complexity_multipliers = {
            'simple': 0.5,
            'basic': 0.7,
            'standard': 1.0,
            'complex': 2.0,
            'enterprise': 3.5,
            'large-scale': 4.0,
            'critical': 2.5
        }
        
        for keyword, multiplier in complexity_multipliers.items():
            if keyword in request_lower:
                base_cost *= multiplier
                break
        
        # Additional costs for specific features
        if 'security' in request_lower or 'auth' in request_lower:
            base_cost *= 1.5
        if 'database' in request_lower:
            base_cost *= 1.3
        if 'api' in request_lower:
            base_cost *= 1.2
        if 'frontend' in request_lower or 'ui' in request_lower:
            base_cost *= 1.4
        if 'mobile' in request_lower:
            base_cost *= 2.0
        if 'integration' in request_lower:
            base_cost *= 1.6
        
        # Timeline adjustments
        if timeline:
            timeline_lower = timeline.lower()
            if 'urgent' in timeline_lower or 'asap' in timeline_lower:
                base_cost *= 1.5  # Rush charges
            elif 'flexible' in timeline_lower:
                base_cost *= 0.9  # Discount for flexible timeline
        
        return round(base_cost, 2)
    
    def _generate_cost_breakdown(self, request: str, total_cost: float) -> Dict:
        """Break down estimated cost into categories"""
        
        # Standard cost distribution
        breakdown = {
            'development_labor': total_cost * 0.60,    # 60% labor
            'infrastructure': total_cost * 0.15,       # 15% infrastructure
            'tools_licenses': total_cost * 0.10,       # 10% tools/licenses
            'testing_qa': total_cost * 0.08,           # 8% testing
            'project_management': total_cost * 0.05,   # 5% PM
            'contingency': total_cost * 0.02           # 2% contingency
        }
        
        # Adjust based on project type
        request_lower = request.lower()
        if 'infrastructure' in request_lower or 'deploy' in request_lower:
            breakdown['infrastructure'] *= 2
            breakdown['development_labor'] *= 0.8
        
        if 'security' in request_lower:
            breakdown['testing_qa'] *= 2
            breakdown['tools_licenses'] *= 1.5
        
        # Round to 2 decimal places
        for key in breakdown:
            breakdown[key] = round(breakdown[key], 2)
        
        return breakdown
    
    async def _calculate_roi(self, request: str, cost: float, timeline: str = None) -> Dict:
        """Calculate ROI projections for the investment"""
        
        # Estimate benefits based on project type
        annual_benefit = self._estimate_annual_benefit(request, cost)
        
        # Calculate payback period
        payback_months = (cost / annual_benefit * 12) if annual_benefit > 0 else float('inf')
        
        # Calculate NPV over 3 years
        years = 3
        cash_flows = [-cost]  # Initial investment
        for year in range(1, years + 1):
            # Assume benefits grow slightly each year
            yearly_benefit = annual_benefit * (1.05 ** year)
            cash_flows.append(yearly_benefit)
        
        npv = sum([cf / ((1 + self.company_params['discount_rate']) ** i) 
                   for i, cf in enumerate(cash_flows)])
        
        # ROI percentage
        roi_percentage = ((annual_benefit * years - cost) / cost * 100) if cost > 0 else 0
        
        return {
            'estimated_annual_benefit': round(annual_benefit, 2),
            'payback_period_months': round(payback_months, 1) if payback_months != float('inf') else 'Never',
            'three_year_npv': round(npv, 2),
            'three_year_roi_percentage': round(roi_percentage, 1),
            'break_even_timeline': f"{round(payback_months, 1)} months" if payback_months != float('inf') else 'Does not break even',
            'cash_flows': [round(cf, 2) for cf in cash_flows]
        }
    
    def _estimate_annual_benefit(self, request: str, cost: float) -> float:
        """Estimate annual financial benefit from the project"""
        
        request_lower = request.lower()
        base_benefit = cost * 0.3  # Default 30% annual return
        
        # Project-specific benefit multipliers
        if 'automation' in request_lower:
            base_benefit = cost * 0.8  # High ROI for automation
        elif 'efficiency' in request_lower or 'optimize' in request_lower:
            base_benefit = cost * 0.6  # Good ROI for optimization
        elif 'security' in request_lower:
            base_benefit = cost * 0.2  # Lower direct ROI, but risk mitigation
        elif 'customer' in request_lower or 'user' in request_lower:
            base_benefit = cost * 0.5  # Customer-facing features
        elif 'internal' in request_lower or 'tool' in request_lower:
            base_benefit = cost * 0.4  # Internal tools
        elif 'compliance' in request_lower:
            base_benefit = cost * 0.1  # Compliance has low direct ROI but prevents penalties
        
        return base_benefit
    
    def _assess_financial_risk(self, request: str, cost: float) -> Dict:
        """Assess financial risks of the project"""
        
        risk_assessment = {
            'overall_risk': 'medium',
            'risk_factors': [],
            'mitigation_strategies': [],
            'budget_buffer_recommended': 0.1,  # 10% buffer
            'probability_of_overrun': 0.3      # 30% chance of overrun
        }
        
        request_lower = request.lower()
        
        # Assess risk factors
        if cost > 50000:
            risk_assessment['risk_factors'].append('High-value project increases financial exposure')
            risk_assessment['overall_risk'] = 'high'
            risk_assessment['budget_buffer_recommended'] = 0.15
        
        if 'new technology' in request_lower or 'experimental' in request_lower:
            risk_assessment['risk_factors'].append('Unproven technology increases implementation risk')
            risk_assessment['probability_of_overrun'] = 0.5
        
        if 'integration' in request_lower:
            risk_assessment['risk_factors'].append('System integration often takes longer than expected')
            risk_assessment['budget_buffer_recommended'] += 0.05
        
        if 'urgent' in request_lower:
            risk_assessment['risk_factors'].append('Rush timeline increases costs and error probability')
            risk_assessment['probability_of_overrun'] = 0.4
        
        # Mitigation strategies
        risk_assessment['mitigation_strategies'] = [
            'Phase implementation to reduce upfront risk',
            'Set up regular budget reviews and milestone checks',
            'Maintain contingency fund for unexpected costs',
            'Consider MVP approach to validate before full investment'
        ]
        
        if cost > self.company_params['cash_reserves'] * 0.1:  # More than 10% of cash reserves
            risk_assessment['risk_factors'].append('Significant portion of cash reserves required')
            risk_assessment['mitigation_strategies'].append('Consider phased funding or external financing')
        
        return risk_assessment
    
    def _generate_recommendation(self, analysis: Dict) -> str:
        """Generate financial recommendation based on analysis"""
        
        cost = analysis['estimated_cost']
        roi_data = analysis['roi_analysis']
        risk_data = analysis['risk_assessment']
        
        # Decision criteria
        if roi_data['three_year_roi_percentage'] > 50:
            recommendation = f"STRONG APPROVE: Excellent ROI of {roi_data['three_year_roi_percentage']}% over 3 years."
        elif roi_data['three_year_roi_percentage'] > 20:
            recommendation = f"APPROVE: Good ROI of {roi_data['three_year_roi_percentage']}% over 3 years."
        elif roi_data['three_year_roi_percentage'] > 0:
            recommendation = f"CONDITIONAL APPROVE: Modest ROI of {roi_data['three_year_roi_percentage']}%. Consider if strategic value justifies investment."
        else:
            recommendation = f"NOT RECOMMENDED: Negative ROI of {roi_data['three_year_roi_percentage']}%. Project does not meet financial criteria."
        
        # Add risk considerations
        if risk_data['overall_risk'] == 'high':
            recommendation += f" HIGH RISK PROJECT - recommend reducing scope or phasing implementation."
        
        # Add budget context
        cash_percentage = (cost / self.company_params['cash_reserves']) * 100
        if cash_percentage > 20:
            recommendation += f" SIGNIFICANT INVESTMENT - requires {cash_percentage:.1f}% of cash reserves."
        
        return recommendation
    
    def _make_approval_decision(self, analysis: Dict) -> str:
        """Make binary approval decision based on analysis"""
        
        roi_percentage = analysis['roi_analysis']['three_year_roi_percentage']
        cost = analysis['estimated_cost']
        risk_level = analysis['risk_assessment']['overall_risk']
        
        # Approval criteria
        if roi_percentage >= 30 and cost <= self.company_params['cash_reserves'] * 0.15:
            return 'approved'
        elif roi_percentage >= 15 and risk_level != 'high' and cost <= self.company_params['cash_reserves'] * 0.1:
            return 'conditional_approval'
        else:
            return 'denied'
    
    async def generate_budget_report(self, project: str = None) -> Dict:
        """Generate comprehensive budget report"""
        
        report = {
            'report_date': datetime.now().isoformat(),
            'reporting_period': 'Current Quarter',
            'company_financial_health': {},
            'budget_summary': {},
            'recommendations': [],
            'cash_flow_projection': {},
            'project_specific': {}
        }
        
        # Company financial health
        report['company_financial_health'] = {
            'annual_revenue': self.company_params['annual_revenue'],
            'annual_expenses': self.company_params['annual_expenses'],
            'net_profit': self.company_params['annual_revenue'] - self.company_params['annual_expenses'],
            'profit_margin': ((self.company_params['annual_revenue'] - self.company_params['annual_expenses']) / 
                            self.company_params['annual_revenue']) * 100,
            'cash_reserves': self.company_params['cash_reserves'],
            'months_of_runway': self.company_params['cash_reserves'] / (self.company_params['annual_expenses'] / 12),
            'financial_rating': 'Healthy' if self.company_params['cash_reserves'] > 300000 else 'Cautious'
        }
        
        # Budget allocation recommendations
        max_project_budget = self.company_params['cash_reserves'] * 0.2  # 20% of cash reserves
        recommended_monthly_spend = self.company_params['annual_revenue'] * 0.05 / 12  # 5% of revenue monthly
        
        report['budget_summary'] = {
            'max_single_project_budget': max_project_budget,
            'recommended_monthly_development_spend': recommended_monthly_spend,
            'available_for_new_projects': max_project_budget * 0.8,  # Reserve 20% buffer
            'emergency_reserve_minimum': self.company_params['cash_reserves'] * 0.3
        }
        
        # Cash flow projection (next 6 months)
        monthly_revenue = self.company_params['annual_revenue'] / 12
        monthly_expenses = self.company_params['annual_expenses'] / 12
        current_cash = self.company_params['cash_reserves']
        
        projection = []
        for month in range(1, 7):
            current_cash += monthly_revenue - monthly_expenses
            projection.append({
                'month': month,
                'projected_cash': round(current_cash, 2),
                'revenue': monthly_revenue,
                'expenses': monthly_expenses,
                'net_change': monthly_revenue - monthly_expenses
            })
        
        report['cash_flow_projection'] = projection
        
        # General recommendations
        report['recommendations'] = [
            'Maintain 6-month expense runway in cash reserves',
            'Invest in high-ROI automation projects to reduce operational costs',
            'Consider revenue diversification to reduce dependency risks',
            'Phase large projects to distribute financial risk over time',
            'Set up monthly budget reviews with department heads'
        ]
        
        # Project-specific analysis if provided
        if project:
            project_analysis = await self.analyze_budget_request(project)
            report['project_specific'] = project_analysis
        
        return report
    
    async def create_financial_forecast(self, scenarios: List[str] = None) -> Dict:
        """Create financial forecasts for different scenarios"""
        
        if scenarios is None:
            scenarios = ['conservative', 'realistic', 'optimistic']
        
        forecast = {
            'forecast_date': datetime.now().isoformat(),
            'scenarios': {},
            'assumptions': {
                'conservative': {'revenue_growth': 0.05, 'expense_growth': 0.08},
                'realistic': {'revenue_growth': 0.12, 'expense_growth': 0.06},
                'optimistic': {'revenue_growth': 0.25, 'expense_growth': 0.04}
            }
        }
        
        base_revenue = self.company_params['annual_revenue']
        base_expenses = self.company_params['annual_expenses']
        
        for scenario in scenarios:
            assumptions = forecast['assumptions'][scenario]
            
            yearly_projections = []
            current_revenue = base_revenue
            current_expenses = base_expenses
            
            for year in range(1, 4):  # 3-year forecast
                current_revenue *= (1 + assumptions['revenue_growth'])
                current_expenses *= (1 + assumptions['expense_growth'])
                
                yearly_projections.append({
                    'year': year,
                    'revenue': round(current_revenue, 2),
                    'expenses': round(current_expenses, 2),
                    'net_profit': round(current_revenue - current_expenses, 2),
                    'profit_margin': round(((current_revenue - current_expenses) / current_revenue) * 100, 1)
                })
            
            forecast['scenarios'][scenario] = yearly_projections
        
        return forecast
    
    async def execute_task(self, task_description: str, context: Dict = None) -> Dict:
        """Execute a financial task and return results"""
        
        self.status = "executing"
        self.current_tasks.append(task_description)
        
        try:
            task_lower = task_description.lower()
            result = {
                'task': task_description,
                'executed_by': f"{self.name} ({self.role})",
                'execution_time': datetime.now().isoformat(),
                'status': 'completed'
            }
            
            # Budget analysis
            if 'budget' in task_lower or 'cost' in task_lower or 'approve' in task_lower:
                amount = context.get('amount') if context else None
                timeline = context.get('timeline') if context else None
                result['budget_analysis'] = await self.analyze_budget_request(task_description, amount, timeline)
            
            # Financial report
            elif 'report' in task_lower or 'financial' in task_lower:
                result['financial_report'] = await self.generate_budget_report(task_description)
            
            # ROI analysis
            elif 'roi' in task_lower or 'return' in task_lower:
                amount = context.get('amount', 10000) if context else 10000
                result['roi_analysis'] = await self._calculate_roi(task_description, amount)
            
            # Forecast
            elif 'forecast' in task_lower or 'projection' in task_lower:
                result['financial_forecast'] = await self.create_financial_forecast()
            
            # General financial advice
            else:
                result['financial_advice'] = await self._provide_financial_advice(task_description, context)
            
            logger.info(f"CFO executed task: {task_description}")
            return result
            
        except Exception as e:
            logger.error(f"CFO task execution failed: {e}")
            return {
                'task': task_description,
                'status': 'failed',
                'error': str(e),
                'executed_by': f"{self.name} ({self.role})"
            }
        finally:
            self.status = "active"
            if task_description in self.current_tasks:
                self.current_tasks.remove(task_description)
    
    async def _provide_financial_advice(self, request: str, context: Dict = None) -> Dict:
        """Provide general financial advice"""
        
        advice = {
            'recommendation': '',
            'financial_impact': '',
            'considerations': [],
            'action_items': []
        }
        
        request_lower = request.lower()
        
        if 'hire' in request_lower or 'staff' in request_lower:
            advice['recommendation'] = 'Hiring should be based on revenue growth and workload sustainability'
            advice['financial_impact'] = f'Each new hire costs approximately ${self.company_params["employee_cost_per_hour"] * 2080:.0f} annually including overhead'
            advice['considerations'] = [
                'Ensure 6+ months of salary coverage in cash reserves',
                'Consider contractor vs full-time employee costs',
                'Evaluate revenue per employee metrics'
            ]
        elif 'investment' in request_lower:
            advice['recommendation'] = 'Investments should target minimum 20% annual ROI'
            advice['considerations'] = [
                'Diversify investment portfolio',
                'Consider opportunity cost vs other projects',
                'Assess risk tolerance against cash reserves'
            ]
        else:
            advice['recommendation'] = 'Focus on revenue growth while maintaining cost discipline'
            advice['considerations'] = [
                'Monitor cash flow weekly',
                'Maintain minimum 6-month expense runway',
                'Invest in automation to reduce operational costs'
            ]
        
        advice['action_items'] = [
            'Review monthly financial statements',
            'Set up automated expense tracking',
            'Schedule quarterly budget reviews'
        ]
        
        return advice
    
    def get_status(self) -> Dict:
        """Get current status of the CFO agent"""
        return {
            'id': self.id,
            'name': self.name,
            'role': self.role,
            'status': self.status,
            'current_tasks': len(self.current_tasks),
            'trust_level': self.trust_level,
            'capabilities': [
                'Budget analysis and approval',
                'ROI calculations and projections',
                'Financial risk assessment',
                'Cash flow forecasting',
                'Cost-benefit analysis',
                'Financial reporting',
                'Investment evaluation'
            ],
            'company_financial_health': {
                'cash_reserves': self.company_params['cash_reserves'],
                'monthly_runway': round(self.company_params['cash_reserves'] / (self.company_params['annual_expenses'] / 12), 1),
                'profit_margin': round(((self.company_params['annual_revenue'] - self.company_params['annual_expenses']) / 
                                      self.company_params['annual_revenue']) * 100, 1)
            }
        }

# For testing
if __name__ == "__main__":
    async def test_cfo():
        cfo = CFOAgent()
        
        # Test budget analysis
        result = await cfo.execute_task("Approve budget for new user authentication system", 
                                       {"amount": 25000, "timeline": "2 weeks"})
        print("CFO Budget Analysis:", json.dumps(result, indent=2, default=str))
        
        # Test financial report
        report_result = await cfo.execute_task("Generate quarterly financial report")
        print("CFO Report:", json.dumps(report_result, indent=2, default=str))
    
    asyncio.run(test_cfo())