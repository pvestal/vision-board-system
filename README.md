# Vision Board System

An interactive multi-agent orchestration system that simulates a Board of Directors meeting where AI agents collaborate, make decisions, and execute tasks based on their specialized roles.

## Features

- 🤖 **Multi-Agent Orchestration** - Board of Directors with specialized AI agents
- 💬 **Interactive Chat Interface** - Toggleable chat threads for each agent
- 🎯 **Decision Points** - User makes strategic decisions
- 📊 **Task Analysis** - Automatic complexity and domain analysis
- 👔 **Role-Based Execution** - Each agent performs their specialized role
- 🗳️ **Voting Mechanism** (Coming Soon)
- 📚 **Knowledge Base Integration** (Coming Soon)
- 📈 **Progress Tracking** (Coming Soon)

## Quick Start

### Backend API
```bash
cd backend/api
python3 vision_board_interactive.py
# API runs on http://localhost:8302
```

### Frontend Chat Interface
```bash
cd frontend
python3 -m http.server 8327
# Visit http://localhost:8327/chat-interface.html
```

## Board Members

- **CEO** (Alexandra Vision) - Strategic oversight and leadership
- **CTO** (Marcus Tech) - Technical architecture and development
- **CFO** (Sarah Finance) - Budget and financial analysis
- **CSO** (David Security) - Security and compliance
- **COO** (Lisa Operations) - Operations and deployment
- **CDO** (James Data) - Data management and analytics
- **CCO** (Emma Creative) - Design and user experience
- **CPO** (Michael Product) - Product strategy and features

## API Endpoints

- `GET /api/board/members` - Get all board members
- `POST /api/board/conversation` - Start new discussion
- `POST /api/board/decision` - Process user decision
- `WS /ws` - WebSocket for real-time updates

## Architecture

The system uses a FastAPI backend with WebSocket support for real-time communication. The frontend provides both a dashboard view and a chat-style interface with toggleable agent threads.

## Development

See [VISION_BOARD_IMPLEMENTATION_PLAN.md](../VISION_BOARD_IMPLEMENTATION_PLAN.md) for complete implementation details and roadmap.

## License

MIT