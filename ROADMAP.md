# Genesis Roadmap

## Vision

Genesis aims to be the universal framework for LLM-driven code evolution across any programming language, execution environment, and optimization objective.

---

## Current Language Support

| Language | Local | Slurm | E2B (base) | E2B (custom) | Notes |
|----------|:-----:|:-----:|:----------:|:------------:|-------|
| **Python** | ✅ | ✅ | ✅ | ✅ | First-class support, all features |
| **Rust** | ✅ | ✅ | ❌ | ✅ | Needs `rustc` in environment |
| **C++** | ✅ | ✅ | ❌ | ✅ | Needs `g++` or `clang++` |
| **CUDA** | ⚠️ | ✅ | ❌ | ❌ | Requires GPU + `nvcc` |
| **JavaScript/TypeScript** | ✅ | ✅ | ✅ | ✅ | Node.js available in E2B base |

### Adding New Language Support

To evolve code in any language, you need:

1. **Compiler/Interpreter** in the execution environment
2. **Python evaluation wrapper** that:
   - Compiles the evolved code (if needed)
   - Runs it against test cases
   - Returns a numeric fitness score

See `examples/mask_to_seg_rust/` for a complete Rust example.

---

## Execution Backend Status

| Backend | Status | Parallelism | GPU Support | Best For |
|---------|:------:|:-----------:|:-----------:|----------|
| **Local** | ✅ Done | 1-4 jobs | If available | Development, testing |
| **Slurm (Docker)** | ✅ Done | Unlimited | ✅ Yes | HPC clusters |
| **Slurm (Conda)** | ✅ Done | Unlimited | ✅ Yes | HPC clusters |
| **E2B** | ✅ Done | ~50 jobs | ❌ No | Cloud parallel execution |
| **Modal** | 🔜 Planned | Unlimited | ✅ Yes | Serverless GPU |
| **Ray** | 💭 Idea | Unlimited | ✅ Yes | Distributed clusters |

---

## Planned Improvements

### High Priority

#### E2B Templates for Compiled Languages
- [ ] Pre-built E2B templates with common toolchains:
  - `genesis-rust` - Rust toolchain (rustc, cargo)
  - `genesis-cpp` - C++ toolchain (g++, clang++, cmake)
  - `genesis-go` - Go toolchain
  - `genesis-full` - All languages combined
- [ ] One-command template deployment
- [ ] Documentation for custom template creation

#### Modal Integration for GPU Code
- [ ] CUDA kernel evolution with GPU execution
- [ ] PyTorch/JAX model optimization
- [ ] Serverless GPU with automatic scaling
- [ ] Cost tracking and budget limits

#### Improved Rust Support
- [ ] Cargo project support (not just single-file rustc)
- [ ] Crate dependency management
- [ ] SIMD-aware optimization prompts
- [ ] Benchmark-driven fitness (criterion.rs integration)

### Medium Priority

#### Language-Specific LLM Optimization Hints
- [ ] **Rust**: Ownership/borrowing patterns, SIMD intrinsics, zero-copy
- [ ] **C++**: Template metaprogramming, cache optimization, vectorization
- [ ] **CUDA**: Warp efficiency, shared memory, occupancy optimization
- [ ] **Python**: NumPy vectorization, Cython hints, memory views

#### Multi-Language Project Evolution
- [ ] Python + Rust (PyO3 bindings)
- [ ] Python + C++ (pybind11)
- [ ] Evolve both sides of FFI boundaries
- [ ] Cross-language fitness evaluation

#### Database Modernization (ClickHouse Backend)
- [ ] **Full ClickHouse Migration**: Move the operational backend (program storage, metadata) entirely to ClickHouse.
  - Replace SQLite `programs` table with ClickHouse `MergeTree` engines.
  - Implement real-time inserts/updates using ClickHouse best practices (ReplacingMergeTree or CollapsingMergeTree).
  - Eliminate the need for local SQLite files, enabling stateless execution nodes.
- [ ] **Data Normalization**: Design a normalized ClickHouse schema to replace the current JSON-heavy structure.
  - Extract metrics into dedicated columns/tables for fast analytical queries.
  - Store lineage and relationships efficiently.
- [ ] **Distributed Storage**: Leverage ClickHouse's distributed tables for multi-node scaling.

#### Enhanced Parallelism
- [ ] Adaptive `max_parallel_jobs` based on backend capacity
- [ ] Job priority queuing
- [ ] Preemption for higher-fitness candidates
- [ ] Distributed island model across backends

#### Reasoning & Verification (Inspired by OpenR)
- [ ] **Process Reward Models for Code**
  - Train PRMs to evaluate intermediate code states (not just final fitness)
  - Discriminative models: Score quality of each code modification step
  - Generative models: Predict likelihood of successful evolution path
  - Dataset: Collect step-by-step evolution traces with fitness improvements
  
- [ ] **Self-Improving Task Prompts (DSPy Integration)**
  - **MIPRO-style Instruction Optimization**: Implement an "outer loop" that treats the `task_sys_msg` as a variable.
    - Run short evolution "micro-sprints" (e.g., 5 generations).
    - Use a meta-LLM to propose variations of the task prompt.
    - Select the prompt phrasing that yields the highest average population fitness.
  - **Bootstrap Few-Shot Selection**: Instead of just random "Top-K" inspirations, use an optimizer to select the *most effective* historical mutations to show as few-shot examples.
    - Learn which types of examples (e.g., "small refactor" vs "algorithm swap") lead to better LLM outputs for the specific task.
  - **Signature-based Prompts**: Abstract prompts into Input/Output signatures (e.g., `Code -> ImprovedCode`), allowing the system to automatically format and structure the prompt implementation.

- [ ] **Advanced Search Strategies**
  - [ ] MCTS for Code Evolution: Tree search over mutation space
    - Each node is a code variant
    - Use PRM to guide expansion (UCB-like selection)
    - Rollouts via LLM-predicted fitness estimates
  - [ ] Beam Search Evolution: Keep top-k candidates per generation
  - [ ] Best-of-N with Learned Verifiers: Generate N mutations, select via PRM
  - [ ] rStar-style Mutual Reasoning: Cross-verify mutations via multiple LLMs
  
- [ ] **Pre-Execution Code Verification**
  - [ ] Static analysis-based fitness prediction (before running code)
  - [ ] Syntax/type correctness filters to avoid wasted evaluations
  - [ ] Complexity analysis (estimated runtime before execution)
  - [ ] LLM-based "plausibility scoring" for proposed mutations
  
- [ ] **Test-Time Compute Optimization**
  - [ ] Adaptive search budget: Spend more compute on promising candidates
  - [ ] Early stopping for unpromising evolution branches
  - [ ] Dynamic temperature/sampling based on search progress
  - [ ] Meta-learning: When to use expensive vs cheap LLMs
  
- [ ] **Self-Supervised Learning from Evolution**
  - [ ] Train reward models from evolution history (no human labels)
  - [ ] Learn which mutations tend to improve fitness
  - [ ] Distill successful evolution strategies into smaller models
  - [ ] Fine-tune LLMs on high-fitness code transitions

### Future Exploration

#### Additional Languages
| Language | Use Case | Complexity |
|----------|----------|------------|
| **WebAssembly** | Browser-based evolution, portable binaries | Medium |
| **Go** | Systems programming, microservices | Easy |
| **Julia** | Scientific computing, differentiable programming | Medium |
| **Zig** | Systems programming with safety | Medium |
| **Mojo** | Python syntax + systems performance | Hard (new language) |
| **Haskell** | Functional algorithm optimization | Medium |
| **Assembly (x86/ARM)** | Extreme low-level optimization | Hard |

#### Advanced Features
- [ ] **Auto-vectorization verification** - Ensure SIMD is actually used
- [ ] **Formal verification integration** - Prove evolved code correctness
- [ ] **Energy-aware optimization** - Minimize power consumption
- [ ] **Hardware-specific tuning** - ARM vs x86, specific CPU features

#### MCP Server Expansion
Genesis already has a basic MCP (Model Context Protocol) server ([`genesis/mcp_server.py`](genesis/mcp_server.py)) that exposes evolution capabilities to MCP clients like Claude Desktop, Cursor, and other AI coding assistants.

**Current Capabilities** (✅ Implemented):
- List recent evolution experiments
- Get experiment metrics and status
- Launch new evolution experiments
- Read best discovered code

**Planned Expansions**:
- [ ] **Real-time Experiment Monitoring**
  - Stream generation progress updates
  - Live fitness score graphs
  - Mutation success/failure notifications
  - WebSocket-based real-time updates

- [ ] **Interactive Evolution Control**
  - Pause/resume experiments
  - Adjust parameters mid-evolution (temperature, mutation rate)
  - Manual mutation injection ("try this specific change")
  - Island management (merge/split islands)

- [ ] **Code Analysis Tools**
  - Compare code variants side-by-side
  - Explain fitness differences between variants
  - Generate mutation lineage trees
  - Semantic code search across evolution history

- [ ] **Advanced Experiment Management**
  - Clone and fork experiments
  - A/B test different evolution strategies
  - Batch experiment launching (grid search over configs)
  - Export experiments to reproducible formats

- [ ] **Integration with AI Coding Assistants**
  - Natural language experiment queries ("Show me the fastest circle packing solution")
  - AI-assisted config generation
  - Automatic fitness function creation from specs
  - Code suggestion based on evolution insights

- [ ] **Multi-User Collaboration**
  - Share experiments across team members
  - Collaborative fitness function design
  - Distributed compute pooling
  - Experiment leaderboards

**Use Cases**:
- Use Claude Desktop to manage Genesis experiments without leaving your IDE
- Query evolution history: "What mutations improved performance on circle packing?"
- Launch experiments from natural language: "Evolve a faster HNSW implementation for 1000 generations"
- Get AI insights: "Why did this mutation succeed?" (analyze with Claude + evolution data)

**Example MCP Configuration** (`.mcp.json`):
```json
{
  "mcpServers": {
    "genesis": {
      "type": "stdio",
      "command": "python3",
      "args": ["-m", "genesis.mcp_server"],
      "cwd": "/path/to/Genesis"
    }
  }
}
```

#### Memory & State Management
- [ ] **Letta/MemGPT Integration** 
  - Long-term memory across evolution sessions
  - Remember successful mutation strategies
  - Learn from past experiments
  - Cross-experiment knowledge transfer
  - Hierarchical memory (working + long-term)
  
- [ ] **Vector Database for Code History**
  - Qdrant/Milvus integration for semantic code search
  - Embed all evolved code variants
  - Find similar solutions from past experiments
  - Novelty detection via embedding distance

---

## Completed Milestones

- [x] Core evolution framework with island model
- [x] Local execution backend
- [x] Slurm cluster support (Docker + Conda)
- [x] E2B cloud sandbox integration
- [x] Python language support
- [x] Rust language support (single-file)
- [x] WebUI for experiment monitoring
- [x] Novelty search and diversity maintenance
- [x] Multi-LLM support (OpenAI, Anthropic, Google, DeepSeek)
- [x] Hydra configuration system
- [x] Basic MCP server for AI assistant integration

---

## Contributing

We welcome contributions! Priority areas:

1. **E2B templates** for compiled languages
2. **Modal backend** implementation
3. **Language-specific examples** and evaluators
4. **Documentation** improvements

See [AGENTS.md](AGENTS.md) for contribution guidelines.
