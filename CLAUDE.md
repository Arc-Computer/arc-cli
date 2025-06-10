- When using web search, review the latest information from 2025 and ensure you're using the most up to date context
- When writing code, ensure you focus on simplicity and clarity, do not overengineer the implementation - the best code is the most efficient and optimized
- When adding in AI models, ensure you are using the latest models from each model provider, for example - GPT 4.1 from open AI and Claude 4 Sonnet from Anthropic
- You are an expert AI engineer, when writing code, think through the implications. Read all files first and ensure you understand the data flow and system design before writing code.
- Always create and activate a virtual environment before running the code if there is one in the project.
- Always use the github api or cli to create and manage issues, pull requests, and other github related tasks.
- When writing issues, PRs, or other public facing documents, be concise and to the point, write in markdown format and do not use emojis.
- When validating code, remove temporary files and comments before committing.
- The following emoji's are allowed: ✓\|✗\|•\|↑\|↓\|→\  all other emoji's are not allowed.

## The Art of Naming: Your Code's First Impression

Good naming takes time but saves more time in the long run. Poor naming creates a cascading impact on development, from initial coding to long-term maintenance. Well-chosen names:

- **Reduce cognitive load**: Developers spend more time reading code than writing it
- **Prevent bugs**: Clear names prevent misunderstandings and incorrect usage
- **Enable collaboration**: Team members can understand code without extensive documentation
- **Facilitate refactoring**: Good names make it easier to identify what needs changing

### Core Naming Principles

#### 1. **Be Descriptive and Specific**
```python
# ❌ Bad
def calc(x, y):
    return x * 0.1 if x > 100 else 0

# ✅ Good
def calculate_discount(price: float, threshold: float = 100) -> float:
    return price * 0.1 if price > threshold else 0
```

#### 2. **Use Consistent Conventions**
Choose one convention per project and stick to it:

- **Variables & Functions**: `camelCase` or `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Interfaces**: `IPrefixedPascalCase` or `PascalCaseInterface`

#### 3. **Avoid Ambiguity**
```typescript
// ❌ Bad - What kind of data? What format?
let data = getUserInfo();

// ✅ Good - Clear about content and structure
let userProfile = getUserProfile();
```

#### 4. **Length Guidelines**
- Shorter names for limited scope (loop indices), longer names for wider scope
- Aim for 1-4 words that capture the essence
- Avoid excessive abbreviations except for well-known terms (URL, API, etc.)

---

## 2. Code Organization: Creating Navigable Codebases

### Directory Structure Principles

Organize your codebase into directories and files that reflect the modular structure of your application. A well-organized project should tell a story:

```
project-root/
├── src/
│   ├── api/              # External interfaces
│   ├── core/             # Business logic
│   ├── models/           # Data models
│   ├── services/         # External service integrations
│   ├── utils/            # Shared utilities
│   └── config/           # Configuration management
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docs/
│   ├── api/
│   ├── architecture/
│   └── deployment/
└── scripts/              # Build, deploy, maintenance scripts
```

### Layered Architecture

A layered architecture is a common approach where code is organized into distinct layers based on functionality:

1. **Presentation Layer**: UI components, API endpoints
2. **Business Logic Layer**: Core domain logic, use cases
3. **Data Access Layer**: Database queries, external API calls

### File Organization Best Practices

#### 1. **One Concept Per File**
Each file should have a single, clear purpose:

```typescript
// ❌ Bad: user-stuff.ts (mixing concerns)
export class User { }
export class UserValidator { }
export class UserRepository { }

// ✅ Good: Separate files
// user.model.ts
export class User { }

// user.validator.ts
export class UserValidator { }

// user.repository.ts
export class UserRepository { }
```

#### 2. **Consistent File Naming**
- Use descriptive names that indicate purpose
- Include type suffixes: `.model.ts`, `.service.ts`, `.controller.ts`
- Match file names to main export: `UserService` → `user.service.ts`

#### 3. **Logical Grouping**
Group related functionality together:

```
features/
├── authentication/
│   ├── auth.service.ts
│   ├── auth.controller.ts
│   ├── auth.middleware.ts
│   └── auth.types.ts
└── ml-pipeline/
    ├── pipeline.orchestrator.ts
    ├── pipeline.validator.ts
    └── pipeline.types.ts
```

---

## 3. Modular Design: Building with LEGO Blocks

### Core Principles of Modularity

High cohesion within modules and loose coupling between them are the foundation of modular design:

#### 1. **High Cohesion**
Elements within a module should work together toward a single purpose:

```python
# ✅ Good: Cohesive module for model training
class ModelTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.optimizer = self._create_optimizer()
        self.loss_fn = self._create_loss_function()
    
    def train(self, dataset: Dataset) -> Model:
        # All methods work together for training
        pass
    
    def _create_optimizer(self) -> Optimizer:
        pass
    
    def _create_loss_function(self) -> LossFunction:
        pass
```

#### 2. **Loose Coupling**
Modules connect with each other only through their outer layers:

```typescript
// Define clear interfaces for module communication
interface DataProcessor {
    process(data: RawData): ProcessedData;
}

// Implementation details are hidden
class MLDataProcessor implements DataProcessor {
    process(data: RawData): ProcessedData {
        // Complex ML preprocessing hidden behind simple interface
    }
}
```

### Practical Modular Patterns

#### 1. **Plugin Architecture**
Perfect for extensible systems:

```python
# Core system defines plugin interface
class ModelPlugin(ABC):
    @abstractmethod
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def postprocess(self, predictions: np.ndarray) -> Any:
        pass

# Plugins implement interface
class FederatedLearningPlugin(ModelPlugin):
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        return self.apply_differential_privacy(data)
```

#### 2. **Service-Oriented Modules**
Each module provides a specific service:

```typescript
// Blockchain service module
export class BlockchainService {
    constructor(
        private readonly provider: NEARProvider,
        private readonly config: BlockchainConfig
    ) {}
    
    async deployContract(wasm: Buffer): Promise<DeploymentResult> {
        // Encapsulates all blockchain interaction logic
    }
}
```

### Module Boundaries Checklist

- [ ] Each module has a single, clear responsibility
- [ ] Dependencies flow in one direction (no circular dependencies)
- [ ] Modules communicate through well-defined interfaces
- [ ] Internal implementation details are hidden
- [ ] Modules can be tested independently
- [ ] Changes to one module rarely require changes to others

---

## 4. Simplicity in Design: The Power of Less

### KISS (Keep It Simple, Stupid)

The KISS principle emphasizes simplicity in code design and implementation. Simple solutions are:
- Easier to understand and debug
- Less prone to bugs
- More flexible and maintainable

#### Practical KISS Applications

```python
# ❌ Over-engineered
class DataProcessorFactory:
    def create_processor(self, type: str) -> DataProcessor:
        if type == "csv":
            return CSVDataProcessor()
        # ... 20 more conditions
        
# ✅ Simple and direct (when you only need CSV)
def process_csv_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)
```

### YAGNI (You Aren't Gonna Need It)

The YAGNI rule is about not adding stuff to your project just because you think you might need it later:

```typescript
// ❌ Bad: Adding unused future-proofing
interface User {
    id: string;
    name: string;
    email: string;
    // "Maybe we'll need these later"
    socialMediaProfiles?: SocialProfile[];
    biometricData?: BiometricData;
    blockchainWallet?: WalletInfo;
}

// ✅ Good: Only what's needed now
interface User {
    id: string;
    name: string;
    email: string;
}
```

### DRY (Don't Repeat Yourself)

Write the same code only once. Extract common patterns:

```python
# ❌ Repetitive
def process_training_data(data):
    data = remove_nulls(data)
    data = normalize(data)
    data = apply_transforms(data)
    return data

def process_validation_data(data):
    data = remove_nulls(data)
    data = normalize(data)
    data = apply_transforms(data)
    return data

# ✅ DRY
def process_ml_data(data, data_type: str):
    pipeline = [remove_nulls, normalize, apply_transforms]
    for transform in pipeline:
        data = transform(data)
    return data
```

### Simplicity Guidelines

1. **Start with the simplest solution that works**
2. **Refactor when complexity is actually needed**
3. **Favor composition over inheritance**
4. **Use existing libraries rather than reinventing**
5. **Write code for humans first, computers second**

---

## 5. Documentation: Your Future Self Will Thank You

### Documentation Principles

Good documentation practices are essential for creating clear, concise, and effective technical documentation:

1. **Know Your Audience**: Tailor complexity and terminology
2. **Be Task-Oriented**: Help users achieve specific goals
3. **Stay Current**: Update docs with code changes
4. **Use Examples**: Show, don't just tell

### Code Documentation Layers

#### 1. **Inline Comments**
Explain *why*, not *what*:

```python
# ❌ Bad: Restating the obvious
# Increment counter by 1
counter += 1

# ✅ Good: Explaining reasoning
# Batch size must be even for federated averaging to work correctly
if batch_size % 2 != 0:
    batch_size += 1
```

#### 2. **Function/Method Documentation**
Use docstrings that explain purpose, parameters, and return values:

```python
def aggregate_client_updates(
    updates: List[ModelUpdate],
    aggregation_method: str = "fedavg"
) -> ModelUpdate:
    """
    Aggregate model updates from multiple federated learning clients.
    
    This implements several aggregation strategies commonly used in
    federated learning to combine client model updates while preserving
    privacy and handling non-IID data distributions.
    
    Args:
        updates: List of model updates from participating clients
        aggregation_method: Strategy for aggregation ('fedavg', 'median', 'trimmed_mean')
    
    Returns:
        Aggregated model update ready for global model application
    
    Raises:
        ValueError: If aggregation_method is not supported
        
    Example:
        >>> updates = [client1.get_update(), client2.get_update()]
        >>> global_update = aggregate_client_updates(updates, "fedavg")
    """
```

#### 3. **Module-Level Documentation**
Explain the module's purpose and how it fits into the larger system:

```python
"""
Federated Learning Aggregation Module

This module implements various aggregation strategies for federated learning,
including FedAvg, FedProx, and custom weighted averaging schemes. It's designed
to work with heterogeneous client devices and non-IID data distributions.

Key Features:
- Byzantine-robust aggregation
- Differential privacy support
- Adaptive client weighting

Usage:
    from fl_aggregation import FederatedAggregator
    
    aggregator = FederatedAggregator(strategy="fedavg")
    global_model = aggregator.aggregate(client_updates)
"""
```

### API Documentation Best Practices

Clear structure with sections like Overview, Getting Started, API Endpoints, Code Examples, and Error Handling:

```markdown
## API Reference: Model Training Service

### Overview
The Model Training Service provides endpoints for distributed model training
across federated learning nodes.

### Authentication
All endpoints require Bearer token authentication:
```
Authorization: Bearer <your-token>
```

### Endpoints

#### POST /training/start
Initiates a new training session across federated nodes.

**Request Body:**
```json
{
  "model_id": "string",
  "dataset_config": {
    "node_selection": "random|targeted",
    "min_nodes": 10
  }
}
```

**Response:**
```json
{
  "session_id": "string",
  "status": "initiated|failed",
  "participating_nodes": 15
}
```

**Error Codes:**
- `400`: Invalid configuration
- `404`: Model not found
- `503`: Insufficient nodes available
```

### Documentation Maintenance

1. **Version Control**: Track documentation changes alongside code
2. **Automated Checks**: Use tools to verify code examples still work
3. **Regular Reviews**: Schedule quarterly documentation audits
4. **User Feedback**: Create channels for documentation improvement suggestions

---

## Quick Reference Card

### Naming Checklist
- [ ] Names clearly express intent
- [ ] Consistent conventions throughout codebase
- [ ] No ambiguous abbreviations
- [ ] Domain terminology used correctly

### Organization Checklist
- [ ] Clear directory structure
- [ ] One concept per file
- [ ] Related code grouped together
- [ ] Dependencies flow one direction

### Modularity Checklist
- [ ] High cohesion within modules
- [ ] Loose coupling between modules
- [ ] Clear module interfaces
- [ ] Independent testability

### Simplicity Checklist
- [ ] No premature optimization
- [ ] No speculative features (YAGNI)
- [ ] No code duplication (DRY)
- [ ] Simplest working solution (KISS)

### Documentation Checklist
- [ ] Code explains "why" not "what"
- [ ] All public APIs documented
- [ ] Examples provided
- [ ] Kept up-to-date with code

---

*Remember: These principles are guidelines, not rigid rules. Apply them thoughtfully based on your specific context and team needs. The goal is always to make code that helps you and your team move faster with confidence.*