# AlignTune Reward Functions Documentation

## Overview

AlignTune provides **27+ prebuilt reward functions** covering all major aspects of text generation quality, safety, and task-specific performance. These reward functions can be used individually or combined to create sophisticated reward landscapes for RLHF training.

## Available Reward Functions

### **Basic Quality Rewards**

#### 1. **Length Reward** (`length`)
- **Purpose**: Encourages appropriate response length
- **Parameters**: `min_length`, `max_length`
- **Use Case**: Preventing too short or too long responses
- **Example**: `{"type": "length", "params": {"min_length": 10, "max_length": 500}}`

#### 2. **Coherence Reward** (`coherence`)
- **Purpose**: Measures logical flow and coherence of text
- **Parameters**: None
- **Use Case**: Ensuring responses make logical sense
- **Example**: `{"type": "coherence", "weight": 1.0}`

#### 3. **Fluency Reward** (`fluency`)
- **Purpose**: Measures grammatical correctness and fluency
- **Parameters**: None
- **Use Case**: Ensuring grammatically correct responses
- **Example**: `{"type": "fluency", "weight": 0.8}`

### **Task-Specific Rewards**

#### 4. **Sentiment Reward** (`sentiment`)
- **Purpose**: Encourages specific sentiment (positive, negative, neutral)
- **Parameters**: `target_sentiment`
- **Use Case**: Customer service, emotional support
- **Example**: `{"type": "sentiment", "params": {"target_sentiment": "positive"}}`

#### 5. **Safety Reward** (`safety`)
- **Purpose**: Detects and penalizes harmful content
- **Parameters**: `strict` (bool)
- **Use Case**: Content moderation, safe AI
- **Example**: `{"type": "safety", "params": {"strict": true}}`

#### 6. **Factuality Reward** (`factuality`)
- **Purpose**: Measures factual accuracy
- **Parameters**: None
- **Use Case**: Information retrieval, Q&A systems
- **Example**: `{"type": "factuality", "weight": 1.2}`

#### 7. **Bias Reward** (`bias`)
- **Purpose**: Detects and penalizes biased content
- **Parameters**: None
- **Use Case**: Fair AI, unbiased responses
- **Example**: `{"type": "bias", "weight": 0.5}`

### **Generation Quality Metrics**

#### 8. **BLEU Reward** (`bleu`)
- **Purpose**: Measures n-gram overlap with reference
- **Parameters**: `n_gram`, `smooth`
- **Use Case**: Translation, summarization
- **Example**: `{"type": "bleu", "params": {"n_gram": 4}}`

#### 9. **ROUGE Reward** (`rouge`)
- **Purpose**: Measures recall-oriented understudy for gisting evaluation
- **Parameters**: `rouge_type` (rouge-1, rouge-2, rouge-l)
- **Use Case**: Summarization, text generation
- **Example**: `{"type": "rouge", "params": {"rouge_type": "rouge-l"}}`

#### 10. **METEOR Reward** (`meteor`)
- **Purpose**: Advanced translation evaluation metric
- **Parameters**: None
- **Use Case**: Translation quality
- **Example**: `{"type": "meteor", "weight": 0.7}`

#### 11. **BERTScore Reward** (`bertscore`)
- **Purpose**: Contextual embedding-based similarity
- **Parameters**: `model_name`
- **Use Case**: Semantic similarity, paraphrase detection
- **Example**: `{"type": "bertscore", "params": {"model_name": "microsoft/DialoGPT-medium"}}`

### **Code Quality Rewards**

#### 12. **Code Syntax Reward** (`code_syntax`)
- **Purpose**: Validates code syntax correctness
- **Parameters**: `language`
- **Use Case**: Code generation, programming assistance
- **Example**: `{"type": "code_syntax", "params": {"language": "python"}}`

#### 13. **Code Execution Reward** (`code_execution`)
- **Purpose**: Tests if code runs without errors
- **Parameters**: `timeout`, `safe_mode`
- **Use Case**: Code generation, programming education
- **Example**: `{"type": "code_execution", "params": {"timeout": 5, "safe_mode": true}}`

#### 14. **Code Completeness Reward** (`code_completeness`)
- **Purpose**: Measures code completeness and structure
- **Parameters**: None
- **Use Case**: Code generation quality
- **Example**: `{"type": "code_completeness", "weight": 1.0}`

### **Math and Reasoning Rewards**

#### 15. **Math Correctness Reward** (`math_correctness`)
- **Purpose**: Validates mathematical correctness
- **Parameters**: `tolerance`
- **Use Case**: Math problem solving, scientific computation
- **Example**: `{"type": "math_correctness", "params": {"tolerance": 1e-6}}`

#### 16. **Logical Consistency Reward** (`logical_consistency`)
- **Purpose**: Measures logical consistency in reasoning
- **Parameters**: None
- **Use Case**: Reasoning tasks, logical problem solving
- **Example**: `{"type": "logical_consistency", "weight": 1.0}`

#### 17. **Commonsense Reward** (`commonsense`)
- **Purpose**: Evaluates commonsense reasoning
- **Parameters**: None
- **Use Case**: General reasoning, everyday knowledge
- **Example**: `{"type": "commonsense", "weight": 0.8}`

### **Specialized Rewards**

#### 18. **Hallucination Reward** (`hallucination`)
- **Purpose**: Detects and penalizes hallucinated information
- **Parameters**: None
- **Use Case**: Factual accuracy, information retrieval
- **Example**: `{"type": "hallucination", "weight": 1.5}`

#### 19. **Toxicity Reward** (`toxicity`)
- **Purpose**: Detects toxic, harmful, or offensive content
- **Parameters**: `threshold`
- **Use Case**: Content moderation, safe AI
- **Example**: `{"type": "toxicity", "params": {"threshold": 0.5}}`

#### 20. **Politeness Reward** (`politeness`)
- **Purpose**: Encourages polite and respectful language
- **Parameters**: None
- **Use Case**: Customer service, professional communication
- **Example**: `{"type": "politeness", "weight": 0.7}`

#### 21. **Helpfulness Reward** (`helpfulness`)
- **Purpose**: Measures how helpful and informative the response is
- **Parameters**: None
- **Use Case**: Q&A systems, customer support
- **Example**: `{"type": "helpfulness", "weight": 1.0}`

#### 22. **Honesty Reward** (`honesty`)
- **Purpose**: Encourages honest and truthful responses
- **Parameters**: None
- **Use Case**: Trustworthy AI, factual accuracy
- **Example**: `{"type": "honesty", "weight": 1.2}`

### **Multi-Modal Rewards (Future)**

#### 23. **Image Relevance Reward** (`image_relevance`)
- **Purpose**: Measures relevance to provided images
- **Parameters**: None
- **Use Case**: Multi-modal tasks, image captioning
- **Example**: `{"type": "image_relevance", "weight": 1.0}`

#### 24. **Audio Quality Reward** (`audio_quality`)
- **Purpose**: Measures audio quality and clarity
- **Parameters**: None
- **Use Case**: Speech generation, audio processing
- **Example**: `{"type": "audio_quality", "weight": 1.0}`

## Usage Examples

### **Basic Configuration**

```yaml
# Single reward function
rewards:
 - type: "length"
 weight: 1.0
 params:
 min_length: 10
 max_length: 500
```

### **Multiple Reward Functions**

```yaml
# Multiple reward functions with different weights
rewards:
 - type: "length"
 weight: 0.5
 params:
 min_length: 20
 max_length: 300
 
 - type: "safety"
 weight: 2.0
 params:
 strict: true
 
 - type: "helpfulness"
 weight: 1.5
 
 - type: "coherence"
 weight: 1.0
```

### **Task-Specific Configurations**

#### **Customer Support Bot**
```yaml
rewards:
 - type: "politeness"
 weight: 1.5
 - type: "helpfulness"
 weight: 2.0
 - type: "safety"
 weight: 1.0
 - type: "length"
 weight: 0.5
 params:
 min_length: 50
 max_length: 200
```

#### **Code Generation Assistant**
```yaml
rewards:
 - type: "code_syntax"
 weight: 2.0
 params:
 language: "python"
 - type: "code_execution"
 weight: 1.5
 params:
 timeout: 5
 safe_mode: true
 - type: "code_completeness"
 weight: 1.0
 - type: "helpfulness"
 weight: 0.8
```

#### **Math Problem Solver**
```yaml
rewards:
 - type: "math_correctness"
 weight: 3.0
 params:
 tolerance: 1e-6
 - type: "logical_consistency"
 weight: 1.5
 - type: "coherence"
 weight: 1.0
 - type: "length"
 weight: 0.3
 params:
 min_length: 10
 max_length: 1000
```

#### **Content Moderation System**
```yaml
rewards:
 - type: "safety"
 weight: 3.0
 params:
 strict: true
 - type: "toxicity"
 weight: 2.5
 params:
 threshold: 0.3
 - type: "bias"
 weight: 2.0
 - type: "factuality"
 weight: 1.5
```

## Advanced Configuration

### **Custom Reward Weights**

```python
from aligntune import RewardConfig, RewardType

# Create custom reward configuration
rewards = [
 RewardConfig(
 reward_type=RewardType.SAFETY,
 weight=2.0,
 params={"strict": True}
 ),
 RewardConfig(
 reward_type=RewardType.HELPFULNESS,
 weight=1.5
 ),
 RewardConfig(
 reward_type=RewardType.LENGTH,
 weight=0.5,
 params={"min_length": 20, "max_length": 300}
 )
]
```

### **Dynamic Reward Weighting**

```yaml
# Example with dynamic weighting based on task
rewards:
 - type: "safety"
 weight: 2.0 # High priority for safety
 - type: "helpfulness"
 weight: 1.5 # Medium-high priority
 - type: "coherence"
 weight: 1.0 # Standard priority
 - type: "length"
 weight: 0.3 # Low priority
```

## Reward Function Performance

### **Computational Efficiency**
- **Fast**: Length, Coherence, Fluency, Sentiment
- **Medium**: Safety, Toxicity, Bias, Politeness
- **Slow**: BERTScore, Code Execution, Math Correctness

### **Memory Usage**
- **Low**: Basic rewards (Length, Coherence, etc.)
- **Medium**: Task-specific rewards (Safety, Sentiment, etc.)
- **High**: Model-based rewards (BERTScore, Code Execution, etc.)

### **Accuracy vs Speed Trade-offs**
- **High Accuracy**: BERTScore, Math Correctness, Code Execution
- **Balanced**: Safety, Toxicity, Helpfulness
- **Fast Approximation**: Length, Coherence, Fluency

## Best Practices

### **Reward Selection**
1. **Start Simple**: Begin with basic rewards (length, coherence, safety)
2. **Add Task-Specific**: Include rewards relevant to your use case
3. **Balance Weights**: Ensure no single reward dominates
4. **Monitor Performance**: Track reward trends during training

### **Weight Tuning**
1. **Safety First**: Always include safety rewards with high weight
2. **Task Relevance**: Higher weights for task-specific rewards
3. **Quality Balance**: Balance between different quality aspects
4. **Iterative Tuning**: Adjust weights based on training results

### **Performance Optimization**
1. **Use Fast Rewards**: Prefer fast rewards for real-time applications
2. **Batch Processing**: Process multiple texts together when possible
3. **Caching**: Cache model outputs for repeated computations
4. **Device Selection**: Use appropriate devices (CPU/GPU) for each reward

## Extending Reward Functions

### **Custom Reward Functions**

```python
from aligntune import RewardFunction, RewardType

class CustomReward(RewardFunction):
 def __init__(self, config: RewardConfig):
 super().__init__(config)
 # Initialize your custom reward logic
 
 def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
 # Implement your custom reward computation
 return 0.8 # Return reward score between 0 and 1

# Register custom reward
from aligntune import RewardRegistry
RewardRegistry.register_reward("custom", CustomReward)
```

### **Combining Rewards**

```python
def combined_reward(text: str, rewards: List[RewardFunction]) -> float:
 """Combine multiple reward functions."""
 total_score = 0.0
 total_weight = 0.0
 
 for reward_func in rewards:
 score = reward_func.compute(text)
 weight = reward_func.config.weight
 total_score += score * weight
 total_weight += weight
 
 return total_score / total_weight if total_weight > 0 else 0.0
```

## Monitoring and Debugging

### **Reward Tracking**
- Monitor individual reward scores during training
- Track reward trends and convergence
- Identify which rewards are most influential
- Debug reward computation issues

### **Common Issues**
1. **Reward Collapse**: All rewards converge to similar values
2. **Reward Instability**: High variance in reward scores
3. **Reward Conflicts**: Conflicting objectives between rewards
4. **Performance Bottlenecks**: Slow reward computation

### **Debugging Tools**
- Reward score logging and visualization
- Individual reward breakdown analysis
- Performance profiling for slow rewards
- A/B testing for reward configurations

---

**AlignTune** provides a comprehensive suite of reward functions to create sophisticated reward landscapes for any RLHF training scenario. Choose the right combination of rewards for your specific use case and achieve optimal model performance!