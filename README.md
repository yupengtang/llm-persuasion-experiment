# LLM Persuasion Experiment

A comprehensive framework for studying whether Large Language Models can persuade each other through dialogue, with controlled identity disclosure conditions.

## Overview

This project implements a systematic approach to test LLM persuasion capabilities based on experimental design principles from interpersonal persuasion research. The framework includes:

- **Dual-model dialogue system** for LLM-to-LLM persuasion testing
- **Identity disclosure experiments** (anonymous, disclosed, fake-disclosed conditions)
- **Interactive dashboard** for results visualization and analysis
- **Statistical analysis tools** for measuring persuasion effectiveness

## Research Questions

1. Can LLMs persuade each other through dialogue?
2. Does identity disclosure affect persuasion outcomes?
3. How do different model pairs perform in persuasion tasks?
4. What role does dialogue length play in persuasion success?

## Installation

```bash
git clone https://github.com/yourusername/llm-persuasion-experiment.git
cd llm-persuasion-experiment
pip install -r requirements.txt
```

## Quick Start

### 1. Set up API key
```bash
export OPENROUTER_API_KEY="sk-or-your-key-here"
```

### 2. Run experiment

#### Basic experiment (2 models)
```bash
python llm_persuasion_experiment.py \
    --models gpt-4o-mini gpt-3.5-turbo \
    --iterations 5 \
    --n_dilemmas 2 \
    --max_rounds 4 \
    --out_dir results/
```

#### Multi-model experiments

**3 models (3 pairings)**
```bash
python llm_persuasion_experiment.py \
    --models gpt-4o-mini gpt-3.5-turbo claude-3-haiku \
    --iterations 3 \
    --n_dilemmas 2 \
    --max_rounds 4 \
    --out_dir multi_model_results/
```

**4 models (6 pairings)**
```bash
python llm_persuasion_experiment.py \
    --models gpt-4o-mini gpt-3.5-turbo claude-3-haiku gemini-pro \
    --iterations 2 \
    --n_dilemmas 2 \
    --max_rounds 4 \
    --out_dir multi_model_results/
```

**5 models (10 pairings) - Recommended for research**
```bash
python llm_persuasion_experiment.py \
    --models gpt-4o-mini gpt-3.5-turbo claude-3-haiku gemini-pro llama-3.1-8b-instruct \
    --iterations 2 \
    --n_dilemmas 2 \
    --max_rounds 4 \
    --out_dir full_research_results/
```

#### Experiment Scale Calculation

**Model Pairings Formula:**
- n models = n(n-1)/2 pairings
- 2 models: 1 pairing (A vs B)
- 3 models: 3 pairings (A vs B, A vs C, B vs C)
- 4 models: 6 pairings
- 5 models: 10 pairings

**Total Sessions Formula:**
```
Total Sessions = Pairings × Iterations × Dilemmas × Identity Conditions
```

**Examples:**
- 2 models: 1 × 5 × 2 × 3 = **30 sessions**
- 3 models: 3 × 3 × 2 × 3 = **54 sessions**
- 4 models: 6 × 2 × 2 × 3 = **72 sessions**
- 5 models: 10 × 2 × 2 × 3 = **120 sessions**

**Cost Estimation:**
| Models | Pairings | Sessions | Est. Cost | Time |
|--------|----------|----------|-----------|------|
| 2      | 1        | 30       | $3-8      | 5-10 min |
| 3      | 3        | 54       | $6-15     | 15-30 min |
| 4      | 6        | 72       | $12-25    | 30-45 min |
| 5      | 10       | 120      | $20-40    | 60-90 min |

### 3. View results
```bash
streamlit run llm_persuasion_dashboard.py
```

## Project Structure

```
├── llm_persuasion_experiment.py    # Main experiment framework
├── llm_persuasion_dashboard.py     # Interactive results dashboard
├── main.py                         # Original consistency testing (legacy)
├── moral_dilemmas.json             # Ethical dilemma test cases
├── requirements.txt                # Python dependencies
├── run_experiments.sh              # Example experiment scripts
└── README.md                       # This file
```

## Experiment Design

### Identity Conditions
- **Anonymous**: Models don't know each other's identity
- **Disclosed**: Models know each other's real identity  
- **Fake-disclosed**: Models are given false identity information

### Dialogue Structure
1. Both models provide initial stance on moral dilemma
2. Models engage in N rounds of dialogue (default: 3)
3. Final stances are recorded and compared
4. Persuasion success is measured by stance changes

### Test Cases
The framework uses carefully designed moral dilemmas covering:
- AI medical decision-making
- Autonomous vehicle ethics
- Climate policy trade-offs
- Digital privacy vs. security
- Space colonization priorities

## Usage Examples

### Basic Experiment
```bash
python llm_persuasion_experiment.py \
    --models gpt-4o-mini claude-3-sonnet \
    --iterations 5 \
    --n_dilemmas 2
```

## Multi-Model Research Design

### Research Benefits of Multiple Models

**Why use multiple models?**
- **Cross-architecture comparison**: Different model families (GPT, Claude, Gemini, Llama)
- **Persuasion pattern analysis**: Which models are better persuaders vs persuadees
- **Statistical robustness**: Larger sample sizes for reliable conclusions
- **Generalizability**: Results apply across different AI systems

### Recommended Model Combinations

**Beginner Research (3 models)**
```bash
python llm_persuasion_experiment.py \
    --models gpt-4o-mini gpt-3.5-turbo claude-3-haiku \
    --iterations 3 \
    --n_dilemmas 2 \
    --max_rounds 4
```
- **3 pairings**: GPT-4o-mini vs GPT-3.5-turbo, GPT-4o-mini vs Claude-3-Haiku, GPT-3.5-turbo vs Claude-3-Haiku
- **54 total sessions**
- **Cost**: ~$6-15

**Comprehensive Research (5 models)**
```bash
python llm_persuasion_experiment.py \
    --models gpt-4o-mini gpt-3.5-turbo claude-3-haiku gemini-pro llama-3.1-8b-instruct \
    --iterations 2 \
    --n_dilemmas 2 \
    --max_rounds 4
```
- **10 pairings**: All possible combinations
- **120 total sessions**
- **Cost**: ~$20-40

### Extended Study
```bash
python llm_persuasion_experiment.py \
    --models gpt-4o-mini claude-3-sonnet llama-3.1-8b-instruct \
    --iterations 20 \
    --n_dilemmas 5 \
    --max_rounds 5 \
    --identity_disclosure_prob 0.6 \
    --fake_identity_prob 0.2
```

### Dashboard Analysis
```bash
streamlit run llm_persuasion_dashboard.py
```

## Output Files

- `persuasion_sessions.jsonl`: Detailed dialogue records
- `persuasion_results.csv`: Structured experiment results
- `persuasion_summary.csv`: Aggregated statistics by condition
- `persuasion_analysis.png`: Visualization charts

## Key Features

### Experiment Framework
- Stateless API calls to avoid memory leakage
- Randomized identity condition assignment
- Comprehensive error handling and retry logic
- Configurable sampling parameters

### Analysis Tools
- Interactive Plotly visualizations
- Statistical significance testing
- Model pair effectiveness analysis
- Dialogue round impact assessment

### Dashboard Features
- Real-time data visualization
- Multiple data source support (demo, upload, directory)
- Export functionality (CSV/JSON)
- Responsive design for different devices

## Methodology

This project implements experimental design principles from interpersonal persuasion research:

- **Benefits vs. downsides framing** to reduce defensiveness
- **Personal experience elicitation** for empathy building
- **Transparent disagreement** while maintaining organic discussion
- **Controlled identity disclosure** to test social influence effects

## Results Interpretation

### Key Metrics
- **Persuasion Rate**: Percentage of sessions where stance changed
- **Model Pair Effectiveness**: Which combinations work best
- **Identity Effect**: Impact of disclosure on persuasion
- **Dialogue Length**: Optimal number of rounds

### Statistical Analysis
- Chi-square tests for condition comparisons
- T-tests for model pair effectiveness
- ANOVA for dialogue length effects

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{llm_persuasion_experiment,
  title={LLM Persuasion Experiment Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/llm-persuasion-experiment}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Experimental design inspired by interpersonal persuasion research literature
- Moral dilemmas adapted from contemporary AI ethics frameworks
- Dashboard built with Streamlit and Plotly

## Troubleshooting

### Common Issues
- **API Key Error**: Ensure OPENROUTER_API_KEY is set correctly
- **Model Unavailable**: Check model names and availability
- **JSON Parsing**: Verify model output format matches expected structure
- **Memory Issues**: Reduce iterations or max_rounds for large experiments

### Debug Mode
```bash
python llm_persuasion_experiment.py --debug --iterations 1 --n_dilemmas 1
```

## Future Work

- [ ] Real-time experiment monitoring
- [ ] Additional statistical tests
- [ ] Multi-language support
- [ ] Custom dilemma creation tools
- [ ] Advanced visualization features
