# ESS Social Cohesion ABM

A beginner-friendly agent-based simulation project using European Social Survey (ESS) data.

## What this project is

This project explores how **trust**, **homophily**, and **social influence** may shape attitudes over time in a population.

The simulation is based on survey-informed agents. Agents are created from ESS respondent characteristics, connected in a social network, and then allowed to influence one another across repeated time steps.

This is an **exploratory research project**, not a predictive model.

## Research question

How might trust, similarity between people, and local social influence affect the evolution of immigration attitudes in a social network?

## What the model does

The model:

1. loads ESS data
2. cleans selected variables
3. rescales variables to a common 0 to 1 range
4. samples synthetic agents from the data
5. builds a network based on similarity between agents
6. updates attitudes over time through neighbor influence
7. saves summary results and plots

## Variables used

The simulation uses these variables from the ESS dataset:

- generalized trust
- institutional trust
- immigration attitude
- age
- education

## Main ideas in the model

### 1. Agent initialization
Each agent starts with values based on sampled ESS respondent data.

### 2. Homophily
Agents are more likely to connect to similar others.

In this model, similarity is based mainly on:
- attitude
- education

### 3. Social influence
At each time step, an agent adjusts their attitude slightly toward the average attitude of their neighbors.

### 4. Trust
Trust affects how strongly an agent is influenced by others.

## Outputs

The model saves results such as:

- mean attitude over time
- attitude variance over time
- cohesion score over time
- network visualizations
- final attitude distributions
- simulation results as a CSV file

## Project structure

```text
ess-social-cohesion-abm/
├── src/
│   └── agent_simulation_ess.py
├── data/
│   └── ESS10e03_3.csv
├── outputs/
├── README.md
├── requirements.txt
└── .gitignore
