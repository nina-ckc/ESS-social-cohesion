# ESS-social-cohesion
A research oriented simulation project using ESS data
## Overview

This project explores how trust, homophily, and social influence may shape immigration attitudes over time in a population.

The model uses a cleaned subset of European Social Survey (ESS) data to initialize agents, connect them in a network, and simulate attitude change across repeated time steps.

## Files

- `agent_simulation_ESS.py` — main simulation script
- `ess_social_cohesion_data.csv` — cleaned project dataset
- `requirements.txt` — Python packages needed to run the project

## What the model does

The script:

1. loads ESS-based data
2. cleans and rescales variables
3. initializes agents from the data
4. builds a homophily-based network
5. updates attitudes through neighbor influence
6. saves plots and results into an `outputs` folder

## Variables used

- generalized trust
- institutional trust
- immigration attitude
- age
- education

## How to run

Install the required packages:

```bash
pip install -r requirements.txt
