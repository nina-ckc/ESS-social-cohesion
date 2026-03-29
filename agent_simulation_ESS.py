import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def load_and_clean_ess(file_path, country_code="GB", round_value=10):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find the file at:\n{file_path}")

    df = pd.read_csv(file_path)

    keep_cols = [
        "cntry", "essround", "ppltrst", "trstplc",
        "imsmetn", "agea", "eduyrs"
    ]
    df = df[keep_cols].copy()

    df.columns = [
        "country", "round", "generalized_trust", "institutional_trust",
        "immigration_attitude", "age", "education"
    ]

    df = df[(df["country"] == country_code) & (df["round"] == round_value)].copy()

    numeric_cols = [
        "generalized_trust", "institutional_trust",
        "immigration_attitude", "age", "education"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    missing_codes = [
        7, 8, 9,
        66, 77, 88, 99,
        666, 777, 888, 999,
        6666, 7777, 8888, 9999
    ]
    df = df.replace(missing_codes, np.nan)

    cleaned_df = df.dropna(subset=numeric_cols).reset_index(drop=True)

    if cleaned_df.empty:
        raise ValueError(
            f"No usable rows found after filtering country={country_code}, round={round_value}."
        )

    print("\n=== CLEANED RAW DESCRIPTIVE STATISTICS ===")
    print(cleaned_df[numeric_cols].describe())

    scaled_df = cleaned_df.copy()
    for col in numeric_cols:
        min_val = scaled_df[col].min()
        max_val = scaled_df[col].max()
        scaled_df[col] = 0.5 if max_val == min_val else (scaled_df[col] - min_val) / (max_val - min_val)

    print("\n=== SCALED DESCRIPTIVE STATISTICS ===")
    print(scaled_df[numeric_cols].describe())
    print("\nRows after cleaning:", len(cleaned_df))

    return cleaned_df, scaled_df


def initialize_agents(scaled_df, n_agents=300, seed=42):
    rng = np.random.default_rng(seed)
    sampled_idx = rng.choice(scaled_df.index, size=n_agents, replace=True)
    agents = scaled_df.loc[sampled_idx].reset_index(drop=True).copy()

    agents["agent_id"] = np.arange(n_agents)
    agents["attitude"] = agents["immigration_attitude"]
    agents["trust"] = (agents["generalized_trust"] + agents["institutional_trust"]) / 2.0

    return agents


def build_network(agents, homophily_strength=3.0, base_tie_prob=0.05, seed=42):
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    G.add_nodes_from(agents["agent_id"].tolist())

    n = len(agents)

    for i in range(n):
        for j in range(i + 1, n):
            a_i = agents.loc[i, "attitude"]
            a_j = agents.loc[j, "attitude"]
            e_i = agents.loc[i, "education"]
            e_j = agents.loc[j, "education"]

            attitude_similarity = 1.0 - abs(a_i - a_j)
            education_similarity = 1.0 - abs(e_i - e_j)

            similarity = 0.7 * attitude_similarity + 0.3 * education_similarity
            tie_prob = base_tie_prob * (similarity ** homophily_strength)

            if rng.random() < tie_prob:
                G.add_edge(i, j)

    return G


def update_attitudes(agents, G, influence_strength=0.2):
    updated_agents = agents.copy()
    attitude_map = agents.set_index("agent_id")["attitude"].to_dict()
    trust_map = agents.set_index("agent_id")["trust"].to_dict()

    new_attitudes = []

    for i in agents["agent_id"]:
        neighbors = list(G.neighbors(i))
        current_attitude = attitude_map[i]
        trust_i = trust_map[i]

        if not neighbors:
            new_attitudes.append(current_attitude)
            continue

        neighbor_mean = np.mean([attitude_map[n] for n in neighbors])
        updated_attitude = current_attitude + influence_strength * trust_i * (neighbor_mean - current_attitude)
        new_attitudes.append(np.clip(updated_attitude, 0.0, 1.0))

    updated_agents["attitude"] = new_attitudes
    return updated_agents


def compute_metrics(agents, G):
    attitudes = agents["attitude"].values
    attitude_map = agents.set_index("agent_id")["attitude"].to_dict()
    trust_map = agents.set_index("agent_id")["trust"].to_dict()

    mean_attitude = float(np.mean(attitudes))
    attitude_variance = float(np.var(attitudes))

    edge_scores = []
    for i, j in G.edges():
        attitude_similarity = 1.0 - abs(attitude_map[i] - attitude_map[j])
        edge_trust = (trust_map[i] + trust_map[j]) / 2.0
        edge_scores.append(attitude_similarity * edge_trust)

    cohesion_score = float(np.mean(edge_scores)) if edge_scores else np.nan

    return {
        "mean_attitude": mean_attitude,
        "attitude_variance": attitude_variance,
        "cohesion_score": cohesion_score
    }


def run_simulation(
    agents,
    homophily_strength=3.0,
    influence_strength=0.2,
    n_steps=30,
    base_tie_prob=0.05,
    seed=42
):
    agents_t = agents.copy()

    G = build_network(
        agents=agents_t,
        homophily_strength=homophily_strength,
        base_tie_prob=base_tie_prob,
        seed=seed
    )

    print(f"\nBuilt network for scenario H={homophily_strength}, I={influence_strength}")
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    results = []

    metrics = compute_metrics(agents_t, G)
    metrics["time"] = 0
    metrics["homophily_strength"] = homophily_strength
    metrics["influence_strength"] = influence_strength
    results.append(metrics)

    for t in range(1, n_steps + 1):
        agents_t = update_attitudes(
            agents=agents_t,
            G=G,
            influence_strength=influence_strength
        )
        metrics = compute_metrics(agents_t, G)
        metrics["time"] = t
        metrics["homophily_strength"] = homophily_strength
        metrics["influence_strength"] = influence_strength
        results.append(metrics)

    return pd.DataFrame(results), G, agents_t


def plot_trajectories(results_df, output_folder):
    results_df = results_df.copy()
    results_df["scenario"] = (
        "H=" + results_df["homophily_strength"].astype(str)
        + ", I=" + results_df["influence_strength"].astype(str)
    )

    metrics_to_plot = ["mean_attitude", "attitude_variance", "cohesion_score"]

    for metric in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(9, 5))

        for scenario, subdf in results_df.groupby("scenario"):
            ax.plot(subdf["time"], subdf[metric], label=scenario)

        ax.set_xlabel("Time")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()} Over Time")
        ax.legend()
        fig.tight_layout()

        save_path = os.path.join(output_folder, f"{metric}.png")
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot: {save_path}")
        plt.close(fig)


def plot_network_snapshot(G, agents, output_folder, filename):
    fig, ax = plt.subplots(figsize=(8, 8))
    pos = nx.spring_layout(G, seed=42)
    node_colors = agents.sort_values("agent_id")["attitude"].values

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.viridis, node_size=40, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5, ax=ax)

    ax.set_title("Network Snapshot Colored by Attitude")
    ax.axis("off")

    save_path = os.path.join(output_folder, filename)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved network plot: {save_path}")
    plt.close(fig)


def plot_attitude_distribution(agents, output_folder, filename):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(agents["attitude"], bins=20)
    ax.set_xlabel("Attitude")
    ax.set_ylabel("Frequency")
    ax.set_title("Final Attitude Distribution")

    save_path = os.path.join(output_folder, filename)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved histogram: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    file_path = r"C:\Users\Naina\OneDrive\Documents\meow\UoA\ESS10-integrated file\ESS10e03_3.csv"
    output_folder = r"C:\Users\Naina\OneDrive\Documents\meow\UoA\simulation_outputs"
    os.makedirs(output_folder, exist_ok=True)

    country_code = "GB"
    round_value = 10

    cleaned_df, scaled_df = load_and_clean_ess(
        file_path=file_path,
        country_code=country_code,
        round_value=round_value
    )

    agents = initialize_agents(scaled_df=scaled_df, n_agents=300, seed=42)

    scenarios = [
        {"homophily_strength": 1.5, "influence_strength": 0.10},
        {"homophily_strength": 3.0, "influence_strength": 0.10},
        {"homophily_strength": 3.0, "influence_strength": 0.25},
        {"homophily_strength": 5.0, "influence_strength": 0.25},
    ]

    all_results = []

    for s in scenarios:
        results_df, G, final_agents = run_simulation(
            agents=agents,
            homophily_strength=s["homophily_strength"],
            influence_strength=s["influence_strength"],
            n_steps=30,
            base_tie_prob=0.05,
            seed=42
        )
        all_results.append(results_df)

        tag = f"H{s['homophily_strength']}_I{s['influence_strength']}"
        plot_network_snapshot(G, final_agents, output_folder, f"network_{tag}.png")
        plot_attitude_distribution(final_agents, output_folder, f"hist_{tag}.png")

    all_results_df = pd.concat(all_results, ignore_index=True)

    print("\n=== SIMULATION RESULTS: FIRST 10 ROWS ===")
    print(all_results_df.head(10))

    print("\n=== FINAL TIME STEP FOR EACH SCENARIO ===")
    final_rows = all_results_df.groupby(
        ["homophily_strength", "influence_strength"], as_index=False
    ).tail(1)
    print(final_rows)

    csv_path = os.path.join(output_folder, "simulation_results.csv")
    all_results_df.to_csv(csv_path, index=False)
    print(f"\nSaved results CSV: {csv_path}")

    plot_trajectories(all_results_df, output_folder)