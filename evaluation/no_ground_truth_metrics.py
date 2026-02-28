import numpy as np
from itertools import combinations
from collections import defaultdict


# ---------------------------------------------------------
# 1) Intra-List Diversity
# ---------------------------------------------------------
def intra_list_diversity(recommended_pids, project_vecs):
    """
    Diversity = average pairwise cosine distance between recommended projects.
    """
    if len(recommended_pids) < 2:
        return 0.0

    distances = []
    for p1, p2 in combinations(recommended_pids, 2):
        v1 = project_vecs[p1]
        v2 = project_vecs[p2]
        sim = float(np.dot(v1, v2))  # cosine similarity (vectors normalized)
        dist = 1 - sim
        distances.append(dist)

    return float(np.mean(distances))


# ---------------------------------------------------------
# 2) Cluster Coherence
# ---------------------------------------------------------
def cluster_coherence(project_index, project_vecs, field="interest"):
    """
    Measures how tight each domain cluster is.
    Lower average distance = better coherence.
    """
    domain_vectors = defaultdict(list)

    for pid, meta in project_index.items():
        for d in meta.get(field, []):
            domain_vectors[d].append(project_vecs[pid])

    coherence_scores = {}

    for domain, vecs in domain_vectors.items():
        if len(vecs) < 2:
            coherence_scores[domain] = 0.0
            continue

        distances = []
        for v1, v2 in combinations(vecs, 2):
            sim = float(np.dot(v1, v2))
            dist = 1 - sim
            distances.append(dist)

        coherence_scores[domain] = float(np.mean(distances))

    return coherence_scores


# ---------------------------------------------------------
# 3) Stability Under Noise
# ---------------------------------------------------------
def stability_under_noise(group_vec, recommender, noise_level=0.02):
    """
    Adds small noise to group_vec and checks how much recommendations change.
    Returns: similarity between original and noisy recommendation lists.
    """
    # Original recommendations
    original = [p["project_id"] for p in recommender.recommend_projects(group_vec)]

    # Add noise
    noise = np.random.normal(0, noise_level, size=group_vec.shape)
    noisy_vec = group_vec + noise
    noisy_vec = noisy_vec / np.linalg.norm(noisy_vec)

    # New recommendations
    new = [p["project_id"] for p in recommender.recommend_projects(noisy_vec)]

    # Overlap ratio
    overlap = len(set(original) & set(new)) / len(original)
    return overlap


# ---------------------------------------------------------
# 4) Sensitivity to Profile Changes
# ---------------------------------------------------------
def sensitivity_to_profile_changes(group_vec, recommender, direction_vec, strength=0.3):
    """
    Tests if recommendations change logically when we push the group_vec
    toward a specific domain vector (e.g., AI/ML).
    """
    # Original recommendations
    original = [p["project_id"] for p in recommender.recommend_projects(group_vec)]

    # Modify group_vec toward a domain
    modified = group_vec + strength * direction_vec
    modified = modified / np.linalg.norm(modified)

    # New recommendations
    new = [p["project_id"] for p in recommender.recommend_projects(modified)]

    # How many new projects appear?
    change_ratio = 1 - (len(set(original) & set(new)) / len(original))
    return change_ratio
