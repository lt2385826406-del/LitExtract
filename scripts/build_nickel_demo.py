"""
build_nickel_dag_v6.py

"""
import json
import glob
import os
import sys
from pathlib import Path
from collections import Counter

# Add project src/ to path for local module imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from dag_construction.cooccurrence_miner import run_step1
from dag_construction.dag_causal_builder import CausalHypothesisGraph

OUTPUT_DIR = 'KG_and_Causal/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("Nickel-based Alloy Causal DAG Build v6 (v7 Prompt semantic results -> all 68 papers)")
print("=" * 70)

#
result_files = sorted(glob.glob('outputs/nickel_results_v7/*_result.json'))
print(f"\nAvailable result files: {len(result_files)}")

all_entities = []
alloys_set = set()
total_process_counter = Counter()
total_phase_counter = Counter()
total_prop_counter = Counter()

for result_file in result_files:
    with open(result_file, 'r', encoding='utf-8') as fp:
        data = json.load(fp)

    sem = data.get('semantic_results', {})
    samples = sem.get('samples', [])
    if not samples:
        continue

    entities = {
        "elements": [],
        "alloy_names": [],
        "processes": [],
        "microstructures": [],
        "properties": [],
        "title": "",
        "folder": os.path.basename(result_file).replace('_result.json', ''),
    }

    for s in samples:
        comp = s.get('composition', {})
        # Alloy name
        name = comp.get('canonical_name', '')
        if name and str(name).lower() not in ('null', 'none', '', 'unknown'):
            entities["alloy_names"].append(str(name))
            alloys_set.add(str(name))

        # Elements from elements array
        elems = comp.get('elements', [])
        if isinstance(elems, list):
            for e in elems:
                if isinstance(e, dict):
                    en = e.get('element', '')
                    if en:
                        entities["elements"].append(en)
                elif isinstance(e, str) and e.strip():
                    entities["elements"].append(e)

        # Processes (v7 array format)
        for p in s.get('processes', []):
            method = p.get('method', '')
            if method and str(method).lower() not in ('null', 'none', ''):
                entities["processes"].append(str(method))
                total_process_counter[str(method)] += 1

        # Microstructures (v7 array format)
        for ms in s.get('microstructures', []):
            pts = ms.get('phase_type', [])
            pts_list = pts if isinstance(pts, list) else ([pts] if pts else [])
            for pt in pts_list:
                if pt and str(pt).lower() not in ('null', 'none', ''):
                    entities["microstructures"].append(str(pt))
                    total_phase_counter[str(pt)] += 1

        # Properties (v7 array format)
        for pr in s.get('properties', []):
            ptype = pr.get('type', '')
            if ptype and str(ptype).lower() not in ('null', 'none', ''):
                entities["properties"].append(str(ptype))
                total_prop_counter[str(ptype)] += 1

    # Deduplicate within paper
    for k in ["elements", "alloy_names", "processes", "microstructures", "properties"]:
        entities[k] = list(set(entities[k]))

    if any(entities[k] for k in ["elements", "alloy_names", "processes", "microstructures", "properties"]):
        all_entities.append(entities)

print(f"Valid papers: {len(all_entities)}/{len(result_files)}")

#
print(f"\n{'='*70}")
print("Entity Diversity Statistics:")
print(f"{'='*70}")
print(f"  Alloy types: {len(alloys_set)}")
print(f"  Process types: {len(total_process_counter)} ({sum(total_process_counter.values())} occurrences)")
print(f"  Phase types: {len(total_phase_counter)} ({sum(total_phase_counter.values())} occurrences)")
print(f"  Property types: {len(total_prop_counter)} ({sum(total_prop_counter.values())} occurrences)")

#
semantic_data_path = f'{OUTPUT_DIR}/nickel_v6_semantic_data.json'
with open(semantic_data_path, 'w', encoding='utf-8') as f:
    json.dump(all_entities, f, indent=2, ensure_ascii=False)
print(f"\nSemantic data saved to: {semantic_data_path}")

#
print(f"\n{'='*70}")
print("Step 1: Co-occurrence mining (min_freq=2)...")
print(f"{'='*70}")

candidates_path = f'{OUTPUT_DIR}/nickel_v6_candidates.json'
candidates = run_step1(semantic_data_path, candidates_path, min_freq=2)
print(f"Co-occurrence statistics complete: candidate edges={len(candidates)}")

#
print(f"\n{'='*70}")
print("Step 2+3: Domain constraints + Cycle removal...")
print(f"{'='*70}")

chg = CausalHypothesisGraph()
result = chg.build_dag_from_candidates(
    candidates,
    apply_constraints=True,
    enforce_dag=True,
    verbose=True,
)

dag = chg.cg
print(f"\n{'='*70}")
print("DAG Build Summary")
print(f"{'='*70}")
print(f"  Step1 candidates: {result['step1_added']}")
print(f"  Step2 removed:     {result['step2_removed']}")
print(f"  Step3 removed:     {result['step3_removed']}")
print(f"  Final nodes:       {result['final_nodes']}")
print(f"  Final edges:       {result['final_edges']}")
print(f"  Is DAG:         {result['is_dag']}")

# ========== 6. Save DAG result ==========
dag_result = {
    'metadata': {
        'source': 'nickel_alloys_v6',
        'n_papers': len(all_entities),
        'papers': [e['folder'] for e in all_entities],
        'version': 'v6_from_v7_prompt',
        'prompt_version': 'v7 (natural language schema + few-shot)',
    },
    'statistics': {
        'step1_candidates': result['step1_added'],
        'step2_removed': result['step2_removed'],
        'step3_removed': result['step3_removed'],
        'final_nodes': result['final_nodes'],
        'final_edges': result['final_edges'],
        'is_dag': result['is_dag'],
        'entity_diversity': {
            'alloys': len(alloys_set),
            'processes': len(total_process_counter),
            'microstructures': len(total_phase_counter),
            'properties': len(total_prop_counter),
        }
    },
    'entity_summary': {
        'alloys': sorted(list(alloys_set)),
        'top_processes': total_process_counter.most_common(30),
        'top_phases': total_phase_counter.most_common(30),
        'top_properties': total_prop_counter.most_common(30),
    },
    'nodes': [
        {
            'id': n,
            'type': dag.nodes[n].get('node_type', 'unknown'),
            'name': n.split(':')[1] if ':' in n else n,
        }
        for n in dag.nodes()
    ],
    'edges': [
        {
            'source': u, 'target': v,
            'source_name': u.split(':')[1] if ':' in u else u,
            'target_name': v.split(':')[1] if ':' in v else v,
            'confidence': d.get('confidence', 0),
            'strength': d.get('strength', 0),
            'cooccurrence_freq': len(d.get('evidence_ids', [])),
            'polarity': d.get('polarity', ''),
            'claim_type': d.get('claim_type', ''),
            'evidence_text': d.get('evidence_text', ''),
        }
        for u, v, d in dag.edges(data=True)
    ],
}

result_path = f'{OUTPUT_DIR}/nickel_v6_dag_result.json'
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(dag_result, f, indent=2, ensure_ascii=False)
print(f"\nDAG result saved to: {result_path}")

#
print(f"\n{'='*70}")
print("Top 10 Strongest Causal Edges (by confidence)")
print(f"{'='*70}")

edges_sorted = sorted(dag.edges(data=True),
                      key=lambda x: x[2].get('confidence', 0), reverse=True)

print(f"\n{'Rank':<5} {'Source':<55} {'Target':<55} {'Conf':<8} {'Type'}")
print("-" * 135)
for i, (u, v, d) in enumerate(edges_sorted[:10]):
    ut = dag.nodes[u].get('node_type', '?')
    vt = dag.nodes[v].get('node_type', '?')
    u_name = u.split(':')[1] if ':' in u else u
    v_name = v.split(':')[1] if ':' in v else v
    if len(u_name) > 52: u_name = u_name[:49] + '...'
    if len(v_name) > 52: v_name = v_name[:49] + '...'
    conf = d.get('confidence', 0)
    print(f"{i+1:<5} {u_name:<55} {v_name:<55} {conf:<8.4f} {ut}→{vt}")

#
print(f"\n{'='*70}")
print("Top 10 Material-Significant Causal Chains")
print("(Composition/Processing → Microstructure → Property)")
print(f"{'='*70}")

meaningful = []
for u, v, d in dag.edges(data=True):
    ut = dag.nodes[u].get('node_type', '')
    vt = dag.nodes[v].get('node_type', '')
    if ut in ('Processing', 'Element', 'Alloy') and vt in ('Microstructure', 'Property'):
        meaningful.append((u, v, d))
    elif ut == 'Microstructure' and vt == 'Property':
        meaningful.append((u, v, d))

meaningful.sort(key=lambda x: x[2].get('confidence', 0), reverse=True)
print(f"Found {len(meaningful)} material-meaningful causal edges")

print(f"\n{'Rank':<5} {'Source (Type)':<50} {'→':<3} {'Target (Type)':<50} {'Conf':<8}")
print("-" * 120)
for i, (u, v, d) in enumerate(meaningful[:10]):
    ut = dag.nodes[u].get('node_type', '')
    vt = dag.nodes[v].get('node_type', '')
    u_name = u.split(':')[1] if ':' in u else u
    v_name = v.split(':')[1] if ':' in v else v
    if len(u_name) > 48: u_name = u_name[:45] + '...'
    if len(v_name) > 48: v_name = v_name[:45] + '...'
    conf = d.get('confidence', 0)
    print(f"{i+1:<5} {u_name} ({ut:<20}) → {v_name} ({vt:<20}) {conf:<8.4f}")

#
print(f"\n{'='*70}")
print("Node Type Distribution:")
print(f"{'='*70}")
node_types = Counter()
for n in dag.nodes():
    nt = dag.nodes[n].get('node_type', 'unknown')
    node_types[nt] += 1
for nt, count in node_types.most_common():
    print(f"  {nt:<20}: {count:>4}")

# ========== 10. Save statistics info ==========
stats_path = f'{OUTPUT_DIR}/nickel_v6_dag_stats.json'
full_stats = {
    'metadata': dag_result['metadata'],
    'statistics': dag_result['statistics'],
    'node_type_distribution': dict(node_types),
    'entity_diversity': {
        'alloys': sorted(list(alloys_set)),
        'top_processes': total_process_counter.most_common(30),
        'top_phases': total_phase_counter.most_common(30),
        'top_properties': total_prop_counter.most_common(30),
    },
    'top_causal_edges': [
        {
            'rank': i+1,
            'source': u.split(':')[1] if ':' in u else u,
            'target': v.split(':')[1] if ':' in v else v,
            'src_type': dag.nodes[u].get('node_type', ''),
            'tgt_type': dag.nodes[v].get('node_type', ''),
            'confidence': d.get('confidence', 0),
            'polarity': d.get('polarity', ''),
        }
        for i, (u, v, d) in enumerate(edges_sorted[:20])
    ],
    'top_material_chains': [
        {
            'rank': i+1,
            'source': u.split(':')[1] if ':' in u else u,
            'target': v.split(':')[1] if ':' in v else v,
            'src_type': dag.nodes[u].get('node_type', ''),
            'tgt_type': dag.nodes[v].get('node_type', ''),
            'confidence': d.get('confidence', 0),
        }
        for i, (u, v, d) in enumerate(meaningful[:20])
    ],
}

with open(stats_path, 'w', encoding='utf-8') as f:
    json.dump(full_stats, f, indent=2, ensure_ascii=False)
print(f"\nStatistics saved to: {stats_path}")

print(f"\n{'='*70}")
print("Complete! DAG result saved")
print(f"{'='*70}")
