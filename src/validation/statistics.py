import json, os, math

with open(r'D:\python\LitExtract_Agent\verification\output_20260529\robustness_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

baseline = data['baseline']
print('=== BASELINE ===')
print(f'Nodes: {baseline["nodes"]}')
print(f'Edges: {baseline["edges"]}')
print(f'Weighted recovery: {baseline["weighted_recovery_rate"]:.4f}')
print(f'Direct recovery: {baseline["mechanism_summary"][0].get("direct_rate", "N/A")}')  # placeholder

print()
print('=== MECHANISM SUMMARY (Baseline) ===')
for m in baseline['mechanism_summary']:
    print(f'  {m["mechanism_id"]} {m["mechanism_name"]}: total={m["total"]}, direct={m["direct"]}, recovered_any={m["recovered_any"]}, rate={m["recovery_rate"]:.2%}')

print()
print('=== ROBUSTNESS SUMMARY ===')
for exp in data['experiments']:
    rate = exp['rate']
    scores = []
    node_rets = []
    edge_rets = []
    node_counts = []
    edge_counts = []
    for seed in exp['seeds']:
        if 'mechanism_eval' in seed:
            s = seed['mechanism_eval']['summary']['weighted_recovery_rate']
            scores.append(s)
        if 'graph_metrics' in seed:
            gm = seed['graph_metrics']['comparison']
            node_rets.append(gm['node_retention'])
            edge_rets.append(gm['edge_retention'])
        if 'dag_stats' in seed:
            node_counts.append(seed['dag_stats']['final_nodes'])
            edge_counts.append(seed['dag_stats']['final_edges'])
    
    if scores:
        avg_s = sum(scores)/len(scores)
        min_s = min(scores)
        max_s = max(scores)
        std_s = math.sqrt(sum((s-avg_s)**2 for s in scores)/len(scores))
        
        avg_nr = sum(node_rets)/len(node_rets)
        std_nr = math.sqrt(sum((nr-avg_nr)**2 for nr in node_rets)/len(node_rets))
        
        avg_er = sum(edge_rets)/len(edge_rets)
        std_er = math.sqrt(sum((er-avg_er)**2 for er in edge_rets)/len(edge_rets))
        
        avg_nodes = sum(node_counts)/len(node_counts)
        std_nodes = math.sqrt(sum((n-avg_nodes)**2 for n in node_counts)/len(node_counts))
        
        avg_edges = sum(edge_counts)/len(edge_counts)
        std_edges = math.sqrt(sum((e-avg_edges)**2 for e in edge_counts)/len(edge_counts))
        
        print(f'Rate {rate:.0%}:')
        print(f'  weighted_recovery: {avg_s:.4f} +/- {std_s:.4f}  [{min_s:.4f}, {max_s:.4f}]')
        print(f'  node_retention: {avg_nr:.4f} +/- {std_nr:.4f}')
        print(f'  edge_retention: {avg_er:.4f} +/- {std_er:.4f}')
        print(f'  nodes: {avg_nodes:.0f} +/- {std_nodes:.0f}')
        print(f'  edges: {avg_edges:.0f} +/- {std_edges:.0f}')
