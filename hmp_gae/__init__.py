# hmp_gae package
# Hypergraph Message-Passing Graph AutoEncoder for FedLLM immunization.
#
# Sub-modules:
#   - node_features : eta_i = f_enc(JL(Delta_i), stats, history)
#   - hypergraph    : k-NN hypergraphs (update + probe-behavior views),
#                     mutual/consensus relations, propagation operator
#   - encoder       : L-layer HMP encoder (node <-> hyperedge <-> node)
#   - decoder       : normalized-cosine adjacency decoder + hyperedge decoder
#   - losses        : V8 objective (incidence BCE + fixed-topology adjacency
#                     BCE + historical consistency)
#   - trust_scorer  : CSE-reject decision rules V4 / V5 / V8
#   - runtime       : HMPGAERuntime wiring everything together (used by the
#                     defense package)
