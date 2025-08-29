def extract_dag(m, n, Ap, Ai, Ax, B, C):
	DAG = [[] for i in range(m)]
	for j1 in range(0, m):
		for j2 in range(Ap[j1], Ap[j1 + 1]):
			DAG[j1].append(Ai[j2])
	return DAG
