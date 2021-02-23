#include "cometmodule.h"


double *lnfacs;
int *m2w_info;
weight_t *weightHash;
margin2weight_t *margin2weightHash;

/******************************************************************************
* Python utility functions
******************************************************************************/
PyObject *py_set_random_seed(PyObject *self, PyObject *args){
  // Parse params
    int seed;
    if (! PyArg_ParseTuple( args, "i", &seed )) return NULL;

    // Intialize random number generator with the given seed
    srand((unsigned) seed);

    return Py_BuildValue(""); // returns NULL
}

PyObject *py_precompute_factorials(PyObject *self, PyObject *args){
	// Parse params
    int N;
    if (! PyArg_ParseTuple( args, "i", &N )) return NULL;

    // Compute log factorials
    precompute_factorials(N);
    return Py_BuildValue(""); // returns NULL
}

PyObject *free_factorials(PyObject *self, PyObject *args) {
	free(lnfacs);
	return Py_BuildValue(""); // returns NULL
}


PyObject *py_set_weight(PyObject *self, PyObject *args) {
    char weight_function_char;
    if (! PyArg_ParseTuple( args, "c", &weight_function_char)) {
        return NULL;
    }
    weight_func_c =  weight_function_char;
    return Py_BuildValue("");
}

/******************************************************************************
* Python enumeration/sampling functions
******************************************************************************/
PyObject *py_exhaustive(PyObject *self, PyObject *args){
    int k, numPatients, numGenes, numSets; // parameters
    PyObject *solns, *weights, *patients2mutatedGenes, *gene2numMutations;
    PyObject *probs, *tables;
    PyObject *ret;
    mutation_data_t *mut_data;
    frozen_arrays_t *frozen;
    double pvalthresh;

    /* Parse Python arguments */
    if (! PyArg_ParseTuple( args, "iiiO!O!d", &k, &numGenes, &numPatients, &PyList_Type,
                            &patients2mutatedGenes, &PyList_Type,
                            &gene2numMutations, &pvalthresh)) {
        return NULL;
    }

    mut_data = mut_data_allocate(numPatients, numGenes);
    init_mutation_data (mut_data, numPatients, numGenes,
                        patients2mutatedGenes, gene2numMutations);

    numSets = choose(numGenes, k);

    frozen = comet_exhaustive(mut_data, k, &numSets, pvalthresh);
    mut_data_free(mut_data);

    solns = PyList_New(numSets);
    weights = PyList_New(numSets);
    tables = PyList_New(numSets);
    probs = PyList_New(numSets);

    frozensets2pylists(frozen, numSets, k, solns, weights, tables, probs);
    ret = Py_BuildValue("OOOO", solns, weights, tables, probs);

    frozen_free(numSets, frozen);
    Py_DECREF(tables);
    Py_DECREF(probs);
    Py_DECREF(solns);
    Py_DECREF(weights);
    return ret;
}

/* the phi score, callable from Python given required information*/
PyObject *py_comet_phi(PyObject *self, PyObject *args){
    int i, k, numPatients, numGenes, coCutoff, numPermutations, func; // parameters
    int *genes;
    PyObject *py_genes, *patients2mutatedGenes, *gene2numMutations, *result;
    mutation_data_t *A;
    double exactPvalthresh, binomPvalthresh, score;

    /* Parse Python arguments */
    if (! PyArg_ParseTuple( args, "iiiO!O!ddiiO!", &k, &numGenes, &numPatients,
                            &PyList_Type, &patients2mutatedGenes, &PyList_Type, &gene2numMutations,
                            &exactPvalthresh, &binomPvalthresh, &coCutoff, &numPermutations,
                            &PyList_Type, &py_genes)) {
        return NULL;
    }
    // printf("1 done...");
    int num_entries = 1 << k;
    int tbl[num_entries];
    int **co_elem = malloc(sizeof(int *) * k);
    genes = malloc(sizeof(int) * k);
    for (i = 0; i < k; i++) genes[i] = (int) PyLong_AsLong (PyList_GetItem(py_genes, i));
    A = mut_data_allocate(numPatients, numGenes);
    init_mutation_data (A, numPatients, numGenes,
                        patients2mutatedGenes, gene2numMutations);

    contingency_tbl_gen(A, k, tbl, genes);
    // printf("2 done...");

    for (i=0; i< k; i++){
      co_elem[i] = get_co_cells(i+1);
    }

    comet_phi(genes, k, numPatients, A, tbl, co_elem, &score, &func, coCutoff, numPermutations, binomPvalthresh, exactPvalthresh);
    // printf("3 done...");

    result = Py_BuildValue("di", score, func);
    // printf("4 done...\n");
    // printf("%f %d\n", score, func);
    // free(tbl);
    // printf("5 done...\n");
    for (i = 0; i < k; i++)
      if(co_elem[i] != NULL) free(co_elem[i]);
    // printf("6 done...\n");
    return result;


}

PyObject *py_exact_test_v2(PyObject *self, PyObject *args){
    int i, k, numPatients, numGenes, coCutoff, numPermutations, func; // parameters
    int *genes;
    PyObject *py_genes, *patients2mutatedGenes, *gene2numMutations, *result;
    mutation_data_t *A;
    double exactPvalthresh, binomPvalthresh, score;

    /* Parse Python arguments */
    if (! PyArg_ParseTuple( args, "iiiO!O!ddiiO!", &k, &numGenes, &numPatients,
                            &PyList_Type, &patients2mutatedGenes, &PyList_Type, &gene2numMutations,
                            &exactPvalthresh, &binomPvalthresh, &coCutoff, &numPermutations,
                            &PyList_Type, &py_genes)) {
        return NULL;
    }
    // printf("1 done...");
    int num_entries = 1 << k;
    int tbl[num_entries];
    int **co_elem = malloc(sizeof(int *) * k);
    genes = malloc(sizeof(int) * k);
    for (i = 0; i < k; i++) genes[i] = (int) PyLong_AsLong (PyList_GetItem(py_genes, i));
    A = mut_data_allocate(numPatients, numGenes);
    init_mutation_data (A, numPatients, numGenes,
                        patients2mutatedGenes, gene2numMutations);

    contingency_tbl_gen(A, k, tbl, genes);
    // printf("2 done...");

    for (i=0; i< k; i++){
      co_elem[i] = get_co_cells(i+1);
    }

    comet_phi_e(genes, k, numPatients, A, tbl, co_elem, &score, &func, coCutoff, numPermutations, binomPvalthresh, exactPvalthresh, 0);
    // printf("3 done...");

    result = Py_BuildValue("di", score, func);
    // printf("4 done...\n");
    // printf("%f %d\n", score, func);
    // free(tbl);
    // printf("5 done...\n");
    for (i = 0; i < k; i++)
      if(co_elem[i] != NULL) free(co_elem[i]);
    // printf("6 done...\n");
    return result;
}

/* the permutation test, callable from Python given required information*/
PyObject *py_permutation_test(PyObject *self, PyObject *args){
    int i, k, numPatients, numGenes, numPermutations, permute_count; // parameters
    PyObject *py_genes, *patients2mutatedGenes, *gene2numMutations, *result;
    mutation_data_t *A;

    /* Parse Python arguments */
    if (! PyArg_ParseTuple( args, "iiiO!O!iO!", &k, &numGenes, &numPatients,
                            &PyList_Type, &patients2mutatedGenes, &PyList_Type, &gene2numMutations,
                            &numPermutations, &PyList_Type, &py_genes)) {
        return NULL;
    }
    int num_entries = 1 << k;
    int tbl[num_entries];
    int *freq = malloc(sizeof(int) * k);
    int *genes = malloc(sizeof(int) * k);
    for (i = 0; i < k; i++) genes[i] = (int) PyLong_AsLong (PyList_GetItem(py_genes, i));
    A = mut_data_allocate(numPatients, numGenes);
    init_mutation_data (A, numPatients, numGenes,
                        patients2mutatedGenes, gene2numMutations);

    contingency_tbl_gen(A, k, tbl, genes);

    for (i=0; i< k; i++){
      freq[i] = A->G2numMuts[genes[i]];
    }
    permute_count = comet_permutation_test(k, numPermutations, numPatients, genes, freq, tbl);

    result = Py_BuildValue("d", permute_count/numPermutations);
    // free(tbl);
    free(genes);
    free(freq);

    return result;


}

/* The general exact test, callable from the Python driver. */
PyObject *py_exact_test(PyObject *self, PyObject *args){
  // Parameters
  int k, N; // k: gene set size; N: number of samples
  PyObject *py_tbl, *result; // FLAT Python contingency table
  int *tbl; // C contingency table
  double pvalthresh;

  // Computation variables
  double final_p_value;
  int final_num_tbls;

  int num_entries, i; // Helper variables

  // Parse parameters
  if (! PyArg_ParseTuple( args, "iiO!d", &k, &N, &PyList_Type, &py_tbl, &pvalthresh ))
    return NULL;

  // Convert PyList into C array
  num_entries = 1 << k;
  tbl = malloc(sizeof(int) * num_entries);

  for (i=0; i < num_entries; i++)
    tbl[i] = (int) PyLong_AsLong (PyList_GetItem(py_tbl, i));

  final_p_value = comet_exact_test(k, N, tbl, &final_num_tbls, pvalthresh, 0);
  result = Py_BuildValue("id", final_num_tbls, final_p_value);

  free(tbl);
  return result;

}

/* The binomial test, callable from the Python driver. */
PyObject *py_binomial_test(PyObject *self, PyObject *args){
  // Parameters
  int k, N; // k: gene set size; N: number of samples
  PyObject *py_tbl, *result; // FLAT Python contingency table
  int *tbl; // C contingency table
  double pvalthresh;

  // Computation variables
  double final_p_value;

  int num_entries, i; // Helper variables

  // Parse parameters
  if (! PyArg_ParseTuple( args, "iiO!d", &k, &N, &PyList_Type, &py_tbl, &pvalthresh ))
    return NULL;

  // Convert PyList into C array
  num_entries = 1 << k;
  tbl = malloc(sizeof(int) * num_entries);

  for (i=0; i < num_entries; i++)
    tbl[i] = (int) PyLong_AsLong (PyList_GetItem(py_tbl, i));

  final_p_value = comet_binomial_test(k, N, tbl, pvalthresh);
  result = Py_BuildValue("d", final_p_value);

  free(tbl);
  return result;

}

/* load precomuted gene set score from file */
PyObject *py_load_precomputed_scores(PyObject *self, PyObject *args){
  // Parameters
  double score;
  PyObject *py_set; // FLAT Python contingency table
  int size, func, i; // Helper variables

  // Parse parameters
  if (! PyArg_ParseTuple( args, "diiO!", &score, &size, &func, &PyList_Type, &py_set))
    return NULL;

  geneset_t *key;
  weight_t *w;
  // Generate the key by sorting the arr
  key = malloc( sizeof(geneset_t) );
  memset(key, 0, sizeof(geneset_t));
  for (i = 0; i < size; i++) key->genes[i] = (int) PyLong_AsLong (PyList_GetItem(py_set, i));
  qsort(key->genes, size, sizeof(int), ascending);

  // Try and find the geneset
  HASH_FIND(hh, weightHash, key, sizeof(geneset_t), w);  /* id already in the hash? */
  if (w == NULL) {
    w = malloc(sizeof(weight_t));
    w->id = *key;
    w->weight = score;
    w->function = func;
    HASH_ADD( hh, weightHash, id, sizeof(geneset_t), w );  /* id: name of key field */
  }
  return Py_BuildValue(""); // returns NULL
}

PyObject *py_comet(PyObject *self, PyObject *args){
  // Parameters
  int i, j, ell, amp, nt, subtype_size;
  int t, step_len, num_iterations, numFrozen, numPatients, numGenes, verbose, use_marginHash;
  //bool **A;
  int **gene_sets;
  int *ks_c, *initial_soln_c;
  double **weights, binom_pvalthreshold, pvalthresh;
  int **functions;
  PyObject *solns, *soln, *val, *geneset, *initial_soln;
  PyObject *patients2mutatedGenes, *gene2numMutations, *ks;
  /* Parse Python arguments */
  if (! PyArg_ParseTuple( args, "iiiO!O!O!iiiidO!idii", &t, &numGenes, &numPatients,
                          &PyList_Type, &patients2mutatedGenes, &PyList_Type, &gene2numMutations, &PyList_Type, &ks,
                          &num_iterations, &step_len, &amp, &nt, &binom_pvalthreshold, &PyList_Type, &initial_soln,
                          &subtype_size, &pvalthresh, &verbose, &use_marginHash)) {
    return NULL;
  }
  //amp, subt, p_iter, nt, hybrid_pvalthreshold

  mutation_data_t *A = mut_data_allocate(numPatients, numGenes);
  init_mutation_data (A, numPatients, numGenes, patients2mutatedGenes, gene2numMutations);

  ks_c = malloc(sizeof(int) * t);
  for (i = 0; i < t; i++){
    ks_c[i] = (int) PyLong_AsLong(PyList_GetItem(ks, i));
  }

  initial_soln_c = malloc(sizeof(int) * PyList_Size(initial_soln));
  for (i = 0; i < PyList_Size(initial_soln); i++){
    initial_soln_c[i] = (int) PyLong_AsLong(PyList_GetItem(initial_soln, i));
  }

  // Allocate memory for the gene sets
  numFrozen = num_iterations / step_len;
  gene_sets = malloc(sizeof(int *) * numFrozen);
  weights = malloc(sizeof(double *) * numFrozen);
  functions = malloc(sizeof(int *) * numFrozen);
  for (i=0; i < numFrozen; i++){
    gene_sets[i] = malloc(sizeof(int) * sum_array(ks_c, 0, t) );
    weights[i] = malloc(sizeof(double) * t);
    functions[i] = malloc(sizeof(int) * t);
  }

  int *group_index_sum;
  group_index_sum = malloc(sizeof(int) * t);
  for (i=0; i < t; i++){
    group_index_sum[i] = sum_array(ks_c, 0, i);

  }
  // Run the MCMC
  printf("Start runnning MCMC in C... \n");
  double time_cost = comet_mcmc(A, numGenes, numPatients, ks_c, t, num_iterations, step_len, 
                                amp, nt, binom_pvalthreshold, initial_soln_c, PyList_Size(initial_soln), 
                                subtype_size, pvalthresh, gene_sets, weights, functions, verbose, use_marginHash);
  printf("End runnning MCMC in C... \n");

  // Convert the gene sets identified into Python objects
  solns = PyList_New(numFrozen);
  for (i = 0; i < numFrozen; i++){
    soln = PyList_New(t);
    for (j = 0; j < t; j++ ){

      geneset = PyList_New(ks_c[j]+2); // plus tw for weight and function
      for (ell = 0; ell < ks_c[j]; ell++){
        val = PyLong_FromLong((long) gene_sets[i][group_index_sum[j] + ell]);
        PyList_SET_ITEM(geneset, ell, val);
      }
      val = PyFloat_FromDouble(weights[i][j]);
      PyList_SET_ITEM(geneset, ks_c[j], val);
      PyList_SET_ITEM(geneset, ks_c[j]+1, PyLong_FromLong((long)functions[i][j]));
      PyList_SET_ITEM(soln, j, geneset);

    }
    PyList_SET_ITEM(solns, i, soln);
  }
  // Free memory
  mut_data_free(A);
  free(group_index_sum);
  free(ks_c);
  free(initial_soln_c);

  for (i = 0; i < numFrozen; i++){
    free(gene_sets[i]);
    free(weights[i]);
    free(functions[i]);
  }

  // Return results!
  return Py_BuildValue("Od", solns, time_cost);
}

/******************************************************************************
* Export functions to python
******************************************************************************/
PyMethodDef CoMEtMethods[] = {
    {"exhaustive", py_exhaustive, METH_VARARGS, ""},
    {"set_weight", py_set_weight, METH_VARARGS, ""},
    {"exact_test", py_exact_test, METH_VARARGS, "Computes Dendrix++ exact test."},
    {"exact_test_v2", py_exact_test_v2, METH_VARARGS, "Computes Dendrix++ exact test."},
    {"binom_test", py_binomial_test, METH_VARARGS, "Computes Dendrix++ exact test."},
    {"permutation_test", py_permutation_test, METH_VARARGS, "Computes Dendrix++ exact test."},
    {"comet_phi", py_comet_phi, METH_VARARGS, "Computes comet score (phi) from three tests."},
    {"comet", py_comet, METH_VARARGS, "Computes Dendrix++ in MCMC."},
    {"precompute_factorials", py_precompute_factorials, METH_VARARGS, "Precomputes factorials for 0...N"},
    {"load_precomputed_scores", py_load_precomputed_scores, METH_VARARGS, "Loading precomputed scores from file"},
    {"free_factorials", free_factorials, METH_NOARGS, "Frees memory used for factorials"},
    {"set_random_seed", py_set_random_seed, METH_VARARGS, "Set the C PRNG seed."},
    {NULL, NULL, 0, NULL},
};

/*
PyMODINIT_FUNC initcComet(void) {
    PyObject *m = Py_InitModule("cComet", CoMEtMethods);
    if (m == NULL) {
        return;
    }
}
*/
struct PyModuleDef cComet =
{
    PyModuleDef_HEAD_INIT,
    "cComet", /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,   /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    CoMEtMethods
};

PyMODINIT_FUNC PyInit_cComet(void)
{
    return PyModule_Create(&cComet);
}
