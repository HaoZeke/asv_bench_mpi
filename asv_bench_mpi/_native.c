/*
 * asv_bench_mpi._native — free-threaded CPython extension for launching
 * compiled C/Fortran work with the GIL released on the hot path.
 *
 * Heap types via PyType_FromSpec; module declares Py_MOD_GIL_NOT_USED when
 * the CPython headers provide Py_mod_gil.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <dlfcn.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/* ------------------------------------------------------------------ */
/* timing helper                                                      */
/* ------------------------------------------------------------------ */

static double
mono_seconds(void)
{
    struct timespec ts;
#if defined(CLOCK_MONOTONIC)
    clock_gettime(CLOCK_MONOTONIC, &ts);
#else
    clock_gettime(CLOCK_REALTIME, &ts);
#endif
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

typedef double (*asv_kernel_fn)(long);

/* ------------------------------------------------------------------ */
/* NativeKernel heap type                                             */
/* ------------------------------------------------------------------ */

typedef struct {
    PyObject_HEAD
    void *handle;
    asv_kernel_fn fn;
    char *so_path;
    char *symbol;
} NativeKernelObject;

typedef struct {
    PyObject *NativeKernel_Type;
} native_state;

#define NATIVE_STATE(m) ((native_state *)PyModule_GetState(m))

static int
NativeKernel_clear(PyObject *self)
{
    return 0;
}

static int
NativeKernel_traverse(PyObject *self, visitproc visit, void *arg)
{
    return 0;
}

static void
NativeKernel_finalize(PyObject *self)
{
    NativeKernelObject *op = (NativeKernelObject *)self;
    if (op->handle) {
        dlclose(op->handle);
        op->handle = NULL;
        op->fn = NULL;
    }
    free(op->so_path);
    op->so_path = NULL;
    free(op->symbol);
    op->symbol = NULL;
}

static void
NativeKernel_dealloc(PyObject *self)
{
    PyObject_GC_UnTrack(self);
    NativeKernel_finalize(self);
    {
        PyTypeObject *tp = Py_TYPE(self);
        freefunc free_fn = (freefunc)PyType_GetSlot(tp, Py_tp_free);
        free_fn(self);
        Py_DECREF(tp);
    }
}

static PyObject *
NativeKernel_new(PyTypeObject *type, PyObject *args, PyObject *kw)
{
    allocfunc alloc = (allocfunc)PyType_GetSlot(type, Py_tp_alloc);
    NativeKernelObject *self = (NativeKernelObject *)alloc(type, 0);
    if (self == NULL)
        return NULL;
    self->handle = NULL;
    self->fn = NULL;
    self->so_path = NULL;
    self->symbol = NULL;
    return (PyObject *)self;
}

static int
NativeKernel_init(PyObject *self, PyObject *args, PyObject *kw)
{
    NativeKernelObject *op = (NativeKernelObject *)self;
    const char *so_path = NULL;
    const char *symbol = NULL;
    static char *kwlist[] = {"so_path", "symbol", NULL};
    void *handle;
    void *sym;
    char *err;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "ss:NativeKernel", kwlist,
                                     &so_path, &symbol))
        return -1;

    /* Close any previous load (re-init). */
    NativeKernel_finalize(self);

    handle = dlopen(so_path, RTLD_NOW | RTLD_LOCAL);
    if (handle == NULL) {
        PyErr_Format(PyExc_OSError, "dlopen(%s): %s", so_path, dlerror());
        return -1;
    }
    dlerror();
    sym = dlsym(handle, symbol);
    err = dlerror();
    if (err != NULL || sym == NULL) {
        dlclose(handle);
        PyErr_Format(PyExc_OSError, "dlsym(%s, %s): %s", so_path, symbol,
                     err ? err : "NULL");
        return -1;
    }

    op->handle = handle;
    op->fn = (asv_kernel_fn)sym;
    op->so_path = strdup(so_path);
    op->symbol = strdup(symbol);
    if (op->so_path == NULL || op->symbol == NULL) {
        NativeKernel_finalize(self);
        PyErr_NoMemory();
        return -1;
    }
    return 0;
}

/* call(n=0) -> float : invoke kernel with GIL released */
static PyObject *
NativeKernel_call(PyObject *self, PyObject *args, PyObject *kw)
{
    NativeKernelObject *op = (NativeKernelObject *)self;
    long n = 0;
    double result;
    static char *kwlist[] = {"n", NULL};
    asv_kernel_fn fn;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "|l:call", kwlist, &n))
        return NULL;
    if (op->fn == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "NativeKernel not initialized");
        return NULL;
    }
    fn = op->fn;
    Py_BEGIN_ALLOW_THREADS
    result = fn(n);
    Py_END_ALLOW_THREADS
    return PyFloat_FromDouble(result);
}

/* time(n=0) -> float : wall seconds of one kernel call, GIL released */
static PyObject *
NativeKernel_time(PyObject *self, PyObject *args, PyObject *kw)
{
    NativeKernelObject *op = (NativeKernelObject *)self;
    long n = 0;
    double t0, t1, result;
    static char *kwlist[] = {"n", NULL};
    asv_kernel_fn fn;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "|l:time", kwlist, &n))
        return NULL;
    if (op->fn == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "NativeKernel not initialized");
        return NULL;
    }
    fn = op->fn;
    Py_BEGIN_ALLOW_THREADS
    t0 = mono_seconds();
    result = fn(n);
    t1 = mono_seconds();
    (void)result;
    Py_END_ALLOW_THREADS
    return PyFloat_FromDouble(t1 - t0);
}

static PyObject *
NativeKernel_repr(PyObject *self)
{
    NativeKernelObject *op = (NativeKernelObject *)self;
    return PyUnicode_FromFormat("NativeKernel(%s, %s)",
                                op->so_path ? op->so_path : "?",
                                op->symbol ? op->symbol : "?");
}

static PyMethodDef NativeKernel_methods[] = {
    {"call", (PyCFunction)NativeKernel_call, METH_VARARGS | METH_KEYWORDS,
     "call(n=0) -> float\n\nCall double (*)(long) with the GIL released."},
    {"time", (PyCFunction)NativeKernel_time, METH_VARARGS | METH_KEYWORDS,
     "time(n=0) -> float\n\nWall seconds for one call; GIL released."},
    {NULL, NULL, 0, NULL}
};

static PyType_Slot NativeKernel_Type_slots[] = {
    {Py_tp_dealloc, (void *)NativeKernel_dealloc},
    {Py_tp_traverse, (void *)NativeKernel_traverse},
    {Py_tp_clear, (void *)NativeKernel_clear},
    {Py_tp_finalize, (void *)NativeKernel_finalize},
    {Py_tp_methods, NativeKernel_methods},
    {Py_tp_init, (void *)NativeKernel_init},
    {Py_tp_new, (void *)NativeKernel_new},
    {Py_tp_repr, (void *)NativeKernel_repr},
    {0, 0},
};

static PyType_Spec NativeKernel_Type_spec = {
    "asv_bench_mpi._native.NativeKernel",
    sizeof(NativeKernelObject),
    0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_HEAPTYPE,
    NativeKernel_Type_slots
};

/* ------------------------------------------------------------------ */
/* run_executable(argv, env=None) -> float                            */
/* ------------------------------------------------------------------ */

static PyObject *
native_run_executable(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject *argv_obj = NULL;
    PyObject *env_obj = Py_None;
    static char *kwlist[] = {"argv", "env", NULL};
    Py_ssize_t n, i;
    char **argv = NULL;
    double t0, t1;
    pid_t pid;
    int status;
    int rc = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "O|O:run_executable", kwlist,
                                     &argv_obj, &env_obj))
        return NULL;
    if (!PyList_Check(argv_obj) && !PyTuple_Check(argv_obj)) {
        PyErr_SetString(PyExc_TypeError, "argv must be a list or tuple of str");
        return NULL;
    }
    n = PySequence_Size(argv_obj);
    if (n < 1) {
        PyErr_SetString(PyExc_ValueError, "argv must be non-empty");
        return NULL;
    }
    argv = (char **)PyMem_Calloc((size_t)n + 1, sizeof(char *));
    if (argv == NULL)
        return PyErr_NoMemory();
    for (i = 0; i < n; ++i) {
        PyObject *item = PySequence_GetItem(argv_obj, i);
        const char *s;
        if (item == NULL)
            goto fail;
        s = PyUnicode_AsUTF8(item);
        Py_DECREF(item);
        if (s == NULL)
            goto fail;
        argv[i] = (char *)s; /* borrowed from Unicode — valid until args die */
    }
    argv[n] = NULL;

    if (env_obj != Py_None && !PyDict_Check(env_obj)) {
        PyErr_SetString(PyExc_TypeError, "env must be a dict or None");
        goto fail;
    }

    /* Note: custom env dict not applied in v1 (use os.environ from Python).
     * Wall-clock wait runs with the GIL released. */
    t0 = mono_seconds();
    Py_BEGIN_ALLOW_THREADS
    pid = fork();
    if (pid == 0) {
        execvp(argv[0], argv);
        _exit(127);
    } else if (pid < 0) {
        rc = -1;
    } else {
        if (waitpid(pid, &status, 0) < 0)
            rc = -2;
        else if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
            rc = 1;
    }
    t1 = mono_seconds();
    Py_END_ALLOW_THREADS

    if (rc == -1) {
        PyErr_SetFromErrno(PyExc_OSError);
        goto fail;
    }
    if (rc == -2) {
        PyErr_SetFromErrno(PyExc_OSError);
        goto fail;
    }
    if (rc == 1) {
        PyErr_Format(PyExc_RuntimeError,
                     "executable exited non-zero: %s", argv[0]);
        goto fail;
    }
    PyMem_Free(argv);
    return PyFloat_FromDouble(t1 - t0);

fail:
    PyMem_Free(argv);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* extension_flags()                                                  */
/* ------------------------------------------------------------------ */

static PyObject *
native_extension_flags(PyObject *self, PyObject *Py_UNUSED(args))
{
    PyObject *d = PyDict_New();
    int gil_not_used = 0;
    int heap_type = 1;
#ifdef Py_mod_gil
    gil_not_used = 1;
#endif
    if (d == NULL)
        return NULL;
    if (PyDict_SetItemString(d, "gil_not_used", gil_not_used ? Py_True : Py_False) < 0)
        goto err;
    if (PyDict_SetItemString(d, "heap_type", heap_type ? Py_True : Py_False) < 0)
        goto err;
    if (PyDict_SetItemString(d, "py_begin_allow_threads", Py_True) < 0)
        goto err;
    if (PyDict_SetItemString(d, "api", PyUnicode_FromString("phase-a-native-launch")) < 0)
        goto err;
    return d;
err:
    Py_DECREF(d);
    return NULL;
}

static PyMethodDef module_methods[] = {
    {"run_executable", (PyCFunction)native_run_executable,
     METH_VARARGS | METH_KEYWORDS,
     "run_executable(argv, env=None) -> float\n\n"
     "Spawn argv[0], wait, return wall seconds. GIL released during wait."},
    {"extension_flags", (PyCFunction)native_extension_flags, METH_NOARGS,
     "extension_flags() -> dict of build/capability markers."},
    {NULL, NULL, 0, NULL}
};

/* ------------------------------------------------------------------ */
/* module lifecycle                                                   */
/* ------------------------------------------------------------------ */

static int
native_modexec(PyObject *m)
{
    native_state *state = NATIVE_STATE(m);
    state->NativeKernel_Type = PyType_FromSpec(&NativeKernel_Type_spec);
    if (state->NativeKernel_Type == NULL)
        return -1;
    /* Keep a state-owned ref; AddObject steals one. */
    Py_INCREF(state->NativeKernel_Type);
    if (PyModule_AddObject(m, "NativeKernel", state->NativeKernel_Type) < 0) {
        Py_DECREF(state->NativeKernel_Type);
        Py_CLEAR(state->NativeKernel_Type);
        return -1;
    }
    return 0;
}

static int
native_traverse(PyObject *module, visitproc visit, void *arg)
{
    native_state *state = NATIVE_STATE(module);
    Py_VISIT(state->NativeKernel_Type);
    return 0;
}

static int
native_clear(PyObject *module)
{
    native_state *state = NATIVE_STATE(module);
    Py_CLEAR(state->NativeKernel_Type);
    return 0;
}

static void
native_free(void *module)
{
    (void)native_clear((PyObject *)module);
}

static PyModuleDef_Slot native_slots[] = {
    {Py_mod_exec, (void *)native_modexec},
#ifdef Py_mod_gil
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, NULL}
};

PyDoc_STRVAR(module_doc,
"Native launch helpers for asv_bench_mpi (GIL-released C/Fortran calls).");

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "asv_bench_mpi._native",
    module_doc,
    sizeof(native_state),
    module_methods,
    native_slots,
    native_traverse,
    native_clear,
    native_free
};

PyMODINIT_FUNC
PyInit__native(void)
{
    return PyModuleDef_Init(&moduledef);
}
